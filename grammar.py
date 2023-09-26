from enum import IntEnum
from typing import Tuple, List, Dict, Union, Optional, Set
from extensions.output_template.utils import encode, decode, shared, get_token_dictionary
import re

RE_RULE = re.compile(r'\s*([-a-z]+)\s*::=\s*(.*)', re.MULTILINE | re.DOTALL)
RE_NEWLINE = re.compile(r'[ \t]*\n[ \t\n]*(.*)', re.MULTILINE | re.DOTALL)
RE_TERMINAL = re.compile(r'[ \t]*([-a-z]+)[ \t]*(.*)', re.DOTALL)
RE_OR = re.compile(r'[ \t\n]*\|[ \t]*(.*)', re.MULTILINE | re.DOTALL)
MINUS_INF = -float("inf")


class Grammar:
    """ Grammar and also state machine used to match LLM output """

    def __init__(self, definition: str):
        self.rules: Dict[str, Symbol] = {}
        self.active_symbol: Optional[Symbol] = None
        self.reset(definition)

    def stop(self):
        self.active_symbol = None

    def reset(self, definition: str = None):
        self.stop()
        if definition:
            text = definition

            while text:
                m = RE_RULE.match(text)
                if not m:
                    raise GrammarError("expected rule")
                rule_name, text = m.groups()
                self.rules[rule_name], text = parse_rule(text)

            if "root" not in self.rules:
                raise ValidationError("missing 'root' rule")
            self.rules["root"].validate(self)

        self.enter_rule("root")

    def resolve(self, symbol: "Symbol") -> "Symbol":
        # Resolves NonTerminal into rule and returns Symbol it represents
        dont_loop: Set[Terminal] = set([symbol])
        while isinstance(symbol, NonTerminal):
            dont_loop.add(symbol)
            if symbol.name not in self.rules:
                raise ValidationError(f"invalid rule name: '{symbol.name}'")
            if self.rules[symbol.name] in dont_loop:
                raise ValidationError(f"infinite loop detected at symbol '{symbol.name}'")
            symbol = self.rules[symbol.name]
        return symbol

    def enter_rule(self, name: str):
        """
        Sets active symbol to specific rule.
        Used for testing.
        """
        if name not in self.rules:
            raise ValueError(f"invalid rule name: '{name}'")
        self.active_symbol = self.resolve(self.rules[name])
        self.active_symbol.reset(self)
        print(f"active symbol set to {self.active_symbol}")

    def get_rule_name(self, symbol: "Symbol"):
        for (name, s) in self.rules.items():
            if symbol is s:
                return name
        return None

    def get_effective_symbol(self) -> Optional["Symbol"]:
        """
        Recursively descends rule hierarchy and returns symbol
        that will effectively decide on next token
        """
        return self.active_symbol.get_effective_symbol() if self.active_symbol else None

    def update_scores(self, scores: "Tensor") -> List[int]:
        """
        Calculates probability scores of next token according to current state.
        May update and return same object as one that was passed as argument.
        """
        # TODO: how to cache or optimize this?
        if self.active_symbol:
            d = get_token_dictionary()
            for token_id in d:
                a = self.active_symbol.match(token_id)
                if a == Match.Reject:
                    scores[..., token_id] = MINUS_INF
                elif a == Match.TryNext and token_id != shared.tokenizer.eos_token_id:
                    scores[..., token_id] = MINUS_INF
        else:
            # Grammar reached terminal token. Force EOS
            eos_token_id = int(shared.tokenizer.eos_token_id)
            scores[0] = scores[0].new_full(scores[0].shape, -float("inf"))
            scores[..., eos_token_id] = 1000.0
        return scores

    def advance(self, token_id: int):
        try:
            if self.active_symbol:
                from extensions.output_template.script import logger
                logger.warning(f"Feeding {token_id} into {self.active_symbol}.")
                a = self.active_symbol.advance(self, token_id)
                if a == Advance.Reject:
                    if token_id == shared.tokenizer.eos_token_id:
                        self.active_symbol = None
                    else:
                        raise GenerationError
                elif a == Advance.Next:
                    self.active_symbol = None
        except GenerationError as e:
            from extensions.output_template.script import logger
            logger.warning("LLM failed to generate token conforming to grammar")
            self.active_symbol = None


class GrammarError(ValueError):
    pass


class ValidationError(ValueError):
    pass


class GenerationError(Exception):
    pass


class Match(IntEnum):
    """ Glorious tree-state boolean """
    Reject = 0
    Accept = 1
    TryNext = 2


class Advance(IntEnum):
    Again = 0
    Next = 1
    Reject = 2


class Symbol:
    def validate(self, g: Grammar):
        """
        Validates against grammar.
        Currently just checks that for each Terminal there is rule defined.
        """
        pass

    def reset(self, g: Grammar):
        """
        Resets internal state, if any.
        Called upon entering symbol.
        """
        pass

    def debug(self) -> str:
        """
        Returns string describing current state.
        There's no logic here, just what seem'd most readable at the time
        """
        return repr(self)

    def get_effective_symbol(self) -> "Symbol":
        return self

    def advance(self, g: Grammar, token_id: int) -> Advance:
        """
        Returns Advance.Again if given token matches symbol only partially and more tokens are expected.
        Returns Advance.Next if given token (at current state) matches rest of this symbol.
        Returns Advance.Reject if token doesn't match at all and next symbol should be retried with same token.
        May raise GenerationError if token doesn't match when it definitely should.
        """
        raise NotImplementedError

    def match(self, token_id: int) -> Match:
        raise NotImplementedError(f"match on {self.__class__.__name__}")


class NonTerminal(Symbol):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def validate(self, g: Grammar):
        r = g.resolve(self)
        if r is self:
            raise GrammarError(f"'{self.name}' cannot directly reference itself")
        r.validate(g)

    def reset(self, g: Grammar):
        raise TypeError("NonTerminal cannot be reset(). Follow grammar.rules[self.name]")

    def match(self, token_id: int) -> Match:
        raise TypeError("NonTerminal cannot match(). Follow grammar.rules[self.name]")

    def advance(self, g: Grammar, token_id: int) -> bool:
        raise TypeError("NonTerminal cannot advance(). Follow grammar.rules[self.name]")


class Terminal(Symbol):
    def __init__(self, value: str):
        self.value = value
        self.index = 0

    def __repr__(self):
        return f't{repr(self.value)}'

    def debug(self) -> str:
        if self.index <= 0 or self.index >= len(self.value):
            return repr(self)
        return f"""t'{repr(self.value[:self.index]).strip("'")}[{repr(self.value[self.index:]).strip("'")}]'"""

    def reset(self, g: Grammar):
        self.index = 0

    def match(self, token_id: int) -> Match:
        if token_id == shared.tokenizer.eos_token_id:
            return Match.Reject
        token_string = get_token_dictionary()[token_id]
        return Match(self.value[self.index:self.index + len(token_string)] == token_string)

    def advance(self, g: Grammar, token_id: int) -> Advance:
        if not self.match(token_id):
            raise GenerationError
        self.index += len(get_token_dictionary()[token_id])
        return Advance.Next if self.index >= len(self.value) else Advance.Again


class RegExp(Symbol):
    def __init__(self, value: str):
        self.value = value
        self.re = re.compile("^" + value + "$", re.MULTILINE | re.DOTALL)

    def __repr__(self):
        return f'r{self.value}'

    def match(self, token_id: int) -> Match:
        if token_id == shared.tokenizer.eos_token_id:
            return Match.Reject
        token_string = get_token_dictionary()[token_id]
        return Match(bool(self.re.match(token_string)))

    def advance(self, g: Grammar, token_id: int) -> Advance:
        if not self.match(token_id):
            raise Advance.Reject
        return Advance.Next


class Collection(Symbol):
    def __init__(self, items: List[Symbol]):
        self.items = items

    def validate(self, g: Grammar):
        for i in self.items:
            i = g.resolve(i)
            if i is self:
                raise GrammarError(f"'{g.get_rule_name(self) or self}' cannot directly reference itself")
            elif isinstance(i, Alternative):
                if self in (g.resolve(j) for j in i.items):
                    raise GrammarError(f"{g.get_rule_name(self) or self} -> {i} cannot indirectly reference itself")
            i.validate(g)


class Sequence(Collection):
    def __init__(self, items: List[Symbol]):
        super().__init__(items)
        self.effective = []
        self.index = 0

    def __repr__(self):
        return f'({" ".join([repr(x) for x in self.items ])})'

    def debug(self) -> str:
        return f'''({" ".join([
            f"[{self.items[i].debug()}]" if i == self.index
            else self.items[i].debug()
            for i in range(len(self.items))
        ])})'''

    def get_effective_symbol(self) -> "Symbol":
        if self.index == len(self.effective) - 1:
            return self.effective[self.index].get_effective_symbol()
        return self

    def reset(self, g: Grammar):
        self.effective = [g.resolve(s) for s in self.items]
        self.index = 0
        if self.effective:
            self.effective[0].reset(g)

    def next_symbol(self) -> Optional[Symbol]:
        self.index += 1
        if self.index >= len(self.effective):
            return None
        return self.effective[self.index]

    def match(self, token_id: int) -> Match:
        if self.index < len(self.effective):
            m = self.effective[self.index].match(token_id)
            if m == Match.TryNext:
                i = self.index + 1
                while m == Match.TryNext and i < len(self.effective):
                    m = self.effective[i].match(token_id)
                    i += 1
            return m
        return Match.Reject

    def advance(self, g: Grammar, token_id: int) -> Advance:
        if self.index < len(self.effective):
            a = self.effective[self.index].advance(g, token_id)
            if a == Advance.Next:
                # TODO: why is reseting item that is just getting left necessary?
                self.effective[self.index].reset(g)
                self.index += 1
                if self.index >= len(self.effective):
                    return Advance.Next
                self.effective[self.index].reset(g)
                return Advance.Again
            elif a == Advance.Reject:
                # TODO: why is reseting item that is just getting left necessary?
                self.effective[self.index].reset(g)
                self.index += 1
                if self.index >= len(self.effective):
                    return Advance.Reject
                self.effective[self.index].reset(g)
                # if self.effective[self.index].match(token_id) != Match.Accept:
                #     return Advance.Reject
                return self.advance(g, token_id)
            return a
        return Advance.Next


class Alternative(Collection):
    def __init__(self, items: List[Symbol]):
        super().__init__([])
        for i in items:
            if isinstance(i, Alternative):
                self.items += i.items
            else:
                self.items.append(i)
        self.possible = set()

    def __repr__(self):
        return f'({" | ".join([repr(x) for x in self.items ])})'

    def debug(self):
        return f'({" | ".join([x.debug() for x in self.possible ])})'

    def get_effective_symbol(self) -> "Symbol":
        if len(self.possible) == 1:
            return list(self.possible)[0].get_effective_symbol()
        return self

    def reset(self, g: Grammar):
        self.possible = {g.resolve(s) for s in self.items}
        for s in self.possible:
            s.reset(g)

    def match(self, token_id: str) -> Match:
        for s in self.possible:
            if s.match(token_id):
                return Match.Accept
        return Match.Reject

    def advance(self, g: Grammar, token_id: int) -> Advance:
        any_ok = False
        for s in list(self.possible):
            if s.match(token_id) != Match.Reject:
                a = s.advance(g, token_id)
                any_ok = True
                if a in (Advance.Next, Advance.Reject):
                    self.possible.remove(s)
            else:
                self.possible.remove(s)
        if len(self.possible) == 0:
            if any_ok:
                return Advance.Next
            raise GenerationError
        return Advance.Again


class Repeat(Symbol):

    class Mode(IntEnum):
        OneOrNone = 1   # '?'
        ManyOrNone = 2  # '*'
        OneOrMore = 3   # '+'
        ''' ThisThenNone - OneOrNone ('?') after item accepted 1st token '''
        ThisThenNone = 4
        ''' ThisThenMore - '+' and '*' after item accepted 1st token '''
        ThisThenMore = 5

    def __init__(self, mode: str, item: Symbol):
        if isinstance(item, Repeat):
            item = item.item
        self.item = item
        assert mode in "+*?"
        self.mode = mode
        self.effective_item = None
        self.effective_mode = Repeat.Mode.OneOrMore

    def __repr__(self):
        if isinstance(self.item, (Terminal, NonTerminal)):
            return f'({repr(self.item)}){self.mode}'
        else:
            return f'{repr(self.item)}{self.mode}'

    def reset(self, g: Grammar):
        if self.mode == "+":
            self.effective_mode = Repeat.Mode.OneOrMore
        elif self.mode == "?":
            self.effective_mode = Repeat.Mode.OneOrNone
        else:  # self.mode == "*"
            self.effective_mode = Repeat.Mode.ManyOrNone
        self.effective_item = g.resolve(self.item)
        self.effective_item.reset(g)

    def validate(self, g: Grammar):
        self.item.validate(g)

    def match(self, token_id: int) -> Match:
        m = self.effective_item.match(token_id)
        if m == Match.Reject and self.effective_mode <= Repeat.Mode.OneOrMore:
            # if token_id == int(shared.tokenizer.eos_token_id):
            #     if self.effective_mode <= Repeat.Mode.ManyOrNone:
            #         return Match.TryNext
            #     return Match.Reject
            if self.effective_mode <= Repeat.Mode.ManyOrNone:
                # Zero or more. Rejection means continue on next
                return Match.TryNext
        return m

    def advance(self, g: Grammar, token_id: int) -> Advance:
        m = self.match(token_id)
        if m == Match.Accept:
            a = self.effective_item.advance(g, token_id)
            if a == Advance.Reject:
                raise GenerationError
            elif a == Advance.Next:
                if self.effective_mode in (Repeat.Mode.OneOrMore, Repeat.Mode.ManyOrNone, Repeat.Mode.ThisThenMore):
                    self.effective_mode = Repeat.Mode.ManyOrNone
                if self.effective_mode in (Repeat.Mode.OneOrNone, Repeat.Mode.ThisThenNone):
                    return Advance.Next
                self.effective_item.reset(g)
                return Advance.Again
            else:
                if self.effective_mode in (Repeat.Mode.OneOrMore, Repeat.Mode.ManyOrNone):
                    self.effective_mode = Repeat.Mode.ThisThenMore
                if self.effective_mode == Repeat.Mode.OneOrNone:
                    self.effective_mode = Repeat.Mode.ThisThenNone
            return a
        elif m == Match.Reject:
            raise GenerationError
        elif m == Match.TryNext:
            if self.effective_mode <= Repeat.Mode.ManyOrNone:
                return Advance.Reject
            raise GenerationError
        return Advance.Reject


def find_unescaped_index(haystack: str, needle: str, start=0) -> int:
    index = start
    while True:
        index = haystack.index(needle, index)
        if haystack[index - 1] != "\\":
            return index
        index += 1


def parse_sequence(text: str, parentheses=False) -> Tuple[Sequence, str]:
    seq = []
    while text:
        if text[0] in '"\'':
            # Terminal rule
            try:
                end_index = find_unescaped_index(text, text[0], 1)
                t = text[1:end_index].encode("utf-8").decode("unicode_escape")
                seq.append(Terminal(t))
                text = text[end_index+1:]
            except ValueError:
                raise GrammarError(f"unmatched {text[0]}")
        elif RE_TERMINAL.match(text):
            # Non-terminal rule
            t, text = RE_TERMINAL.match(text).groups()
            seq.append(NonTerminal(t))
        elif text[0] in " \t":
            # Whitespace
            text = text[1:]
        elif text[0] == "[":
            # Regexp rule
            try:
                end_index = find_unescaped_index(text, "]", 1)
            except ValueError:
                raise GrammarError(f"unmatched {text[0]}")
            try:
                seq.append(RegExp(text[0:end_index+1]))
                text = text[end_index+1:]
            except ValueError:
                raise GrammarError(f"invalid pattern {text[0:end_index]}")
        elif text[0] == "(":
            # Parenthesized rule
            text = text[1:]
            t, text = parse_rule(text, parentheses=True)
            seq.append(t)
            pass
        elif parentheses and text[0] == ")":
            text = text[1:]
            break
        elif text[0] in "+*?":
            # Repeat rule
            if not seq:
                raise GrammarError(f"unexpected '{text[0]}'")
            left = seq.pop()
            if text[0] == "+" and isinstance(left, RegExp):
                seq.append(Repeat(text[0], RegExp(left.value + text[0])))
            else:
                seq.append(Repeat(text[0], left))
            text = text[1:]
        elif RE_OR.match(text):
            text, = RE_OR.match(text).groups()
            if not seq:
                raise GrammarError(f"unexpected '|'")
            left = seq.pop()
            right, text = parse_rule(text, parentheses=parentheses)
            seq.append(Alternative([left, right]))
            break
        elif RE_NEWLINE.match(text):
            # Newline
            text, = RE_NEWLINE.match(text).groups()
            if not parentheses:
                break
        else:
            raise GrammarError(f"unexpected '{text[0:5]}'...")

    return Sequence(seq), text


def parse_rule(text: str, parentheses=False) -> Tuple[Symbol, str]:
    rv, text = parse_sequence(text, parentheses=parentheses)
    if len(rv.items) == 1:
        rv = rv.items[0]
    return rv, text
