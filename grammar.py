from typing import Tuple, List, Dict, Union, Optional, Set
from extensions.output_template.symbols import Symbol, Terminal, NonTerminal, Sequence, Alternative, Repeat, RegExp
from extensions.output_template.state_machine import GenerationError, Advance, Matcher
from extensions.output_template.utils import encode, decode, shared, AllowedTokens
import re

RE_RULE = re.compile(r'\s*([-a-z]+)\s*::=\s*(.*)', re.MULTILINE | re.DOTALL)
RE_NEWLINE = re.compile(r'[ \t]*\n[ \t\n]*(.*)', re.MULTILINE | re.DOTALL)
RE_TERMINAL = re.compile(r'[ \t]*([-a-z]+)[ \t]*(.*)', re.DOTALL)
RE_OR = re.compile(r'[ \t\n]*\|[ \t]*(.*)', re.MULTILINE | re.DOTALL)
RE_COMMENT = re.compile(r'([^#]*)#[^\n]*(.*)', re.MULTILINE | re.DOTALL)


class Grammar:
    """ Grammar and also state machine used to match LLM output """

    def __init__(self, definition: str):
        self.rules: Dict[str, Symbol] = {}
        self.active_matcher: Optional[Matcher] = None
        self.reset(definition)

    def stop(self):
        self.active_matcher = None

    def reset(self, definition: str = None):
        self.stop()
        if definition:
            text = definition
            self.rules = {}

            # Strip comments
            m = RE_COMMENT.match(text)
            while m:
                text = m.group(1) + m.group(2)
                m = RE_COMMENT.match(text)

            while text:
                m = RE_RULE.match(text)
                if not m:
                    raise GrammarError("expected rule")
                rule_name, text = m.groups()
                if rule_name in self.rules:
                    raise ValidationError(f"duplicate rule '{rule_name}'")
                self.rules[rule_name], text = parse_rule(text)

            if "root" not in self.rules:
                raise ValidationError("missing 'root' rule")
            for rule in self.rules.values():
                rule.validate(self)

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
        self.active_matcher = self.resolve(self.rules[name]).enter(self)

    def get_rule_name(self, symbol: "Symbol"):
        for (name, s) in self.rules.items():
            if symbol is s:
                return name
        return None

    def get_effective_matcher(self) -> Optional["Matcher"]:
        """
        Recursively descends rule hierarchy and returns symbol
        that will effectively decide on next token
        """
        return self.active_matcher.get_effective_matcher() if self.active_matcher else None

    def update_scores(self, scores: "Tensor") -> List[int]:
        """
        Calculates probability scores of next token according to current state.
        May update and return same object as one that was passed as argument.
        """
        # TODO: how to cache or optimize this?
        if self.active_matcher:
            allowed = self.active_matcher.get_allowed_tokens(self)
            if allowed.look_ahead:
                allowed.allow_eos = True

            allowed.apply(scores)
        else:
            # Grammar reached terminal token. Force EOS
            AllowedTokens(allowed={int(shared.tokenizer.eos_token_id)}, allow_eos=True).apply(scores)
        return scores

    def advance(self, token_id: int):
        try:
            if self.active_matcher:
                # from extensions.output_template.script import logger
                # logger.warning(f"Feeding {token_id} into {self.active_matcher}.")
                a = self.active_matcher.advance(self, token_id)
                if a == Advance.Reject:
                    if token_id == shared.tokenizer.eos_token_id:
                        self.active_matcher = None
                    else:
                        raise GenerationError
                elif a == Advance.Done:
                    self.active_matcher = None
        except GenerationError as e:
            from extensions.output_template.script import logger
            logger.warning("LLM failed to generate token conforming to grammar")
            self.active_matcher = None


class GrammarError(ValueError):
    pass


class ValidationError(GrammarError):
    pass


def find_unescaped_index(haystack: str, needle: str, start=0) -> int:
    index = start
    while True:
        index, index2 = haystack.find(needle, index), haystack.find("\\", index)
        if index < 0:
            return len(haystack)
        if index2 >= 0 and index2 < index:
            index = index2 + 2
        else:
            return index


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
        elif text[0] in "*?+":
            # Repeat rule
            if not seq:
                raise GrammarError(f"unexpected '{text[0]}'")
            left = seq.pop()
            if text[0] == "+":
                # A+ is converted into sequence (A A*)
                if text[0] in "+" and isinstance(left, RegExp):
                    # If child is regexp, extend its rule so multi-character tokens are matched
                    left = RegExp(left.value + text[0])
                seq.append(Sequence([left, Repeat("*", left)]))
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
