from typing import List, Optional, Set
from extensions.output_template.utils import get_token_dictionary, AllowedTokens
from enum import IntEnum


class Advance(IntEnum):
    Again = 0
    Done = 1
    Reject = 2
    TryNext = 3


class Matcher:
    ''' Piece of state machine used to match the Symbol '''

    def __init__(self, symbol: "Symbol"):
        self.symbol = symbol

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.debug()}>"

    def debug(self) -> str:
        """
        Returns string describing current state.
        There's no logic here, just what seem'd most readable at the time
        """
        return repr(self.symbol)

    def get_effective_matcher(self) -> Optional["Matcher"]:
        raise NotImplementedError(f"get_effective_matcher on {self.__class__.__name__}") 

    def advance(self, g: "Grammar", token_id: int) -> Advance:
        """
        Returns Advance.Again if given token matches symbol only partially and more tokens are expected.
        Returns Advance.Done if given token (at current state) matches rest of this symbol.
        Method should not be called again after returning Advance.Done.
        Returns Advance.Reject if token doesn't match
        Returns Advance.TryNext if not matching token is expected and next symbol should be tried
        """
        raise NotImplementedError(f"advance on {self.__class__.__name__}")

    def get_allowed_tokens(self, g: "Grammar") -> AllowedTokens:
        raise NotImplementedError(f"get_allowed_tokens on {self.__class__.__name__}")


class TerminalMatcher(Matcher):
    symbol: "Terminal"

    def __init__(self, t: "Terminal"):
        super().__init__(t)
        self.index = 0

    def debug(self) -> str:
        if self.index <= 0 or self.index >= len(self.symbol.value):
            return repr(self.symbol)
        return f"""t'{repr(self.symbol.value[:self.index]).strip("'")}[{repr(self.symbol.value[self.index:]).strip("'")}]'"""

    def get_effective_matcher(self) -> "Matcher":
        return self

    def get_allowed_tokens(self, g: "Grammar") -> AllowedTokens:
        if self.index not in self.symbol.allowed_cache:
            d = get_token_dictionary()
            allowed = set()
            for token_id in d:
                if d[token_id] and d[token_id] == self.symbol.value[self.index:self.index+len(d[token_id])]:
                    allowed.add(token_id)
            self.symbol.allowed_cache[self.index] = allowed
        return AllowedTokens(allowed=self.symbol.allowed_cache[self.index])

    def advance(self, g: "Grammar", token_id: int) -> Advance:
        d = get_token_dictionary()
        t = d[token_id]
        if t != self.symbol.value[self.index:self.index + len(t)]:
            if self.index == 0:
                # Special case, allow entering mid-token
                t = get_suffix_prefix(t, self.symbol.value)
                if not t:
                    return Advance.Reject
            else:
                return Advance.Reject
        self.index += len(t)
        if self.index >= len(self.symbol.value):
            return Advance.Done
        return Advance.Again


def get_suffix_prefix(suffix_from, prefix_from) -> str:
    i = 1
    while i <= min(len(suffix_from), len(prefix_from)):
        if suffix_from[-i:] != prefix_from[:i]:
            break
        i += 1
    return prefix_from[:i-1]


class RegExpMatcher(Matcher):
    symbol: "RegExp"

    def __init__(self, t: "RegExp"):
        super().__init__(t)

    def debug(self) -> str:
        return f"""r{self.symbol.value}"""

    def get_effective_matcher(self) -> "Matcher":
        return self

    def get_allowed_tokens(self, g: "Grammar") -> AllowedTokens:
        if self.symbol.negative:
            if self.symbol.value not in self.symbol.banned_cache:
                d = get_token_dictionary()
                banned = set()
                for token_id in d:
                    if self.symbol.re.search(d[token_id]):
                        if self.symbol.next:
                            t = d[token_id]
                            # Check if there's prefix of next terminal that is also suffix of this token
                            s = get_suffix_prefix(t, self.symbol.next.value)
                            # If yes, check if rest of this token can be allowed
                            if s and len(s) < len(t) and not self.symbol.re.search(t[0:-len(s)]):
                                # Yes, allow that token
                                pass
                            else:
                                # No, ban entire token
                                banned.add(token_id)
                        else:
                            banned.add(token_id)
                self.symbol.banned_cache[self.symbol.value] = banned
            return AllowedTokens(banned=self.symbol.banned_cache[self.symbol.value])
        else:
            if self.symbol.value not in self.symbol.allowed_cache:
                d = get_token_dictionary()
                allowed = set()
                for token_id in d:
                    if self.symbol.re.match(d[token_id]):
                        allowed.add(token_id)
                self.symbol.allowed_cache[self.symbol.value] = allowed
            return AllowedTokens(allowed=self.symbol.allowed_cache[self.symbol.value])

    def advance(self, g: "Grammar", token_id: int) -> Advance:
        d = get_token_dictionary()
        if self.symbol.negative:
            if self.symbol.re.search(d[token_id]):
                return Advance.Reject
        elif not self.symbol.re.match(d[token_id]):
            return Advance.Reject
        # TODO: use index? How to deal with tokens that match partially?
        return Advance.Done


class AnyTokenMatcher(Matcher):
    symbol: "AnyToken"

    def debug(self) -> str:
        return ".*"

    def get_effective_matcher(self) -> "Matcher":
        return self

    def get_allowed_tokens(self, g: "Grammar") -> AllowedTokens:
        return AllowedTokens(allow_eos=True)

    def advance(self, g: "Grammar", token_id: int) -> Advance:
        return Advance.Again


class SequenceMatcher(Matcher):
    symbol: "Sequence"

    def __init__(self, symbol: "Sequence", items: List[Optional[Matcher]]):
        super().__init__(symbol)
        self.items = items
        self.index = 0

    def debug(self) -> str:
        return f'''({" ".join([
            f"[{self.items[i].debug()}]" if i == self.index and self.items[i]
            else repr(self.symbol.items[i])
            for i in range(len(self.symbol.items))
        ])})'''

    def get_effective_matcher(self) -> Optional[Matcher]:
        assert self.index < len(self.items)
        if self.items[self.index]:
            return self.items[self.index].get_effective_matcher()
        return None

    def ensure_matcher(self, g: "Grammar", i=0) -> "SequenceMatcher":
        if not self.items[i]:
            self.items[i] = g.resolve(self.symbol.items[i]).enter(g)
        return self

    def get_allowed_tokens(self, g: "Grammar") -> AllowedTokens:
        assert self.index < len(self.items)
        rv = self.items[self.index].get_allowed_tokens(g)
        if rv.look_ahead:
            i = self.index
            ahead = rv
            while i < len(self.symbol.items) - 1:
                i += 1
                self.ensure_matcher(g, i)
                ahead = self.items[i].get_allowed_tokens(g)
                rv = rv.combine(ahead)
                if not ahead.look_ahead:
                    break
            if not ahead.look_ahead:
                rv.look_ahead = False
        return rv

    def advance(self, g: "Grammar", token_id: int) -> Advance:
        a = self.items[self.index].advance(g, token_id)
        if a in (Advance.Done, Advance.TryNext):
            if self.index < len(self.symbol.items) - 1:
                self.index += 1
                self.ensure_matcher(g, self.index)
                if a == Advance.TryNext:
                    return self.advance(g, token_id)
                a = Advance.Again
        return a


class AlternativeMatcher(Matcher):
    symbol: "Alternative"

    def __init__(self, symbol: "Alternative", items: Set[Matcher]):
        super().__init__(symbol)
        self.items = items

    def debug(self):
        return f'({" | ".join([x.debug() for x in self.items])})'

    def get_effective_matcher(self) -> "Matcher":
        if len(self.items) == 1:
            return list(self.items)[0].get_effective_matcher()
        return self

    def get_allowed_tokens(self, g: "Grammar") -> AllowedTokens:
        rv = None
        for i in self.items:
            a = i.get_allowed_tokens(g)
            rv = rv.combine(a) if rv else a
        # TODO: should this return 'ban everything' if no alternative is left?
        # TODO: should such state be even possible?
        return rv or AllowedTokens()

    def advance(self, g: "Grammar", token_id: int) -> Advance:
        best_a = Advance.Reject
        for i in list(self.items):
            a = i.advance(g, token_id)
            if a in (Advance.Reject, Advance.TryNext):
                if a == Advance.TryNext and best_a == Advance.Reject:
                    best_a = Advance.TryNext
                self.items.remove(i)
            else:
                best_a = Advance.Done
                if a == Advance.Done:
                    self.items.remove(i)
        if len(self.items) == 0:
            return best_a
        return Advance.Again


class RepeatMatcher(Matcher):
    symbol: "Repeat"

    def __init__(self, symbol: "Symbol", effective_item: Matcher):
        super().__init__(symbol)
        self.effective_item = effective_item
        self.inside = False

    def get_effective_matcher(self) -> "Matcher":
        return self.effective_item.get_effective_matcher() if self.inside else self

    def get_allowed_tokens(self, g: "Grammar") -> AllowedTokens:
        rv = self.effective_item.get_allowed_tokens(g)
        if not self.inside:
            return rv.set_ahead()
        return rv

    def advance(self, g: "Grammar", token_id: int) -> Advance:
        a = self.effective_item.advance(g, token_id)
        if a == Advance.Reject:
            if self.inside:
                return a
            return Advance.TryNext
        elif a == Advance.Done:
            if self.symbol.mode == "*":
                self.effective_item = g.resolve(self.symbol.item).enter(g)
                self.inside = False
                return Advance.Again
            else:   # mode == "?"
                return Advance.Done
        elif a == Advance.Again:
            self.inside = True
        return a
