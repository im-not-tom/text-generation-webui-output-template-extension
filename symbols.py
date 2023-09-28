from typing import Tuple, List, Dict, Union, Optional, Set

from extensions.output_template.state_machine import Matcher, RepeatMatcher, TerminalMatcher, SequenceMatcher, \
    RegExpMatcher, AlternativeMatcher
from extensions.output_template.utils import encode, decode, get_token_dictionary, AllowedTokens
from enum import IntEnum
import re


class Symbol:
    def validate(self, g: "Grammar"):
        """
        Validates against grammar.
        Currently just checks that for each Terminal there is rule defined.
        """
        pass

    def enter(self, g: "Grammar") -> Matcher:
        raise NotImplementedError(f"enter on {self.__class__.__name__}")


class NonTerminal(Symbol):
    def __init__(self, name: str):
        self.name = name

    def __repr__(self):
        return self.name

    def validate(self, g: "Grammar"):
        g.resolve(self).validate(g)

    def enter(self, g: "Grammar") -> Matcher:
        return g.resolve(self).enter(g)


class Terminal(Symbol):
    def __init__(self, value: str):
        self.value = value
        self.allowed_cache = {}

    def __repr__(self):
        return f't{repr(self.value)}'

    def validate(self, g: "Grammar"):
        if not self.value:
            raise ValueError("empty terminal")

    def enter(self, g: "Grammar") -> Matcher:
        return TerminalMatcher(self)


class RegExp(Symbol):
    allowed_cache = {}
    banned_cache = {}

    def __init__(self, value: str):
        self.value = value
        if self.value.startswith("[^"):
            # Special handling to optimize.
            # Instead of allowing all tokens but these, rest is banned
            self.negative = True
            self.re = re.compile("[" + value[2:], re.MULTILINE | re.DOTALL)
        else:
            self.negative = False
            self.re = re.compile("^" + value + "$", re.MULTILINE | re.DOTALL)

    def __repr__(self):
        return f'r{self.value}'

    def enter(self, g: "Grammar") -> Matcher:
        return RegExpMatcher(self)


class Collection(Symbol):
    def __init__(self, items: List[Symbol]):
        self.items = items


class Sequence(Collection):
    def __init__(self, items: List[Symbol]):
        super().__init__(items)
        self.effective = []
        self.index = 0

    def __repr__(self):
        return f'({" ".join([repr(x) for x in self.items])})'

    def enter(self, g: "Grammar") -> Matcher:
        return SequenceMatcher(self, [
            None
            for m in self.items
        ]).ensure_matcher(g)


class Alternative(Collection):
    def __init__(self, items: List[Matcher]):
        super().__init__([])
        for i in items:
            if isinstance(i, Alternative):
                self.items += i.items
            else:
                self.items.append(i)
        self.possible = set()

    def __repr__(self):
        return f'({" | ".join([repr(x) for x in self.items])})'

    def validate(self, g: "Grammar"):
        for item in self.items:
            g.resolve(item).validate(g)

    def enter(self, g: "Grammar") -> "Matcher":
        return AlternativeMatcher(self, {
            g.resolve(m).enter(g)
            for m in self.items
        })


class Repeat(Symbol):
    def __init__(self, mode: str, item: Symbol):
        assert mode in "*?"
        self.item = item
        self.mode = mode

    def __repr__(self):
        if isinstance(self.item, (Terminal, NonTerminal)):
            return f'({repr(self.item)}){self.mode}'
        else:
            return f'{repr(self.item)}{self.mode}'

    def validate(self, g: "Grammar"):
        self.item.validate(g)

    def enter(self, g: "Grammar") -> Matcher:
        return RepeatMatcher(self, g.resolve(self.item).enter(g))
