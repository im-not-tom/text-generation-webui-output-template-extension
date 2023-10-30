from extensions.output_template.state_machine import *
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
        self.next: Optional[Terminal] = None
        if self.value.startswith("[^"):
            # To prevent generating giant set of almost all tokens,
            # tokens matching negative are banned instead
            self.negative = True
            self.re = re.compile("[" + value[2:], re.MULTILINE | re.DOTALL)
        else:
            self.negative = False
            self.re = re.compile("^" + value + "$", re.MULTILINE | re.DOTALL)

    def make_re(self):
        # https://youtu.be/iQrjbRz3y7A
        if self.value.startswith("[^"):
            # To prevent generating giant set of almost all tokens,
            # tokens matching this  rest is banned
            self.negative = True
            r = "[" + self.value[2:]
            if self.next:
                r += "(" + re.escape(self.next.value) + ")?"
            self.re = re.compile(r, re.MULTILINE | re.DOTALL)
        else:
            self.negative = False
            self.re = re.compile("^" + self.value + "$", re.MULTILINE | re.DOTALL)

    def allow_next(self, t: Terminal):
        """
        When regexp is followed by terminal, regexp is configured to also match tokens that include that terminal.
        This prevents banning perfectly good tokens and biasing LLM output.
        (also see 'test_allow_next')
        """
        self.next = t

    def __repr__(self):
        return f'r{self.value}'

    def enter(self, g: "Grammar") -> Matcher:
        return RegExpMatcher(self)


class AnyToken(Symbol):
    """
    Special symbol to which '.*' is translated.
    Just matches anything, basically turning grammar off.
    """

    def __repr__(self):
        return f'.*'

    def enter(self, g: "Grammar") -> Matcher:
        return AnyTokenMatcher(self)


class Collection(Symbol):
    def __init__(self, items: List[Symbol]):
        self.items = items


class Sequence(Collection):
    def __init__(self, items: List[Symbol]):
        super().__init__(items)
        self.effective = []
        self.index = 0

    def validate(self, g: "Grammar"):
        super().validate(g)
        for i in range(len(self.items) - 1):
            if (True
                and isinstance(self.items[i], Repeat)
                and isinstance(self.items[i].item, RegExp)
                and isinstance(g.resolve(self.items[i + 1]), Terminal)
            ):
                # See test_allow_next
                self.items[i].item.allow_next(g.resolve(self.items[i + 1]))

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
