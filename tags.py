from extensions.output_template.utils import encode, decode, shared, AllowedTokens, get_token_dictionary


class TemplateTag:
    tag_name = ""

    def __init__(self, *, attrs):
        self.children = []
        self.child_index = 0
        self.attrs = attrs

    def __repr__(self):
        return f"<{self.tag_name} {self.children}>"

    def get_text(self) -> str:
        """ Returns plain text contained in this tag or its children """
        return "".join([ c.get_text() for c in self.children ])

    def debug(self) -> str:
        """
        Returns string describing current state of tag.
        There's no logic here, just what seem'd most readable at the time
        """
        s = ' '.join([
            f'{{{c.debug()}}}' if i == self.child_index else c.debug()
            for (i, c) in enumerate(self.children)
        ])
        return f"<{self.tag_name} {s}>"

    def get_banned_tokens(self) -> AllowedTokens:
        """ Parses 'ban' attribute, if any and bans any token containing specified text """
        rv = AllowedTokens()
        if "ban" in self.attrs:
            d = get_token_dictionary()
            text = self.attrs["ban"].replace("\\n", "\n")   # TODO: ugly hack
            # TODO: cache this? performance impact should be checked
            rv.banned = { k for k in d if text in d[k] }
        if "eos" in self.attrs:
            rv.allow_eos = True
        return rv

    def get_allowed_tokens(self) -> AllowedTokens:
        if self.child_index < len(self.children):
            rv = self.children[self.child_index].get_allowed_tokens()
            return rv.combine(self.get_banned_tokens())
        else:
            return self.get_banned_tokens()

    def accepts(self, token: int) -> int:
        """
        Returns:
            - 1 or positive number if tokens so far matches expected pattern
              but there are more tokens left in pattern
            - 0 if tokens so far matches expected pattern exactly (tokens are accepted)
            - -1 or negative number if token doesn't match expected pattern (tokens are refused)
        """
        rv = -1
        if self.child_index < len(self.children):
            rv = self.children[self.child_index].accepts(token)
            if rv == 0:
                self.child_index += 1
                if self.child_index < len(self.children):
                    rv = 1
        return rv

    def reset(self):
        """ Resets internal state, if any """
        self.child_index = 0
        for c in self.children:
            c.reset()

    def finalize(self): pass


class SimpleText(TemplateTag):
    """
    SimpleText is (not a) tag which simply forces given text as next part of output.
    """
    tag_name = "*"

    def __init__(self, text):
        super().__init__(attrs={})
        self.text = text
        self.reset()

    def ensure_tokens(self):
        if self.tokens is None:
            self.tokens = encode(self.text)

    def get_allowed_tokens(self) -> AllowedTokens:
        self.ensure_tokens()
        if not len(self.tokens):
            # Ran out of tokens, tested string is longer than expected
            return AllowedTokens()
        return AllowedTokens(allowed={self.tokens[0]})

    def get_text(self) -> str:
        return self.text

    def reset(self):
        self.tokens = None   # computed on demand

    def accepts(self, token: int) -> int:
        if token in self.get_allowed_tokens().allowed:
            self.tokens = self.tokens[1:]
            if not self.tokens:
                # Just passed accepted string
                return 0
            # In middle of accepted string
            return 1
        else:
            # Not accepted string
            return -1

    def __repr__(self):
        return repr(self.text)

    def debug(self) -> str:
        rest = decode(self.tokens) if self.tokens else None
        if rest and self.text.endswith(rest):
            return repr(f'{self.text[:-len(rest)]}[{rest}]')
        else:
            return repr(self.text)


class NewLine(SimpleText):
    """
    When template is parsed, newlines are ignored.
    This tag exists to force newline (\n) when necessary.
    """
    tag_name = "newline"

    def __init__(self, *, attrs):
        super().__init__("\n")


class Space(SimpleText):
    """
    Spaces are usually not ignored, but this may be used
    to force padding or to make template more readable.
    """
    tag_name = "space"

    def __init__(self, *, attrs):
        super().__init__(" ")


class Stop(SimpleText):
    """
    </s> (end-of-string token) is banned by most of the tags
    to prevent generation prematurely.
    This tag can be used to allow (or even force) </s> token.
    """
    tag_name = "stop"

    def __init__(self, *, attrs):
        super().__init__("#")

    def ensure_tokens(self):
        if self.tokens is None:
            self.tokens = [int(shared.tokenizer.eos_token_id)]


class AnyTag(TemplateTag):
    """ Terminates template prematurely and lets LLM generate rest of output normally """
    tag_name = "any"

    def __init__(self, *, attrs):
        super().__init__(attrs=attrs)
        self.all_tokens = None

    def __repr__(self):
        return f"<{self.tag_name}/>"

    def get_allowed_tokens(self) -> AllowedTokens:
        rv = super().get_banned_tokens()
        rv.allow_all = True
        return rv

    def accepts(self, token: int) -> int:
        # Takes anything, forever
        return 1


class UntilTag(TemplateTag):
    """
    Lets LLM generate any output until specified text is occurred.
    """
    tag_name = "until"

    def __init__(self, *, attrs):
        super().__init__(attrs=attrs)
        self.banned_tokens = None

    def debug(self) -> str:
        return f"<{self.tag_name} {repr(self.attrs.get('text'))}>"

    def get_allowed_tokens(self) -> AllowedTokens:
        # TODO: should children be considered at all?
        if self.banned_tokens is None:
            self.banned_tokens = set()
            text = self.attrs.get("text") or ""
            if text:
                 d = get_token_dictionary()
                 for k in d:
                     # TODO: possibly check only next token in child instead of whole text?
                     if text in d[k] and not d[k].endswith(text):
                         self.banned_tokens.add(k)

        return AllowedTokens(allow_all=True, banned=self.banned_tokens.union(self.get_banned_tokens().banned))

    def accepts(self, token: int) -> int:
        text = self.attrs.get("text") or ""
        if not text:
            # With no text set, this accepts anything
            return 1
        if decode([token]).endswith(text):
            # Special case for matching token suffix
            # print("UNTIL suffix accepted", token)
            return 0
        return 1


class RepeatTag(TemplateTag):
    """
    Repeats child node(s) forever, or, if "times" attribute is set, that many times.
    """
    tag_name = "repeat"

    def __init__(self, *, attrs):
        super().__init__(attrs=attrs)
        self.times = -1
        if "times" in self.attrs:
            try:
                self.times = int(self.attrs['times'])
            except ValueError:
                self.times = 0

    def debug(self) -> str:
        rv = super().debug()
        if self.times >= 0:
            rv = f"<{self.times} times {rv[2 + len(self.tag_name):]}"
        return rv

    def accepts(self, token: int) -> int:
        # TODO: this is almost copy-paste of super method
        rv = -1
        if self.child_index < len(self.children):
            rv = self.children[self.child_index].accepts(token)
            if rv == 0:
                self.child_index += 1
                # TODO: only difference is this
                if self.child_index >= len(self.children):
                    if self.times == 0:
                        return 0
                    self.reset()
                    # print("-- repeat repeats --")
                    if self.times > 0:
                        self.times -= 1
                rv = 1
        return rv


class LineTag(UntilTag):
    """ Lets LLM generate output normally until newline is occurred """
    tag_name = "line"

    def __init__(self, *, attrs):
        super().__init__("\n", attrs=attrs)

    def __repr__(self):
        return f"<{self.tag_name}/>"


class ChoicesTag(TemplateTag):
    tag_name = "choices"

    def finalize(self):
        self.children = [ c for c in self.children if c.tag_name == ChoiceTag.tag_name ]
        self.possible_children = self.children

    def debug(self) -> str:
        if not self.possible_children:
            return super().debug()
        s = ""
        if len(self.possible_children) > 1:
            s = f"{len(self.possible_children)} possible"
        else:
            s = self.possible_children[0].debug()[2 + len(ChoiceTag.tag_name):-1]
        return f"<{self.tag_name} {s}>"

    def reset(self):
        super().reset()
        self.possible_children = self.children

    def get_allowed_tokens(self) -> AllowedTokens:
        rv = self.get_banned_tokens()
        for c in self.possible_children:
            rv = rv.combine(c.get_allowed_tokens())
        return rv

    def accepts(self, token_id: int) -> int:
        children, self.possible_children = self.possible_children, []
        rv = -1
        for c in children:
            a = c.accepts(token_id)
            # print("%", token, rv, c)
            if a == 0:
                return 0
            elif a > 0:
                self.possible_children.append(c)
                # print(token_id, repr(decode([token_id])), "ACCEPTED BY", c)
                rv = a
        return rv


class ChoiceTag(TemplateTag):
    tag_name = "choice"


tag_name_to_class = {
    ChoicesTag.tag_name: ChoicesTag,
    ChoiceTag.tag_name: ChoiceTag,
    LineTag.tag_name: LineTag,
    UntilTag.tag_name: UntilTag,
    RepeatTag.tag_name: RepeatTag,
    AnyTag.tag_name: AnyTag,
    NewLine.tag_name: NewLine,
    Space.tag_name: Space,
    Stop.tag_name: Stop,
}