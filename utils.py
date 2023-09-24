from typing import List, Set
import os

if "OT_TESTING" in os.environ:
    from collections import namedtuple
    shared = namedtuple("Shared", ['tokenizer'])(namedtuple("Tokenizer", ['eos_token_id'])(0))
    ooba_encode = None
    ooba_decode = None
else:
    from modules.text_generation import encode as ooba_encode, decode as ooba_decode
    from modules import shared


class AllowedTokens:
    """
    Utility class used to combine and eventually allow (or ban) generation of those tokens that may match
    (or for sure don't match) template.

    When applied on scores, following order is used:
      1. if 'allowed' is not empty, all but allowed tokens are banned
         if 'allowed' is empty and 'allow_all' is True, all but EOS tokens are allowed
      2. if 'allow_eos' is True, EOS token is allowed
      3. if 'banned' is not empty, it is applied to ban tokens
    """
    def __init__(self, *, allowed=None, banned=None, allow_all=False, allow_eos=False):
        self.allowed: Set[int] = allowed or set()
        self.banned: Set[int] = banned or set()
        self.allow_eos = allow_eos
        self.allow_all = allow_all

    def combine(self, other: "AllowedTokens") -> "AllowedTokens":
        """ Returns new instance which is combination of self and other """
        return AllowedTokens(
            allowed=self.allowed.union(other.allowed),
            banned=self.banned.union(other.banned),
            allow_eos=self.allow_eos or other.allow_eos,
            allow_all=self.allow_all or other.allow_all,
        )

    def __repr__(self):
        tag = ""
        if self.allow_eos:
            tag += "E"
        if self.allow_all:
            tag += "A"
        if tag:
            tag = " " + tag
        return f"<AllowedTokens{tag} allowed={self.allowed} banned={self.banned}>"

    def apply(self, scores_: list[list[int]]):
        scores = scores_[0]
        if self.allowed:
            s = scores.new_full(scores.shape, -float("inf"))
            for a in self.allowed:
                s[..., a] = scores[a]
            scores_[0] = scores = s
        elif not self.allow_all:
            scores.fill_(-float('inf'))
            if self.allow_eos:
                eos_token_id = int(shared.tokenizer.eos_token_id)
                scores[..., eos_token_id] = 1.0
        if not self.allow_eos:
            eos_token_id = int(shared.tokenizer.eos_token_id)
            scores[..., eos_token_id] = -float("inf")
        for b in self.banned:
            scores[..., b] = -float("inf")


def get_token_dictionary():
    from extensions.output_template.script import params, logger
    if not params["token_dictionary"] or params["used_tokenizer"] is not shared.tokenizer:
        assert params["scores_size"]
        logger.info("output_template: Creating token dictionary. This takes few seconds, but is done only once.")
        params["token_dictionary"] = { i: decode([i]) for i in range(params["scores_size"]) }
        params["used_tokenizer"] = shared.tokenizer
        logger.info("output_template: Done creating token dictionary.")
    return params["token_dictionary"]


def encode(text) -> List[int]:
    if ooba_encode:
        return [int(i) for i in ooba_encode(text, add_bos_token=False)[0]]
    else:
        # This branch is used only in testing and makes no sense for normal use.
        # Additionally, following replacements exist only to simulate issues with matching
        # text to tokens when text is mapped to multiple or only part of a token.
        text = text.replace("six", "\x06").replace('..."', "\x08").replace("...", "\x07")
        return [ord(x) for x in text]


def decode(token_ids: List[int]) -> str:
    if ooba_decode:
        return ooba_decode(token_ids)
    else:
        # See above
        text = "".join([chr(x) for x in token_ids])
        text = text.replace("\x08", '..."').replace("\x07", "...").replace("\x06", "six")
        return text
