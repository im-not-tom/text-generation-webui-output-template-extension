from typing import List, Set, Dict
import os

if "OT_TESTING" in os.environ:
    from collections import namedtuple
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained('../../models/oobabooga_llama-tokenizer', use_fast=False)
    shared = namedtuple("Shared", ['tokenizer'])(namedtuple("Tokenizer", ['eos_token_id'])(0))
    from extensions.output_template.test_tokenizer import encode, decode 
else:
    from modules import shared

    def encode(text) -> List[int]:
        return shared.tokenizer.encode(str(text), add_special_tokens=False)

    def decode(token_ids: List[int]) -> str:
        return shared.tokenizer.decode(token_ids)



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


def get_token_dictionary() -> Dict[int, str]:
    from extensions.output_template.script import params, logger
    if not params["token_dictionary"] or params["used_tokenizer"] is not shared.tokenizer:
        assert params["scores_size"]
        logger.info("output_template: Creating token dictionary. This takes few seconds, but is done only once.")
        params["token_dictionary"] = { i: decode([i]) for i in range(params["scores_size"]) }
        params["used_tokenizer"] = shared.tokenizer
        logger.info("output_template: Done creating token dictionary.")
    return params["token_dictionary"]
