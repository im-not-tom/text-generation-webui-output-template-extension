from typing import List, Set, Dict
import torch, os
MINUS_INF = -float("inf")


if "OT_TESTING" in os.environ:
    from collections import namedtuple
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

    When applied on scores, following order is done:
      1. If 'banned' is not empty, all but banned tokens are allowed.
         If token is both allowed and banned, it's allowed. 
      2. if 'allowed' is not empty (and banned is), all but allowed tokens are banned.
      3. if 'allow_eos' is False, end-of-string token is banned in any case.

    'look_ahead' is used by Repeat symbol to signal that next symbol should also be considered.
    """
    def __init__(self, *, allowed=None, banned=None, look_ahead=False, allow_eos=False):
        self.allowed: Set[int] = allowed or set()
        self.banned: Set[int] = banned or set()
        assert (self.allowed and not self.banned) or (self.banned and not self.allowed) or not (self.allowed and self.banned)
        self.look_ahead = look_ahead
        self.allow_eos = allow_eos

    def combine(self, other: "AllowedTokens") -> "AllowedTokens":
        """ Returns new instance which is combination of self and other """
        allowed = set()
        banned = set()
        if (not self.allowed and not self.banned) or (not other.allowed and not other.banned):
            # One of self/other is 'allow all'
            pass
        elif self.allowed and other.allowed:
            # Both are 'allow only these'
            assert not self.banned and not other.banned
            allowed = self.allowed.union(other.allowed)
        elif self.banned and other.banned:
            # Both are 'ban only these'
            assert not self.allowed and not other.allowed
            banned = self.banned.intersection(other.banned)
        elif self.allowed and other.banned:
            # I have allowed tokens, other has banned tokens.
            # Allow everything but those we both banned
            assert not self.banned and not other.allowed
            banned = other.banned - self.allowed
        elif other.allowed and self.banned:
            # As above but reversed
            return other.combine(self)
        else:
            assert False, "impossible combination"

        return AllowedTokens(
            allow_eos=self.allow_eos or other.allow_eos,
            look_ahead=self.look_ahead or other.look_ahead,
            allowed=allowed,
            banned=banned,
        )

    def set_ahead(self):
        """ Returns copy of self with 'look_ahead' set to True """
        return AllowedTokens(
            allow_eos=self.allow_eos,
            allowed=self.allowed,
            banned=self.banned,
            look_ahead=True,
        )

    def __repr__(self):
        data = []
        if self.look_ahead or self.allow_eos:
            data.append(",".join([
                "ahead" if self.look_ahead else "",
                "eos" if self.allow_eos else ""
            ]).strip(","))
        data.append(f"allowed={self.allowed}")
        data.append(f"banned={self.banned}")
        return f"<AllowedTokens {' '.join(data)}>"

    def apply(self, scores: torch.FloatTensor):
        if self.allowed and not self.banned:
            s = scores.new_full(scores.shape, False, dtype=torch.bool)
            for a in self.allowed:
                s[..., a] = True
            s[..., shared.tokenizer.eos_token_id] = True
            scores[~s] = MINUS_INF
        if self.banned or not self.allow_eos:
            s = scores.new_full(scores.shape, True, dtype=torch.bool)
            for a in self.banned:
                if a not in self.allowed:
                    s[..., a] = False
            if not self.allow_eos:
                s[..., shared.tokenizer.eos_token_id] = False
            scores[~s] = MINUS_INF


def get_token_dictionary() -> Dict[int, str]:
    from extensions.output_template.script import params, logger
    if not params["token_dictionary"] or params["used_tokenizer"] is not shared.tokenizer:
        assert params["scores_size"]
        logger.info("output_template: Creating token dictionary. This takes few seconds, but is done only once.")
        params["token_dictionary"] = { i: decode([i]) for i in range(params["scores_size"]) }
        params["used_tokenizer"] = shared.tokenizer
        logger.info("output_template: Done creating token dictionary.")
    return params["token_dictionary"]
