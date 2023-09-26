"""
Fake tokenizer implementation used for testing.
Uses only 127 tokens with few IDs reserved for multi-character tokens
to simulate possible issues.
"""
from typing import List

tokens = {
    0:  "\x00",  # EOS
    10: "\n",    # newline
    # Randomly chosen strings
    # Ordered from longest to shortest and order is important.
    1:  "Universe",
    2:  "token",
    3:  "world",
    4:  "stock",
    5:  '..."',
    6:  "hall",
    7:  "the",
    8:  "...",
    9:  "and",
    11: "com",
    12: "Neg",
    13: "   ",
    14: "end",
    15: "six",
    16: "tab",
    17: "- [",
    18: "gg",
    19: "He",
    20: "- ",
    21: "ni",
    22: "oo",
    23: "[]",
    24: "or",
    25: "ro",
    26: "),",
    27: "of",
    28: "to",
    29: "by",
    30: "++",
    31: "],",
    # From space above, all printable characters
    **{i: chr(i) for i in range(32, 127)},
}


def encode(text) -> List[int]:
    rv = []
    while text:
        for i in tokens:
            if text.startswith(tokens[i]):
                rv.append(i)
                text = text[len(tokens[i]):]
                break
        else:
            text = text[1:]
    return rv


def decode(token_ids: List[int]) -> str:
    return "".join([tokens[i] for i in token_ids])
