import os
from typing import Union

os.environ["OT_TESTING"] = "1"
from extensions.output_template.script import TemplatingLogitsProcessor, params
from extensions.output_template.grammar import Grammar, Repeat, Alternative, RegExp, MINUS_INF, Advance
from extensions.output_template.utils import encode, decode, shared
from torch import Tensor
import math, random

input_ids = Tensor([[]])
EOS = shared.tokenizer.eos_token_id
TEMPLATE = """
root ::= "Alice:" space action
greeting ::= "hello \\"world\\" \\U00003041"
action ::= speech | bullet | command
space ::= " "
bullet ::= ("- " "I'll go" space "to" space location space "and do") [^\n]+
command ::= "/go " location
location ::= ("hall" | "kitchen")
speech ::= "\\"" [^"\n]+ "\\""
"""


def random_scores():
    return Tensor([[ math.floor(random.random() * 100) / 100.0 for _ in range(127) ]])


def set_score(token_id: Union[str, int], scores, value=1000.0):
    if type(token_id) is str:
        token_id = encode(token_id)[0]
    scores[..., token_id] = value
    return scores


def scores_to_text(scores):
    scores = scores[0]
    return " ".join([
        f"{repr(chr(i))}:{scores[i]}"
        for i in range(len(scores))
        if scores[i] > 0
    ])


def sample_test(scores) -> int:
    # Returns single generated token
    TemplatingLogitsProcessor()(input_ids, scores)
    best = int(scores.argmax())
    grammar: Grammar = params["grammar"]
    grammar.advance(best)
    return best


def get_text(until=EOS) -> str:
    tokens = []
    while True:
        t = sample_test(random_scores())
        if t == until:
            break
        tokens.append(t)

    return decode(tokens)


def test_grammar_parser():
    g = Grammar(TEMPLATE)
    assert len(g.rules) == 8


def test_terminal():
    random.seed()
    grammar: Grammar = params["grammar"]
    grammar.reset("""root ::= 'one' ('two' 'three' ('four' 'five')) 'six'""")
    assert get_text() == "onetwothreefourfivesix"
    grammar.reset("""root ::= 'Hello world' [\n]+""")
    assert get_text(encode("\n")[0]) == "Hello world"
    assert grammar.active_symbol

    grammar.reset()
    scores = grammar.update_scores(random_scores())
    assert len(encode("He")) == 1
    assert scores[..., encode("He")] > MINUS_INF
    assert scores[..., encode("H")] > MINUS_INF
    grammar.advance(encode("H")[0])
    assert ord('e') == sample_test(random_scores())
    symbol = grammar.get_effective_symbol()
    while grammar.get_effective_symbol() is symbol:
        sample_test(random_scores())
    assert isinstance(grammar.get_effective_symbol(), (RegExp, Repeat))


def test_choice():
    random.seed()
    grammar: Grammar = params["grammar"]
    grammar.reset(TEMPLATE)
    grammar.enter_rule("action")
    sample_test(set_score("/", random_scores()))
    tokens = []
    while True:
        t = sample_test(random_scores())
        if t == EOS:
            break
        tokens.append(t)

    assert decode(tokens) in ("go hall", "go kitchen")


def test_sequence():
    random.seed()
    grammar: Grammar = params["grammar"]
    grammar.reset(TEMPLATE)
    get_text(encode(" ")[0])
    sample_test(set_score("-", random_scores()))
    assert 32 == sample_test(random_scores())
    tokens = []
    while not isinstance(grammar.get_effective_symbol(), (Repeat, RegExp)):
        t = sample_test(random_scores())
        if t == EOS:
            break
        tokens.append(t)

    assert decode(tokens) in (
        "I'll go to hall and do",
        "I'll go to kitchen and do",
    )


def test_regexp():
    grammar: Grammar = params["grammar"]
    grammar.reset("root ::= [a-z]")
    scores = TemplatingLogitsProcessor()(input_ids, random_scores())
    assert len([x for x in scores[0] if x > MINUS_INF]) == 26
    assert scores[..., EOS] == MINUS_INF        # Also make sure that EOS is banned in any case

    # Tests that combination of repetition and regexp allows multi-char tokens
    grammar.reset("root ::= [a-z]+")
    scores = TemplatingLogitsProcessor()(input_ids, random_scores())
    assert len([x for x in scores[0] if x > MINUS_INF]) > 26
    assert scores[..., encode("world")] > MINUS_INF
    assert scores[..., EOS] == MINUS_INF

    # Tests banning specific character
    grammar.reset("root ::= [^\n]+")
    scores = TemplatingLogitsProcessor()(input_ids, random_scores())
    assert scores[..., 10] < 0
    assert scores[..., EOS] == MINUS_INF


def test_repeat():
    # Tests "speech" rule in TEMPLATE grammar
    grammar: Grammar = params["grammar"]
    grammar.reset(TEMPLATE)
    get_text(encode(" ")[0])
    sample_test(set_score('"', random_scores()))
    q = encode('"')[0]
    for _ in range(10):
        # Just generate few random chars
        assert sample_test(set_score('"', random_scores(), -1000)) not in (EOS, q)
    scores = TemplatingLogitsProcessor()(input_ids, set_score('"', random_scores()))
    assert scores[..., q] > MINUS_INF
    assert q == sample_test(scores)
    assert EOS == sample_test(random_scores())

    # Tests grammar that generates repeated combination of foobars
    grammar: Grammar = params["grammar"]
    grammar.reset("""
        root ::= many
        many ::= one one one+
        one ::= foo | bar
        foo ::= "foo"
        bar ::= "b" "a"+ "r"
    """)
    random.seed(31544201)
    text = get_text()
    assert "foo" in text
    assert "aa" in text

    # Actual case that was broken originally
    random.seed(941815)
    grammar: Grammar = params["grammar"]
    grammar.reset("""
        root ::= line line line
        line ::= "Alice:" space action newline
        action ::= speech | bullet | command
        space ::= " "
        newline ::= "\n"
        bullet ::= ("- " "I'll go" space "to" space location space "and do") [^\n]+
        command ::= "/go" space location
        location ::= ("hall" | "out" | "room")
        speech ::= "\\"" [^"\n]+ "\\""
    """)
    text = get_text()
    assert len(text.split("/go")) == 4


if __name__ == "__main__":
    params["scores_size"] = 127
    params["enabled"] = True
    test_grammar_parser()
    test_terminal()
    test_choice()
    test_sequence()
    test_regexp()
    test_repeat()
