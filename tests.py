import os
from typing import Union

os.environ["OT_TESTING"] = "1"
from extensions.output_template.state_machine import AnyTokenMatcher, TerminalMatcher
from extensions.output_template.script import TemplatingLogitsProcessor, params
from extensions.output_template.utils import encode, decode, shared, MINUS_INF
from extensions.output_template.grammar import Grammar, Repeat, RegExp
from torch import Tensor
import math, random, json

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
    return Tensor([[ 0.0001 + (math.floor(random.random() * 100) / 100.0) for _ in range(127) ]])


def set_score(token_id: Union[str, int, list], scores, value=1000.0):
    if type(token_id) is list:
        for i in token_id:
            scores = set_score(i, scores, value)
        return scores
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
    TemplatingLogitsProcessor()(None, scores)
    best = int(scores.argmax())
    grammar: Grammar = params["grammar"]
    grammar.advance(best)
    return best


def get_text(until=EOS, score_fn=random_scores) -> str:
    tokens = []
    while True:
        t = sample_test(score_fn())
        if t != EOS:
            tokens.append(t)
        if type(until) is str and until in decode([t]):
            break
        elif t == until:
            break

    return decode(tokens)


def test_grammar_parser():
    g = Grammar(TEMPLATE)
    assert len(g.rules) == 8

    g.reset("""
        root ::= "hi"
        # Testing case when grammar ends with non-terminated line with comment""")


def test_terminal():
    grammar: Grammar = params["grammar"]
    grammar.reset("""root ::= 'Hello world' [\n]+""")
    EOL = encode("\n")[0]
    t = get_text(EOL)
    assert t == "Hello world\n"
    assert grammar.active_matcher

    grammar.reset()
    scores = grammar.update_scores(random_scores())
    assert len(encode("He")) == 1
    assert scores[..., encode("He")] > MINUS_INF
    assert scores[..., encode("H")] > MINUS_INF
    grammar.advance(encode("H")[0])
    assert ord('e') == sample_test(random_scores())
    matcher = grammar.get_effective_matcher()
    while grammar.get_effective_matcher() is matcher:
        sample_test(random_scores())
    assert isinstance(grammar.get_effective_matcher().symbol, (RegExp, Repeat))


def test_alternate():
    grammar: Grammar = params["grammar"]
    grammar.reset(TEMPLATE)
    grammar.enter_rule("action")
    sample_test(set_score("/", random_scores()))
    text = get_text()
    assert text in ("go hall", "go kitchen")


def test_sequence():
    grammar: Grammar = params["grammar"]
    grammar.reset(TEMPLATE)
    get_text(encode(" ")[0])
    sample_test(set_score("-", random_scores()))
    assert 32 == sample_test(random_scores())
    tokens = []
    while not isinstance(grammar.get_effective_matcher().symbol, (Repeat, RegExp)):
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
    scores = TemplatingLogitsProcessor()(None, random_scores())
    assert len([x for x in scores[0] if x > MINUS_INF]) == 26
    assert scores[..., EOS] == MINUS_INF        # Also make sure that EOS is banned in any case

    # Tests that combination of repetition and regexp allows multi-char tokens
    grammar.reset("root ::= [a-z]+")
    scores = TemplatingLogitsProcessor()(None, random_scores())
    assert len([x for x in scores[0] if x > MINUS_INF]) > 26
    assert scores[..., encode("world")] > MINUS_INF
    assert scores[..., EOS] == MINUS_INF

    # Tests banning specific character
    grammar.reset("root ::= [^\n]+")
    scores = TemplatingLogitsProcessor()(None, random_scores())
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
        assert sample_test(set_score('"', random_scores(), MINUS_INF)) not in (EOS, q)
    # Test that " is still allowed to be generated
    scores = TemplatingLogitsProcessor()(None, set_score('"', random_scores()))
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
    text = get_text()
    assert "foo" in text or "ba" in text

    # Actual case that was broken originally
    grammar: Grammar = params["grammar"]
    grammar.reset("""
        root ::= line line line
        line ::= "Alice:" space action newline
        action ::= speech | bullet | command
        space ::= " "
        newline ::= "\n"
        bullet ::= ("- " "I'll go" space "to" space location space "and do") gibberish
        command ::= "/go" space location
        location ::= ("hall" | "out" | "room")
        speech ::= "\\"" ">" gibberish "\\""
        gibberish ::= [^-/">\n]+
    """)
    text = get_text()
    commands = len(text.split("/go")) - 1
    bullets = len(text.split("-")) - 1
    speech = len(text.split(">")) - 1
    assert 3 == commands + bullets + speech


def test_json():
    grammar: Grammar = params["grammar"]
    grammar.reset("""
        root   ::= object
        value  ::= object | array | string | number | ("true" | "false" | "null") ws

        object ::=
          "{" ws (
                    string ":" ws value
            ("," ws string ":" ws value)*
          )? "}" ws

        array  ::=
          "[" ws (
                    value
            ("," ws value)*
          )? "]" ws

        string ::=
          "\\"" (
            [^"\\\\\n] |
            "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])        # escapes
          )* "\\"" ws

        number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? ([eE] [-+]? [0-9]+)? ws

        # Optional space: by convention, applied in this grammar after literal chars when allowed
        ws ::= ([ \\t\\n] ws)?
    """)

    random.seed(2342343231)
    # 1st token has to be {
    assert ord("{") == sample_test(random_scores())
    # Any number of whitespace has to be allowed
    assert ord(" ") == sample_test(set_score(32, random_scores(), 1000.0))
    assert ord(" ") == sample_test(set_score(32, random_scores(), 1000.0))
    assert ord(" ") == sample_test(set_score(32, random_scores(), 1000.0))
    assert ord("\n") == sample_test(set_score(10, random_scores(), 1000.0))
    # Ban }. If not whitespace, only '"' may follow
    assert ord('"') == sample_test(set_score(list("} \n"), random_scores(), MINUS_INF))
    # Grab some tokens and force string to finish
    for _ in range(5):
        sample_test(set_score('"', random_scores(), MINUS_INF))
    assert ord('"') == sample_test(set_score('"', random_scores(), 1000))
    # Now only whitespace and ':' should be allowed
    scores = random_scores()
    TemplatingLogitsProcessor()(None, scores)
    assert scores[..., ord(':')] > 0
    assert len([x for x in scores[0] if x > 0]) == 3    # ':', space and newline
    # Go over ':' and ban whitespace. Now start-of-value tokens should be allowed
    sample_test(set_score(':', random_scores()))
    scores = set_score(list(' \n'), random_scores(), MINUS_INF)
    TemplatingLogitsProcessor()(None, scores)
    # 14 characters =  0-9, minus sign, {, [ and '"'.
    # Additionally, 3 characters for n, t and f of null, true and false  
    assert len([ x for x in scores[0] if x > 0 ]) == 17
    # Force 'true' and verify repetition
    sample_test(set_score('t', random_scores()))
    assert "true" == 't' + get_text('e')
    assert ord(',') == sample_test(set_score(list('} \n'), random_scores(), MINUS_INF))
    # Force another key and make sure it generates quotes properly (this was failing before)
    sample_test(set_score('"', random_scores()))

    a = get_text('"')
    assert '"' not in a[:-1]
    scores = random_scores()
    TemplatingLogitsProcessor()(None, scores)
    assert len([x for x in scores[0] if x > 0]) == 3    # ':', space and newline
    assert ord(':') == sample_test(set_score(list(' \n'), random_scores(), MINUS_INF))

    # Now just restart grammar and test it generates some proper json
    for _ in range(100):
        # (and do it few times, cos it tends to generate just {})
        grammar.reset()
        a = get_text()
        json.loads(a)


def test_any_token():
    """ Test nonstandard rule to disable grammar """
    grammar: Grammar = params["grammar"]
    grammar.reset("""
        root ::= (donotend)
        donotend ::= (.*)
    """)
    for i in range(255):
        z = sample_test(random_scores())
    assert isinstance(grammar.get_effective_matcher(), AnyTokenMatcher)


def test_allow_next():
    """
    Tests workaround for following issue:
    Given grammar rule like root ::= [^"]* '"' expressing "anything until quotation mark, then quotation mark",
    LogitsProcessor actually bans most of the tokens ending in " as they don't match 1st rule in sequence.
    This would be technically okay and would generate correct output, but banning all those tokens would
    result in changing LLM behaviour too much and cause it to generate much longer quoted strings.
    """
    # TODO: it may be good idea to do similar workaround for regexp followed by Alternative, but for now this
    # TODO: should fix the biggest issue.
    grammar: Grammar = params["grammar"]
    grammar.reset("""
        root ::= '"' [^"]* '"' 'H'
    """)

    # Sanity check for test tokenizer tokenizer
    assert len(encode('."')) == 1
    TOKEN = encode('."')[0]
    Q = encode('"')[0]

    # Go over " to reach 'anything but " part of seuquence'
    scores = TemplatingLogitsProcessor()(None, random_scores())
    assert scores[..., Q] > MINUS_INF
    grammar.advance(Q)
    # Check that '."' token is allowed as it fits above grammar
    scores = TemplatingLogitsProcessor()(None, random_scores())
    assert scores[..., TOKEN] > MINUS_INF
    grammar.advance(TOKEN)

    # Check that grammar moved to last rule
    assert isinstance(grammar.get_effective_matcher(), TerminalMatcher)
    assert grammar.get_effective_matcher().symbol.value == "H"


if __name__ == "__main__":
    params["scores_size"] = 127
    params["enabled"] = True
    test_grammar_parser()
    test_terminal()
    test_alternate()
    test_sequence()
    test_regexp()
    test_repeat()
    test_json()
    test_any_token()
    test_allow_next()
