import os
os.environ["OT_TESTING"] = "1"
from extensions.output_template.script import TemplatingLogitsProcessor, params
from extensions.output_template.utils import encode, decode
from extensions.output_template.template import Template
from torch import Tensor, all as torch_all
from copy import deepcopy
import math, random

input_ids = Tensor([[]])  
TEMPLATE = """### Response:
Alice: <choices>
  <choice>"<until text='"'/><newline/></choice>
  <choice>- <any/></choice>
  <choice>/go<space/>
    <choices>
      <choice>hall</choice>
      <choice>kitchen</choice>
    </choices>
  </choice>
</choices>
"""


def random_scores():
    return Tensor([[ math.floor(random.random() * 100) / 100.0 for i in range(127) ]])


def set_score(token_id, scores, value=1000.0):
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
    template = params["output_template_obj"]
    TemplatingLogitsProcessor()(input_ids, scores)
    best = int(scores.argmax())
    template.move_forward(best)
    return best


def test_choices():
    """ Tests parsing one of choices and reaching end of template """
    template = params["output_template_obj"]
    template.init_template(TEMPLATE)
    template.get_initial_text()
    random.seed(101)    # to 'randomly' get /go hall
    scores = random_scores()

    for _ in "/go hall":
        # print(c, scores_to_text(TemplatingLogitsProcessor()(input_ids, deepcopy(scores))))
        sample_test(deepcopy(scores))
    assert sample_test(scores) == 0


def test_until_tag():
    """ Tests parsing into choice with <until/> tag """
    template = params["output_template_obj"]
    template.init_template(TEMPLATE)
    template.get_initial_text()

    random.seed(44)
    for c in '"Hello':
        scores = set_score(ord("\n"), random_scores(), value=0)
        assert c == '"' and sample_test(scores) == ord(c) or sample_test(scores) != 0
    # Should take anything until 2nd quote, then newline is forced and template terminates (but " is included)
    scores = set_score(ord('"'), random_scores())
    assert sample_test(scores) == ord('"')
    assert sample_test(random_scores()) == ord('\n')
    assert sample_test(random_scores()) == 0


def test_any_tag():
    """ Tests parsing into choice with <any/> tag (never reaching end of template) """
    template = params["output_template_obj"]
    template.init_template(TEMPLATE)
    template.get_initial_text()
    random.seed(32432)
    scores = random_scores()
    for _ in "- ":
        assert chr(sample_test(deepcopy(scores))) in "- "
    # Should take anything after "- " and so template should change scores
    for _ in "whatever\nsome\0stuff":
        scores = random_scores()
        changed_scores = deepcopy(scores)
        sample_test(changed_scores)
        assert torch_all(changed_scores[1:] == scores[1:])


def test_evil_token():
    """
    Tests issue when <until> tag is supposed to stop at character witch
    may be only part of token one or more tokens.
    """
    template_text = '''"<until text='"'/><newline/>'''
    template = params["output_template_obj"]

    # Check that '..."' and 'six' gets encoded to single token id
    text = '"Hello..."'
    assert len(encode(text)) < len(text)
    assert len(encode("six")) == 1
    random.seed()

    # Check that token with matching suffix doesn't break <until> 
    for text in ('"Hi"', '"Hello..."',):
        template.init_template(template_text)
        for i in encode(text):
            z = set_score(i, random_scores())
            assert i == sample_test(z)
        # Passed 2nd '"', now newline should be forced
        assert ord('\n') == sample_test(random_scores())

    # Check that token with matching infix doesn't break <until>
    random.seed(4301)
    template_text = "- <until text='i' />."
    template.init_template(template_text)
    assert "-" == decode([sample_test(random_scores())])
    assert " " == decode([sample_test(random_scores())])
    # Following forces score of 'six' testing token high and checks that until doesn't allow generating it
    i = encode("six")[0]
    assert i != sample_test(set_score(i, random_scores()))
    while "i" != decode([sample_test(random_scores())]):
        pass
    assert "." == decode([sample_test(random_scores())])


def test_template_parser():
    """ This case was broken originally """
    t = Template("""<choice>/go<space/>
    <choices>
      <choice>hall</choice>
    </choices>
</choice>""")
    tag = t.get_next_tag()
    assert len(tag.children) == 3
    assert tag.children[1].text == " "


def test_template_parser_attributes():
    """ So, basically, I've implemented travesty, but it seems to be the fastest way to do this """
    t = Template("""<choice>/go<space/></choice>""")
    tag = t.get_next_tag()
    assert tag.tag_name == "choice"
    assert len(tag.children) == 2

    t.init_template("""<choice ban='"'><any eos="1" /></choice>""")
    tag = t.get_next_tag()
    assert tag.attrs['ban'] == '"'
    assert tag.children[0].attrs['eos']

    t.init_template("""<choice>[ take: <until text=":" ban="]" /> ]</choice>""")
    tag = t.get_next_tag()
    assert tag.children[1].attrs['text'] == ":"
    assert tag.children[1].attrs['ban'] == "]"


def test_repeat_tag():
    """
    Tests processing with repeat tag.
    """
    template_text = '''
    <repeat>
        <choices>
            <choice>"<until text='"'/><newline/></choice>
            <choice>[<until text="]"/>:</choice>
            <choice><stop/></choice>
        </choices>
    </repeat>
    '''
    template = params["output_template_obj"]
    template.init_template(template_text)

    random.seed(333)    # with this seed, every choice seems to be taken bellow
    for i in range(5):
        choice = decode([sample_test(random_scores())])
        if choice == "[":
            while "]" != decode([sample_test(random_scores())]):
                pass
            assert ":" == decode([sample_test(random_scores())])
        elif choice == '"':
            while not decode([sample_test(random_scores())]).endswith('"'):
                pass
            assert "\n" == decode([sample_test(random_scores())])
        elif choice == '\x00':
            # EOS reached
            return
        else:
            assert False, "unexpected choice"
    assert False, "repeat never ends"


def test_ban_attribute():
    """
    'ban' attribute is available on most of the tags and can be used to ban generation
    of specific token within it.
    """
    template_text = '''
    <repeat times="3">
        [<choices ban="]">
            <choice>"<until text='"'/>.</choice>
            <choice>{<until text='}'/>.</choice>
        </choices>]
    </repeat>
    '''
    template = params["output_template_obj"]
    template.init_template(template_text)

    random.seed()
    while True:
        opens = decode([sample_test(random_scores())])
        if opens == "[":
            text = opens
            while not text.endswith("]"):
                c = decode([sample_test(random_scores())])
                # if c in '[]{}abcABC.' or c.endswith('"'):
                text += c
            assert text[1] == text[-3] or (text[1] == "{" and text[-3] == "}")
        elif opens == '\x00':
            # EOS reached
            break
        else:
            assert False, "unexpected character"

    # Template bellow should produce a lot of text but no 'x' characters
    template_text = '''<any eos="1" ban="x" />'''
    template = params["output_template_obj"]
    template.init_template(template_text)
    while True:
        c = sample_test(random_scores())
        if c == 0:
            break
        assert "x" not in decode([c])

    template_text = '''[<until text=']' ban='i' />]'''
    template = params["output_template_obj"]
    template.init_template(template_text)
    while True:
        c = sample_test(random_scores())
        if c == 0:
            break
        assert "i" not in decode([c])


if __name__ == "__main__":
    test_template_parser()
    test_template_parser_attributes()
    test_choices()
    test_until_tag()
    test_any_tag()
    test_evil_token()
    test_repeat_tag()
    test_ban_attribute()
