from typing import Optional
from extensions.output_template.grammar import Grammar
from extensions.output_template.utils import shared
from functools import partial
import torch, transformers
try:
    from modules.logging_colors import logger
except ModuleNotFoundError:
    # Allows testing by running script outside text-generation-ui
    from logging import Logger
    logger = Logger(__name__)
    transformers = None
    class LogitsProcessor:
        pass


EMPTY_GRAMMAR = "root ::= [.]*"

params = {
    "grammar": Grammar(EMPTY_GRAMMAR),
    "enabled": False,
    "template": "",
    "token_dictionary": None,
    "used_tokenizer": None,
    "scores_size": 0,
}


class TemplatingLogitsProcessor(transformers.LogitsProcessor):

    def __init__(self):
        super().__init__()
        self.last_input_size = 0

    def __call__(self, input_ids: Optional[torch.LongTensor], scores: torch.FloatTensor):
        if params["enabled"]:
            params["scores_size"] = len(scores[0])
            grammar: Grammar = params["grammar"]

            if input_ids is not None:
                # input_ids are None when running from tests.
                input_size = len(input_ids[0])
                if input_size <= self.last_input_size:
                    logger.warning("output_template: input size unexpectedly decreased. Restarting grammar (except wrong output)")
                    grammar.reset()
                elif self.last_input_size != 0:
                    for token_id in input_ids[0][self.last_input_size:]:
                        grammar.advance(int(token_id))
                self.last_input_size = input_size

            return grammar.update_scores(scores)
        return scores


def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    processor_list.append(TemplatingLogitsProcessor())
    return processor_list


def token_generated_callback(input_ids, scores):
    if params["enabled"]:
        grammar: Grammar = params["grammar"]
        new_token = input_ids[0][-1]
        grammar.advance(int(new_token))


def input_modifier(string, state, is_chat=False):
    """
    Initializes template and appends initial simple text to input.
    Note: In chat_mode, this extension does nothing.
    """
    if not is_chat:
        if "grammar" in state or params["template"]:
            grammar: Grammar = params["grammar"]
            if "grammar" in state:
                grammar.reset(state["grammar"] or EMPTY_GRAMMAR)
            else:
                grammar.reset(params["template"])
            params["enabled"] = True
        else:
            params["enabled"] = False
    return string


def ui():
    import gradio as gr
    output_template = gr.Textbox(value="", placeholder="Enter output template", label="Output Template",
                               info='Output Template to use', lines=5)

    def update_output_template(x):
        logger.info("output_template updated")
        params.update({"template": x})

    output_template.change(update_output_template, output_template, None)
