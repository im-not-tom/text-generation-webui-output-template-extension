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


params = {
    "grammar": Grammar("root ::= [.]*"),
    "enabled": False,
    "template": "",
    "token_dictionary": None,
    "used_tokenizer": None,
    "scores_size": 0,
}


class TemplatingLogitsProcessor(transformers.LogitsProcessor):

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if params["enabled"]:
            params["scores_size"] = len(scores[0])
            grammar: Grammar = params["grammar"]
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
        if params["template"]:
            grammar: Grammar = params["grammar"]
            grammar.reset(params["template"])
            params["enabled"] = True
        else:
            params["enabled"] = False

        if not hasattr(shared.model, "__output_template_sample_method_patched"):
            # Very hacky way to inject my own callback after new token is sampled
            # This is needed to move state of template forward
            def wrapper(generate_orig, *a, **kwargs):
                assert kwargs['stopping_criteria']
                kwargs['stopping_criteria'].append(token_generated_callback)
                return generate_orig(*a, **kwargs)

            shared.model.generate = partial(wrapper, shared.model.generate)
            shared.model.__output_template_sample_method_patched = True
    return string


def ui():
    import gradio as gr
    output_template = gr.Textbox(value="", placeholder="Enter output template", label="Output Template",
                               info='Output Template to use', lines=5)

    def update_output_template(x):
        logger.info("output_template updated")
        params.update({"template": x})

    output_template.change(update_output_template, output_template, None)
