from extensions.output_template.utils import shared, AllowedTokens
from extensions.output_template.template import Template
from functools import partial
try:
    import transformers
    from transformers import LogitsProcessor
    from modules.logging_colors import logger
except ModuleNotFoundError:
    # Allows testing by running script outside text-generation-ui
    from logging import Logger
    logger = Logger(__name__)
    transformers = None
    class LogitsProcessor:
        pass


params = {
    "output_template_obj": Template(""),
    "output_template": "",
    "token_dictionary": None,
    "used_tokenizer": None,
    "scores_size": 0,
}


class TemplatingLogitsProcessor(LogitsProcessor):
    def __call__(self, _: list[list[int]], scores: list[list[int]]):
        template = params["output_template_obj"]
        tag = template.get_next_tag()
        if tag:
            params["scores_size"] = len(scores[0])
            allowed = tag.get_allowed_tokens()
            allowed.apply(scores)
        else:
            # Ran out of template, force EOS
            AllowedTokens(allow_eos=True).apply(scores)
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
    template = params["output_template_obj"]
    new_token = input_ids[0][-1]
    template.move_forward(int(new_token))


def input_modifier(string, state, is_chat=False):
    """
    Initializes template and appends initial simple text to input.
    Note: In chat_mode, this extension does nothing.
    """
    if not is_chat:
        template = params["output_template_obj"]
        # If template is empty, <any/> is inserted just to not mess with generation
        template.init_template(params["output_template"].strip() or "<any/>")
        prefix = template.get_initial_text().text
        state['output_template_prefix'] = prefix
        string += prefix

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


def output_modifier(string, state, is_chat=False):
    if state.get('output_template_prefix'):
        string = state['output_template_prefix'] + string
        state['output_template_prefix'] = ""
    return string


def ui():
    import gradio as gr
    output_template = gr.Textbox(value="", placeholder="Enter output template", label="Output Template",
                               info='Output Template to use', lines=5)

    def update_output_template(x):
        logger.info("output_template updated")
        params.update({"output_template": x})

    output_template.change(update_output_template, output_template, None)
