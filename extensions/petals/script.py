"""
An example of extension. It does nothing, but you can add transformations
before the return statements to customize the webui behavior.

Starting from history_modifier and ending in output_modifier, the
functions are declared in the same order that they are called at
generation time.
"""

import gradio as gr
import torch
from transformers import LogitsProcessor

from modules import chat, shared
from modules.text_generation import (
    decode,
    encode,
    generate_reply,
)

from modules.logging_colors import logger

from transformers import AutoTokenizer
from petals import AutoDistributedModelForCausalLM
from huggingface_hub import login

params = {
    "display_name": "Petals",
    "is_tab": True,
}

class MyLogits(LogitsProcessor):
    """
    Manipulates the probabilities for the next token before it gets sampled.
    Used in the logits_processor_modifier function below.
    """
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        # probs = torch.softmax(scores, dim=-1, dtype=torch.float)
        # probs[0] /= probs[0].sum()
        # scores = torch.log(probs / (1 - probs))
        return scores

def history_modifier(history):
    """
    Modifies the chat history.
    Only used in chat mode.
    """
    return history

def state_modifier(state):
    """
    Modifies the state variable, which is a dictionary containing the input
    values in the UI like sliders and checkboxes.
    """
    return state

def chat_input_modifier(text, visible_text, state):
    """
    Modifies the user input string in chat mode (visible_text).
    You can also modify the internal representation of the user
    input (text) to change how it will appear in the prompt.
    """
    return text, visible_text

def input_modifier(string, state, is_chat=False):
    """
    In default/notebook modes, modifies the whole prompt.

    In chat mode, it is the same as chat_input_modifier but only applied
    to "text", here called "string", and not to "visible_text".
    """
    return string

def bot_prefix_modifier(string, state):
    """
    Modifies the prefix for the next bot reply in chat mode.
    By default, the prefix will be something like "Bot Name:".
    """
    return string

def tokenizer_modifier(state, prompt, input_ids, input_embeds):
    """
    Modifies the input ids and embeds.
    Used by the multimodal extension to put image embeddings in the prompt.
    Only used by loaders that use the transformers library for sampling.
    """
    return prompt, input_ids, input_embeds

def logits_processor_modifier(processor_list, input_ids):
    """
    Adds logits processors to the list, allowing you to access and modify
    the next token probabilities.
    Only used by loaders that use the transformers library for sampling.
    """
    processor_list.append(MyLogits())
    return processor_list

def output_modifier(string, state, is_chat=False):
    """
    Modifies the LLM output before it gets presented.

    In chat mode, the modified version goes into history['visible'],
    and the original version goes into history['internal'].
    """
    return string

def custom_generate_chat_prompt(user_input, state, **kwargs):
    """
    Replaces the function that generates the prompt from the chat history.
    Only used in chat mode.
    """
    result = chat.generate_chat_prompt(user_input, state, **kwargs)
    return result

# def custom_css():
#     path_to_css = Path(__file__).parent.resolve() / 'style.css'
#     return open(path_to_css, 'r').read()

def custom_js():
    """
    Returns a javascript string that gets appended to the javascript
    for the webui.
    """
    return ''

def setup():
    """
    Gets executed only once, when the extension is imported.
    """
    pass

def ui():
    """
    Gets executed when the UI is drawn. Custom gradio elements and
    their corresponding event handlers should be defined here.

    To learn about gradio components, check out the docs:
    https://gradio.app/docs/
    """
    # with gr.Accordion("Petals", open=True):
    with gr.Row():
        with gr.Column():
            shared.gradio['petals_model_menu'] = gr.Textbox(label="choose model",
                                                            info="Enter the model you would like to run. Models health"
                                                                 " can be found [here](https://health.petals.dev/)",
                                                            value="petals-team/StableBeluga2")
            shared.gradio['petals_hf_token'] = gr.Textbox(label="Hugging face Token",
                                                          placeholder="If you want to use Llama 2, insert token here,"
                                                                      " else leave blank")
            with gr.Row():
                shared.gradio['petals_status'] = gr.Markdown('ready')
                with gr.Column():
                    shared.gradio['petals_load'] = gr.Button("Load petal model")
    shared.gradio['petals_load'].click(load_petal_model,
                                      inputs=[shared.gradio['petals_model_menu'], shared.gradio['petals_hf_token']],
                                      outputs=shared.gradio['petals_status'])

def load_petal_model(model_name, token):
    shared.model_name = "petals: " + model_name

    print(f"{token=}")

    if token != '':
        logger.info('logging in to Hugging Face')
        print(token)
        try:
            login(token=token)
        except(ValueError):
            raise gr.Error("Login failed. Make sure you have the right token")

    logger.info("Loading Petals model \'"+ model_name + "\'")
    shared.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype=torch.float32)
    shared.model = AutoDistributedModelForCausalLM.from_pretrained(model_name)
    shared.model = shared.model.cuda()
    yield "Model was loaded successfully"
