import dataclasses
from dataclasses import dataclass
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import requests

import sys
import pathlib

from barebonesllmchat.chatbot.bots.bot import _Bot

sys.path.append(str(pathlib.Path(__file__).parent.parent.resolve()))
from barebonesllmchat.common.chat_history import CHAT_ROLE, ChatHistory

from dataclasses import dataclass

@dataclass
class DefaultQwenSettings:
    max_new_tokens:int = 1024
    #do_sample:bool = True
    #top_k:int = 50
    #top_p:float = 0.95

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)

class Qwen2p5(_Bot):
    def __init__(self, model_string="Qwen/Qwen2.5-Coder-32B-Instruct-GPTQ-Int4", precision=torch.bfloat16):
        #model_string = "Qwen/Qwen2.5-Coder-14B"

        super().__init__(model_string, precision)

        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_string, 
            trust_remote_code=True,
            torch_dtype="auto",
            attn_implementation="flash_attention_2",
            device_map='auto',
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_string, 
                                                       trust_remote_code=True,
                                                      # torch_dtype=precision,
            device_map='auto'
            )
        print("Qwen2.5 loaded")

    def respond(self, chat, images=None, generation_settings=None):
        if images is not None:
            print("Images found, but Olmo is not multimodal. Images will be ignored.", file=sys.stderr)

        _generation_settings = vars(DefaultQwenSettings())
        if generation_settings is not None:
            _generation_settings.update(**generation_settings)
        del generation_settings

        print()
        print("-----")
        print("Generating with following settings:")
        print(_generation_settings)
        print("-----")
        print()

        messages = chat.to_lowercase_roles().history_without_images



        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.tokenizer([prompt], return_tensors="pt")

        #inputs = self.tokenizer(messages, return_tensors='pt', return_token_type_ids=False)
        #inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        model_inputs = inputs.to(self.model.device)

        # optional verifying cuda
        # inputs = {k: v.to('cuda') for k,v in inputs.items()}
        # olmo = olmo.to('cuda')
        generated_ids = self.model.generate(**model_inputs, **_generation_settings)

        generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

        generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        #response = response[:,inputs.shape[1]:]
        #generated_text = self.tokenizer.batch_decode(response, skip_special_tokens=True)[0]

        print()
        print("-----")
        print(generated_text)
        print(_generation_settings)
        print("-----")
        print()

        chat = chat.add(CHAT_ROLE.ASSISTANT, generated_text)
        return chat