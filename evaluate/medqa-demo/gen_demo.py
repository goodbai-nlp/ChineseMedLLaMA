# coding:utf-8

import json
import sys
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

model_name_or_path=sys.argv[1]

config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
print(f"Loading tokenizer from {model_name_or_path}")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=False, add_eos_token=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
model.cuda()

user_query = "我已经被痤疮的问题困扰很长时间了，试了很多办法，但是治疗效果并不好，想了解一下，痤疮都有哪些类型呢？"

while True:
    user_query = input("请输入你的问题: ")
    if user_query == "none":
        break
    chat = [
        {"role": "system", "content": "你是一个医疗助手，针对用户的问题给出有用且安全的回复。"},
        {"role": "user", "content": user_query.rstrip()},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    with torch.inference_mode():
        inputs = tokenizer(prompt, return_tensors="pt").to(torch.cuda.current_device())
        generate_ids = model.generate(
            **inputs,
            num_beams=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            max_new_tokens=256,
        )
        ori_output = tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        new_ids = generate_ids[:, inputs.input_ids.size(1):]
        output = tokenizer.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    print("Assistant: ", output)
    