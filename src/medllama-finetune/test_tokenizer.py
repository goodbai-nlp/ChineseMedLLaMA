from transformers import AutoTokenizer

chat1 = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "who are you?"}
]
chat2 = [
    {"role": "system", "content": "You are a helpful medical assistant."},
    {"role": "user", "content": "who are you?"},
    {"role": "assistant", "content": "I am Lilith."},
]
tokenizer = AutoTokenizer.from_pretrained("/home/export/base/ycsc_chenkh/chenkh_nvlink/online1/xfbai/data/pretrained-models/llama3-8b-instruct")
full = tokenizer.apply_chat_template(chat2, tokenize=False, add_generation_prompt=False)
prompt = f'{tokenizer.apply_chat_template(chat1, tokenize=False, add_generation_prompt=True)}<|end_of_text|>'

prompt_ids = tokenizer([prompt], max_length=512, padding=False, truncation=True)["input_ids"]
full_ids = tokenizer([full], max_length=512, padding=False, truncation=True)["input_ids"]

labels = [
    [2 for _ in range(len(prompt_id)-1)] + input_id[(len(prompt_id)-1):] for input_id, prompt_id in zip(full_ids, prompt_ids)
]

print("prompt_ids", prompt_ids)
print("input_ids", full_ids)
print("label_ids", labels)

print(tokenizer.batch_decode(prompt_ids))
print(tokenizer.batch_decode(full_ids))
print(tokenizer.batch_decode(labels))
