from transformers import AutoTokenizer

input="You are a helpful medical assistant."
tokenizer = AutoTokenizer.from_pretrained("/home/export/base/ycsc_chenkh/chenkh_nvlink/online1/xfbai/data/pretrained-models/llama3-8b", add_eos_token=True)

tok_ids = tokenizer([input])["input_ids"]
print(tok_ids)
print(tokenizer.batch_decode(tok_ids))