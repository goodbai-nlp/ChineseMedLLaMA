# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import logging
import torch
import json
import sys
import os
import re
from nltk import Tree
from tqdm import tqdm
from transformers import (
    AutoConfig,
    LlamaConfig,
    LlamaTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    AutoTokenizer,
)
from typing import Optional, List, Tuple
from vllm import EngineArgs, LLMEngine, SamplingParams, RequestOutput
from vllm.lora.request import LoRARequest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger(__name__)
world_size = torch.cuda.device_count()


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name",
        type=str,
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--lora_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained model",
        required=False,
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model",
        required=True,
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test_file",
        required=True,
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="testset",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--beam_search",
        action="store_true",
        default=False,
        help="use beam search or not",
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--ngpus",
        type=int,
        default=1,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=4,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Specify num of beams",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=400,
        help="Specify num of return sequences",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument(
        "--use_deepspeed", action="store_true", default=False, help="whether to use deepspeed for inference"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        help="testset",
        required=True,
    )
    parser.add_argument(
        "--instruction",
        type=str,
        default="Generate the constituent tree for a given sentence.",
        help="testset",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="None",
        help="testset",
    )
    parser.add_argument(
        "--input_key",
        type=str,
        default="sentence",
        help="input key of test set",
    )
    parser.add_argument(
        "--decode_special_token",
        action="store_true",
        default=False,
        help="whether to decode special tokens or not"
    )
    parser.add_argument(
        "--bit_8",
        action="store_true", default=False
    )

    args = parser.parse_args()

    return args


def post_process(text):
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text


webnlg_template="""<s>[INST] Following is a set of knowledge graph triples delimited by triple backticks, each on a separate line, in the format: subject | predicate | object.
```
{triples}
```

Generate a coherent piece of text that contains all of the information in the triples. Only use information from the provided triples.[/INST]"""


def create_prompt(data, args, tokenizer=None):
    if args.prompt_template == "chinese-llama3-chat":
        system_prompt = "你是一个医疗助手，针对用户的问题给出有用且安全的回复。"
        assert tokenizer is not None, "tokenizer should not be None when using {args.prompt_template} template."
        prompts = []
        for sample in data:
            chat = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": sample[args.input_key].rstrip()},
            ]
            prompts.append(tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
    else:
        print(f"Invalid Prompt template:{args.prompt_template}, exit ...")
        exit()
    return prompts


def initialize_engine(args) -> LLMEngine:
    """Initialize the LLMEngine."""
    # max_loras: controls the number of LoRAs that can be used in the same
    #   batch. Larger numbers will cause higher memory usage, as each LoRA
    #   slot requires its own preallocated tensor.
    # max_lora_rank: controls the maximum supported rank of all LoRAs. Larger
    #   numbers will cause higher memory usage. If you know that all LoRAs will
    #   use the same rank, it is recommended to set this as low as possible.
    # max_cpu_loras: controls the size of the CPU LoRA cache.
    engine_args = EngineArgs(model=args.model_name_or_path,
                             enable_lora=True if args.lora_name_or_path else False,
                             tensor_parallel_size=world_size,
                             trust_remote_code=True,
                             gpu_memory_utilization=0.90,
                             max_loras=1,
                             max_lora_rank=64,
                             max_cpu_loras=2,
                             max_num_seqs=256)
    return LLMEngine.from_engine_args(engine_args)


def process_requests(engine: LLMEngine, sampling_params: SamplingParams, test_prompts: List[str], lora_request: Optional[LoRARequest]):
    """Continuously process a list of prompts and handle the outputs."""
    request_id = 0
    gen_ids = [[] for _ in range(len(test_prompts))]
    gen_res = ["" for _ in range(len(test_prompts))]
    prompt_res = ["" for _ in range(len(test_prompts))]
    while test_prompts or engine.has_unfinished_requests():
        if test_prompts:
            prompt = test_prompts.pop(0)
            engine.add_request(str(request_id),
                               prompt,
                               sampling_params,
                               lora_request=lora_request)
            request_id += 1
            
        request_outputs: List[RequestOutput] = engine.step()
        for request_output in request_outputs:
            if request_output.finished:
                # print(request_output)
                prompt = request_output.prompt
                iid = int(request_output.request_id)
                generated_text = request_output.outputs[0].text.replace("\n", "")
                gen_ids[iid] = request_output.outputs[0].token_ids
                gen_res[iid] = generated_text
                prompt_res[iid] = prompt
    
    return prompt_res, gen_res, gen_ids


def main():
    args = parse_args()
    lora_request = LoRARequest("llama-lora", 1, args.lora_name_or_path) if args.lora_name_or_path else None
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, fast_tokenizer=False, add_eos_token=False, trust_remote_code=True)
    model_engine = initialize_engine(args)
    print(args.test_file)
    
    if args.test_file.endswith("jsonl"):
        with open(args.test_file, "r", encoding="utf-8") as fin:
            data = [json.loads(line.strip()) for line in fin]
            prompts = create_prompt(data, args, tokenizer)
    elif args.test_file.endswith("json"):
        with open(args.test_file, "r", encoding="utf-8") as fin:
            data = json.load(fin)
            prompts = create_prompt(data, args, tokenizer)
    else:
        print("unsupported file format")

    print(f"Loaded {len(prompts)} data for generation")
    print(f"Example data: {prompts[:5]}")
    # exit()
    
    if args.beam_search:
        gen_params = SamplingParams(n=args.num_beams, use_beam_search=True, temperature=0.0, best_of=args.num_beams, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>", "<|end_of_text|>", tokenizer.eos_token])
    else:
        gen_params = SamplingParams(n=1, use_beam_search=False, best_of=1, temperature=0, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>", "<|end_of_text|>", "###", tokenizer.eos_token])

    prompt_res, gen_res, gen_ids = process_requests(model_engine, gen_params, prompts, lora_request)
    
    if args.decode_special_token:
        gen_res = []
        for idx in tqdm(range(0, len(gen_ids), 100)):
            ith_gen_res = tokenizer.batch_decode(gen_ids[idx:idx+100], skip_special_tokens=False)
            gen_res += [itm.replace("\n", " ").replace(tokenizer.eos_token, "") for itm in ith_gen_res]
    
    out_prefix = args.test_file.split("/")[-1].split(".")[0]
    
    if args.out_path is not None:
        out_path = args.out_path
        if not os.path.exists(out_path):
            os.mkdir(out_path)
    else:
        out_path = args.model_name_or_path
        
    out_file = f"{out_path}/{args.out_prefix}_{out_prefix}_vllm.txt" if args.lora_name_or_path is None else f"{args.lora_name_or_path}/{args.out_prefix}_{out_prefix}_vllm.txt"
    with open(out_file, "w", encoding="utf-8") as fout:
        fout.write("\n".join(gen_res) + "\n")


if __name__ == "__main__":
    main()