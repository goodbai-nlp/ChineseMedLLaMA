# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
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
from vllm import LLM, SamplingParams

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_name",
        type=str,
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
        "--instruction",
        default="Generate the constituent tree for a given sentence.",
        type=str,
        help="input file",
    )
    parser.add_argument(
        "--sys_instruction",
        default="",
        type=str,
        help="input file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        help="Path to test_file",
        required=True,
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
        "--bit_8",
        action="store_true", default=False
    )

    args = parser.parse_args()

    return args


def create_prompt(data, args, tokenizer):
    if args.prompt_template == "None":
        prompts = [itm[args.input_key] for itm in data]
    elif args.prompt_template == "llama2-chat":
        system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
        prompts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{sample[args.input_key].rstrip()} [/INST] " for sample in data]
    elif args.prompt_template == "chat-v1":
        if not len(args.sys_instruction):
            chats = []
        else:
            chats = [{"role":"system", "content": args.sys_instruction}]

        task_instruction = args.instruction if "instruction" not in data[0] else data[0]["instruction"]
        prompts = [
            tokenizer.apply_chat_template(chats + [{"role": "user", "content": f"{sample[args.input_key]} {task_instruction}"}], tokenize=False)
            for sample in data
        ]
    elif args.prompt_template == "pubmedqa":
        prompts = [f"{sample[args.input_key].rstrip()}\nQuestion:\n{sample['QUESTION']}\nPlease respond with yes, no or maybe. The answer to the question is:" for sample in data]
    elif args.prompt_template == "supervised":
        prompts = [
            f"Human:\n{args.instruction} {sample[args.input_key]}\nAssistant:\n"
            for sample in data
        ]
    elif args.prompt_template == "supervised-v2":
        sys_instruction = ""
        task_instruction = args.instruction if "instruction" not in data[0] else data[0]["instruction"]
        prompts = [
            f'{sys_instruction}### Human: {task_instruction} {sample[args.input_key]}\n### Assistant: '
            for sample in data
        ]
    elif args.prompt_template == "vicuna":
        prompts = [
            f"USER: {args.instruction}{sample[args.input_key]} ASSISTANT:"
            for sample in data
        ]
    elif args.prompt_template == "gpt":
        prompts = [
            f"Human:\n{args.instruction}{sample[args.input_key]}\nAssistant:\n"
            for sample in data
        ]
    elif args.prompt_template == "one_shot":
        prompts = [
            f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\n3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.\n4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.\n5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.\n6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.\n7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.\n8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: {sample[args.input_key]}\n### Assistant:"
            for sample in data
        ]
    elif args.prompt_template == "one_shot_sim":
        prompts = [
            f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: {sample[args.input_key]}\n### Assistant:"
            for sample in data
        ]
    elif args.prompt_template == "one_shot_amr":
        task_instruction = args.instruction
        prompts = [
            f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n### Human: Got any creative ideas for a 10 year old’s birthday?\n### Assistant: Of course! Here are some creative ideas for a 10-year-old's birthday party:\n1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.\n2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.\nRemember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!\n### Human: {task_instruction} {sample['amr']}\n### Assistant:"
            for sample in data
        ]
    elif args.prompt_template == "five_shot_amr":
        sys_instruction = 'The Human provides a linearized directed sematic graph where the nodes such as "z1 give-01" are concepts and the edges such as ":ARG0-of" represent semantic relations, and the Assistant generates a descriptive text for the given graph.'
        examples = [
            ("( z0 multi-sentence :snt1 ( z1 give-01 :ARG0 ( z2 history ) :ARG1 ( z3 lesson :ARG1-of ( z4 have-quant-91 :ARG2 ( z5 many ) :ARG3 ( z6 too ) ) ) :ARG2 ( z7 we ) :polarity ( z8 amr-unknown ) ) :snt2 ( z9 and :op1 530 :op2 412 :op3 64 ) )", "Has history given us too many lessons?, 530, 412, 64"),
            ("( z0 and :op1 ( z1 employ-01 :ARG0 ( z2 it ) :ARG1 ( z3 person :quant 2700 ) ) :op2 ( z4 have-03 :ARG0 z2 :ARG1 ( z5 revenue :quant ( z6 rate-entity-91 :ARG1 ( z7 about :op1 ( z8 monetary-quantity :quant 370000000 :unit ( z9 dollar ) ) ) :ARG2 ( z10 temporal-quantity :quant 1 :unit ( z11 year ) ) ) ) ) )", "It employs 2,700 people and has annual revenue of about $ 370 million ."),
            ("( z0 yield-03 :ARG0 ( z1 fund :mod ( z2 top ) :mod ( z3 money ) ) :ARG1 ( z4 over :op1 ( z5 percentage-entity :value 9 ) :degree ( z6 well ) ) :time ( z7 current ) )", "he top money funds are currently yielding well over 9 % ."),
            ("( z0 expose-01 :ARG1 ( z1 person :ARG0-of ( z2 work-01 :location ( z3 factory :ARG0-of ( z4 make-01 :ARG1 ( z5 paper :purpose ( z6 product :wiki \"Kent_(cigarette)\" :name ( z7 name :op1 \"Kent\" ) :ARG0-of ( z8 filter-02 ) ) ) ) ) ) :quant ( z9 about :op1 160 ) ) :ARG2 ( z10 asbestos ) :time ( z11 date-entity :decade 1950 ) )", "About 160 workers at a factory that made paper for the Kent filters were exposed to asbestos in the 1950s ."),
            ("( z0 multi-sentence :snt1 ( z1 good-02 :polarity - :ARG1 ( z2 that ) ) :snt2 ( z3 possible-01 :ARG1 ( z4 speak-01 :ARG0 ( z5 we ) :manner ( z6 amr-unknown ) :manner ( z7 trustworthy :polarity - :mod ( z8 such ) ) ) :ARG1-of ( z9 cause-01 :ARG0 ( z10 country :mod ( z11 big ) :ARG1-of ( z12 responsible-02 ) :domain z5 ) ) ) :snt3 ( z13 cause-01 :ARG0 ( z14 important-01 :ARG1 ( z15 develop-02 :ARG1 ( z16 economy ) ) :ARG0-of ( z17 override-01 ) ) :ARG1 ( z18 and :op1 ( z19 let-01 :mode imperative :ARG0 ( z20 you ) :ARG1 ( z21 matter ) ) :op2 ( z22 mention-01 :polarity - :ARG0 z20 :ARG1 z21 :mod ( z23 again ) ) ) ) )", "That is not good. How can we speak in such a untrustworthy manner, we are a responsible big country - let the matter be, do not mention it again, to develop the economy is of overriding importance."),
        ]
        prompts = [
            f"{sys_instruction}\n### Human: {examples[0][0]}\n### Assistant: {examples[0][1]}\n### Human: {examples[1][0]}\n### Assistant: {examples[1][1]}\n### Human: {examples[2][0]}\n### Assistant: {examples[2][1]}\n### Human: {examples[3][0]}\n### Assistant: {examples[3][1]}\n### Human: {examples[4][0]}\n### Assistant: {examples[4][1]}\n### Human: {sample['amr']}\n### Assistant:"
            for sample in data
        ]
    elif args.prompt_template == "finetune_amr":
        sys_instruction = 'The following is a linearized directed sematic graph where the nodes such as "z2 important-01" denote concepts and the edges such as ":ARG0-of" denote semantic relations.'
        sample = data[0]
        task_instruction = args.instruction if "instruction" not in sample else sample["instruction"]
        prompts = [f'{sys_instruction}\n### Human: {task_instruction} {sample[args.input_key]}\n### Assistant: ' for sample in data]
    else:
        print(f"Invalid Prompt template:{args.prompt_template}, exit ...")
        exit()
    return prompts


def tree_to_lisp(tree):
    """
    Converts an NLTK Tree to a Lisp-style string.
    """
    if isinstance(tree, Tree):
        children_str = ' '.join(tree_to_lisp(t) for t in tree)
        return f"({tree.label()} {children_str})"
    else:
        return tree


def postprocess_text(text):
    #return text
    text = text.lower()
    text = ' '.join(re.split('(\W)', text))
    text = ' '.join(text.split())
    return text

world_size = torch.cuda.device_count()


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, fast_tokenizer=False, add_eos_token=False)
    model = LLM(model=args.model_name if args.model_name else args.model_name_or_path, download_dir=args.model_name_or_path, tensor_parallel_size=world_size, gpu_memory_utilization=0.90, swap_space=1)
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
    
    if args.beam_search:
        gen_params = SamplingParams(n=args.num_beams, use_beam_search=True, temperature=0.0, best_of=args.num_beams, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>", "<|eot_id|>"])
    else:
        gen_params = SamplingParams(n=1, use_beam_search=False, best_of=1, temperature=0, frequency_penalty=args.frequency_penalty, presence_penalty=args.presence_penalty, max_tokens=args.max_new_tokens, stop=["</s>", "<pad>", "<|endoftext|>", "###", "<|eot_id|>"])

    gen_res = ["" for _ in range(len(prompts))]
    prompt_res = ["" for _ in range(len(prompts))]
    outputs = model.generate(prompts, gen_params)
    # print("Output:", outputs)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        iid = int(output.request_id)
        # print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        # gen_res.append(generated_text)
        
        gen_res[iid] = generated_text.replace("\n", "")
        prompt_res[iid] = prompt
    
    out_prefix = args.test_file.split("/")[-1].split(".")[0]
    with open(f"{args.model_name_or_path}/{args.out_prefix}_{out_prefix}_vllm.txt", "w", encoding="utf-8") as fout:
        fout.write("\n".join(gen_res) + "\n")


if __name__ == "__main__":
    main()
