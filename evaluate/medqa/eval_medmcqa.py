# coding:utf-8
import sys
import json

with open(sys.argv[1], 'r', encoding='utf-8') as fgold:
    gold_data = [json.loads(line.strip()) for line in fgold]
    
with open(sys.argv[2], 'r', encoding='utf-8') as fpred:
    pred_data = [line.strip() for line in fpred]
    

assert len(gold_data) == len(pred_data), f"Gold: {len(gold_data)} vs Pred: {len(pred_data)}"


def calc_acc(gold_data, pred_data):
    right = 0
    # pred_res = {}
    label_set={"A.", "B.", "C.", "D."}
    for gold, pred in zip(gold_data, pred_data):
        gold_label = gold["output"] + "."
        rest_set = list(label_set - {gold_label})
        if gold_label in pred and (rest_set[0] not in pred) and (rest_set[1] not in pred) and (rest_set[2] not in pred):
            right += 1
        else:
            print(f"Gold: {gold_label}\nPred: {pred}\n")
        # pred_res[gold['id']] = pred
        # if pred.startswith(gold_label):
        #     right += 1
    print(f"acc:{right}/{len(gold_data)}, {right/len(gold_data)}")


calc_acc(gold_data, pred_data)