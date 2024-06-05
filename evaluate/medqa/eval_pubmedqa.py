# coding:utf-8
import sys
import json

with open(sys.argv[1], 'r', encoding='utf-8') as fgold:
    gold_data = [json.loads(line.strip()) for line in fgold]
    
with open(sys.argv[2], 'r', encoding='utf-8') as fpred:
    pred_data = [line.strip().lower() for line in fpred]


def calc_acc(gold_data, pred_data):
    assert len(gold_data) == len(pred_data)
    right = 0
    pred_res = {}
    label_set={"yes", "no", "maybe"}
    for gold, pred in zip(gold_data, pred_data):
        gold_label = gold["output"]
        rest_set = list(label_set - {gold_label})
        # if pred.startswith(gold_label):
        if gold_label in pred and (rest_set[0] not in pred) and (rest_set[1] not in pred):
            right += 1
        else:
            print(f"Gold: {gold_label}\nPred: {pred}")
        # pred_res[gold['id']] = pred
    print("acc:", right/len(gold_data))
    return pred_res

pred_res = calc_acc(gold_data, pred_data)

# with open(sys.argv[3], 'w', encoding='utf-8') as fout:
#     json.dump(pred_res, fout, indent=4)