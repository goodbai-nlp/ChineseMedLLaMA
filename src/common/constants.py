# coding:utf-8
import os
import json


raw_special_tokens = json.load(
    open(f"{os.path.dirname(__file__)}/additional-tokens.json", "r", encoding="utf-8")
)
special_tokens = [itm.lstrip("Ġ") for itm in raw_special_tokens]