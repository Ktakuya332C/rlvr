import re
import ray
import numpy as np
from datasets import load_dataset


def get_gsm8k(split="train"):

    def _extract_number(text):
        m = re.findall(r"\d+", text)
        return m[-1] if m else ""

    dataset = load_dataset("openai/gsm8k", name="main", split=split).map(
        lambda x: {"answer": _extract_number(x["answer"])}
    )
    return ray.data.from_huggingface(dataset)
