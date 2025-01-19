import re
import ray
import torch
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from rlvr.dist import TorchDistActor


@ray.remote
class TokenizeActor:

    def __init__(self, tokenizer_path, **kwargs):
        self._tok = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)

    def map(self, item, in_key="text", out_key="tokens"):
        in_text = item[in_key]
        out = self._tok(in_text, return_tensors="np", return_attention_mask=False)
        out_text = out["input_ids"][0]
        item[out_key] = out_text
        return item


@ray.remote
class RolloutActor(TorchDistActor):

    def __init__(self, model_path, pad_token_id):
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        self._pad_token_id = pad_token_id

    def map_batch(self, batch, in_key="query", out_key="query_response", **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            in_tokens = [torch.from_numpy(item[in_key]) for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            sequences=in_tokens,
            batch_first=True,
            padding_value=self._pad_token_id,
        )
        attention_mask = input_ids != self._pad_token_id
        outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        for item, output in zip(batch, outputs):
            item[out_key] = output.numpy()
        return batch


@ray.remote
class DeTokenizeActor:

    def __init__(self, tokenizer_path, **kwargs):
        self._tok = AutoTokenizer.from_pretrained(tokenizer_path, **kwargs)

    def map(self, item, in_key="tokens", out_key="text"):
        in_tokens = item[in_key]
        out_text = self._tok.decode(in_tokens, skip_special_tokens=True)
        item[out_key] = out_text
        return item


@ray.remote
class LastIntRewardActor:

    def map(
        self, item, response_key="response", answer_key="answer", reward_key="reward"
    ):
        response = item[response_key]
        answer_str = str(item[answer_key])
        m = re.findall(r"\d+", response)
        last_int_str = m[-1] if m else ""
        item[reward_key] = int(
            answer_str == last_int_str and answer_str != "" and last_int_str != ""
        )
        return item
