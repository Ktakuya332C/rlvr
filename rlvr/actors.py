import ray
import torch
import warnings
import numpy as np
from ray.util import ActorPool
from transformers import AutoTokenizer, AutoModelForCausalLM

from rlvr.dist import TorchDistActor


@ray.remote
class Tokenizer:

    def __init__(self, tokenizer_path, padding_side="left"):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    @ray.method(num_returns=2)
    def process(self, texts, padding_side="left", apply_chat_template=True):
        self._tokenizer.padding_side = padding_side
        if apply_chat_template:
            texts = [[{"role": "user", "content": t}] for t in texts.tolist()]
            input_ids = self._tokenizer.apply_chat_template(
                texts, return_tensors="np", padding=True
            )
            attention_mask = (input_ids != self._tokenizer.pad_token_id).astype(
                np.int64
            )
            return input_ids, attention_mask
        inputs = self._tokenizer(texts.tolist(), return_tensors="np", padding=True)
        return inputs["input_ids"], inputs["attention_mask"]


@ray.remote
class DeTokenizer:

    def __init__(self, tokenizer_path):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def process(self, tokens, attention_mask):
        texts = self._tokenizer.batch_decode(
            tokens * attention_mask, skip_special_tokens=True
        )
        return np.array(texts)


@ray.remote
class RolloutWorker(TorchDistActor):

    def __init__(self, model_path):
        self._model = AutoModelForCausalLM.from_pretrained(model_path)

    @ray.method(num_returns=2)
    def process(self, input_ids, attention_mask, max_length):
        self._model.generation_config.eos_token_id = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            input_ids_torch = torch.from_numpy(input_ids)
            attention_mask_torch = torch.from_numpy(attention_mask)
        outputs = self._model.generate(
            input_ids=input_ids_torch,
            attention_mask=attention_mask_torch,
            max_length=max_length,
        )
        output_mask = _rollout_postprocess(
            outputs=outputs.numpy(),
            input_mask=attention_mask,
            eos_token_id=self._model.config.eos_token_id,
        )
        return outputs, output_mask


def _rollout_postprocess(outputs, input_mask, eos_token_id):
    output_eos_mask = (outputs == eos_token_id).astype(np.int64)
    output_eos_mask[:, : input_mask.shape[1]] *= 1 - input_mask
    output_mask = ((np.cumsum(output_eos_mask, axis=1) - output_eos_mask) == 0).astype(
        np.int64
    )
    output_mask[:, : input_mask.shape[1]] = 0
    return output_mask


@ray.remote
class RolloutDispatcher:

    def __init__(self, rollout_workers):
        self._pool = ActorPool(rollout_workers)

    @ray.method(num_returns=2)
    def process(self, input_ids, attention_mask, batch_size, max_length):
        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) % batch_size == 0
        fn = lambda a, v: a.process.remote(v[0], v[1], max_length)
        args = []
        for bgn in range(0, len(input_ids), batch_size):
            arg = (
                input_ids[bgn : bgn + batch_size],
                attention_mask[bgn : bgn + batch_size],
            )
            args.append(arg)
        outputs_list = []
        output_mask_list = []
        for outputs, output_mask in self._pool.map(fn, args):
            outputs_list.append(outputs)
            output_mask_list.append(output_mask)
        return np.concatenate(outputs_list), np.concatenate(output_mask_list)
