import re
import ray
import torch
import warnings
import numpy as np
from ray.util import ActorPool
from torch.nn import functional as F
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

    @ray.method(num_returns=3)
    def process(
        self,
        input_ids,
        attention_mask,
        max_length,
        do_sample=False,
        temperature=1.0,
        num_return_sequences=1,
    ):
        self._model.generation_config.eos_token_id = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            input_ids_torch = torch.from_numpy(input_ids)
            attention_mask_torch = torch.from_numpy(attention_mask)
        input_outputs = self._model.generate(
            input_ids=input_ids_torch,
            attention_mask=attention_mask_torch,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
        input_output_mask, output_mask = _rollout_postprocess(
            input_outputs=input_outputs.numpy(),
            input_mask=attention_mask,
            eos_token_id=self._model.config.eos_token_id,
        )
        return input_outputs, input_output_mask, output_mask


def _rollout_postprocess(input_outputs, input_mask, eos_token_id):
    input_output_eos_mask = (input_outputs == eos_token_id).astype(np.int64)
    input_output_eos_mask[:, : input_mask.shape[1]] *= 1 - input_mask
    input_output_mask = (
        (np.cumsum(input_output_eos_mask, axis=1) - input_output_eos_mask) == 0
    ).astype(np.int64)
    input_output_mask[:, : input_mask.shape[1]] = input_mask
    output_mask = input_output_mask.copy()
    output_mask[:, : input_mask.shape[1]] = 0
    return input_output_mask, output_mask


@ray.remote
class RolloutDispatcher:

    def __init__(self, rollout_workers):
        self._pool = ActorPool(rollout_workers)

    @ray.method(num_returns=3)
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
        input_outputs_list = []
        input_output_mask_list = []
        output_mask_list = []
        for input_outputs, input_output_mask, output_mask in self._pool.map(fn, args):
            input_outputs_list.append(input_outputs)
            input_output_mask_list.append(input_output_mask)
            output_mask_list.append(output_mask_list)
        return (
            np.concatenate(input_outputs_list),
            np.concatenate(input_output_mask_list),
            np.concatenate(input_output_mask_list),
        )


@ray.remote
class LastIntScorer:

    def process(self, responses, answers):
        assert len(responses) == len(answers)
        scores = np.zeros(len(responses))
        for idx, (response, answer) in enumerate(zip(responses, answers)):
            m = re.findall(r"\d+", response)
            if m and m[-1] != "" and m[-1] == answer:
                scores[idx] = 1.0
        return scores


@ray.remote
class ReferenceWorker:

    def __init__(self, model_path):
        self._model = AutoModelForCausalLM.from_pretrained(model_path)

    def process(
        self,
        input_output_ids,
        input_output_mask,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            input_output_ids_torch = torch.from_numpy(input_output_ids)
            input_output_mask_torch = torch.from_numpy(input_output_mask)
        with torch.no_grad():
            result = self._model(
                input_ids=input_output_ids_torch,
                attention_mask=input_output_mask_torch,
            )
        probs = F.softmax(result.logits.detach(), dim=-1)
        ref_probs = torch.gather(
            input=probs,
            dim=-1,
            index=input_output_ids_torch.unsqueeze(-1),
        ).squeeze(-1)
        return ref_probs.numpy()


@ray.remote
class ReferenceDispatcher:

    def __init__(self, reference_workers):
        self._pool = ActorPool(reference_workers)

    def process(self, input_output_ids, input_output_mask, batch_size):
        assert len(input_output_ids) == len(input_output_mask)
        assert len(input_output_ids) % batch_size == 0
        fn = lambda a, v: a.process.remote(v[0], v[1])
        args = []
        for bgn in range(0, len(input_output_ids), batch_size):
            arg = (
                input_output_ids[bgn : bgn + batch_size],
                input_output_mask[bgn : bgn + batch_size],
            )
            args.append(arg)
        ref_probs_list = []
        for ref_probs in self._pool.map(fn, args):
            ref_probs_list.append(ref_probs)
        return np.concatenate(ref_probs_list)
