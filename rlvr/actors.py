import re
import ray
import torch
import warnings
import numpy as np
from ray.util import ActorPool
from torch.nn import functional as F
from scipy.special import log_softmax
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoTokenizer, AutoModelForCausalLM

from rlvr.dist import TorchDistActor
from rlvr.loggers import NoLogger


@ray.remote
class Tokenizer:

    def __init__(self, tokenizer_path, padding_side="left", logger=None):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self._logger = logger if logger else NoLogger.remote()

    @ray.method(num_returns=2)
    def process(self, texts, max_length, padding_side="left", apply_chat_template=True):
        self._tokenizer.padding_side = padding_side
        if apply_chat_template:
            texts = [[{"role": "user", "content": t}] for t in texts.tolist()]
            input_ids = self._tokenizer.apply_chat_template(
                texts,
                return_tensors="np",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            attention_mask = (input_ids != self._tokenizer.pad_token_id).astype(
                np.int64
            )
            self._logger.store.remote("input_length", input_ids.shape[1])
            return input_ids, attention_mask
        inputs = self._tokenizer(
            texts.tolist(),
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        self._logger.store.remote("input_length", inputs["input_ids"].shape[1])
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

    def __init__(self, model_path, logger=None):
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        super().__init__(self._model)
        self._logger = logger if logger else NoLogger.remote()

    @ray.method(num_returns=4)
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
        results = self._model.generate(
            input_ids=input_ids_torch,
            attention_mask=attention_mask_torch,
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            return_dict_in_generate=True,
            output_logits=True,
            return_legacy_cache=False,
        )
        input_outputs = results["sequences"].numpy()
        attention_mask = attention_mask.repeat(num_return_sequences, axis=0)
        input_output_mask, output_mask, output_log_probs = _rollout_postprocess(
            input_outputs=input_outputs,
            input_mask=attention_mask,
            output_logits=torch.stack(results["logits"], axis=1).numpy(),
            eos_token_id=self._model.config.eos_token_id,
        )
        self._logger.store.remote("input_output_length", input_outputs.shape[1])
        return input_outputs, input_output_mask, output_mask, output_log_probs


def _rollout_postprocess(input_outputs, output_logits, input_mask, eos_token_id):
    input_output_eos_mask = (input_outputs == eos_token_id).astype(np.int64)
    input_output_eos_mask[:, : input_mask.shape[1]] *= 1 - input_mask
    input_output_mask = (
        (np.cumsum(input_output_eos_mask, axis=1) - input_output_eos_mask) == 0
    ).astype(np.int64)
    input_output_mask[:, : input_mask.shape[1]] = input_mask
    output_mask = input_output_mask.copy()
    output_mask[:, : input_mask.shape[1]] = 0

    output_softmax = log_softmax(output_logits, axis=-1)
    output_ids = input_outputs[:, input_mask.shape[1] :]
    output_log_probs = np.take_along_axis(
        arr=output_softmax,
        indices=np.expand_dims(output_ids, -1),
        axis=-1,
    ).squeeze(-1)
    output_log_probs_padded = np.zeros(output_mask.shape)
    output_log_probs_padded[:, input_mask.shape[1] - 1 : -1] = output_log_probs

    return input_output_mask, output_mask, output_log_probs_padded


@ray.remote
class RolloutDispatcher:

    def __init__(self, rollout_workers):
        self._pool = ActorPool(rollout_workers)

    @ray.method(num_returns=4)
    def process(
        self,
        input_ids,
        attention_mask,
        batch_size,
        max_length,
        do_sample=True,
        temperature=1.0,
        num_return_sequences=1,
    ):
        assert len(input_ids) == len(attention_mask)
        assert len(input_ids) % batch_size == 0
        fn = lambda a, v: a.process.remote(
            input_ids=v[0],
            attention_mask=v[1],
            max_length=max_length,
            do_sample=do_sample,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
        )
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
        output_log_probs_list = []
        for (
            input_outputs,
            input_output_mask,
            output_mask,
            output_log_probs,
        ) in self._pool.map(fn, args):
            input_outputs_list.append(input_outputs)
            input_output_mask_list.append(input_output_mask)
            output_mask_list.append(output_mask_list)
            output_log_probs_list.append(output_log_probs)
        return (
            np.concatenate(input_outputs_list),
            np.concatenate(input_output_mask_list),
            np.concatenate(input_output_mask_list),
            np.concatenate(output_log_probs_list),
        )


@ray.remote
class Replicator:

    def process(self, arr, num_replica):
        return arr.repeat(num_replica, axis=0)


@ray.remote
class LastIntScorer:

    def __init__(self, logger=None):
        self._logger = logger if logger else NoLogger.remote()

    def process(self, responses, answers):
        assert len(responses) == len(answers)
        scores = np.zeros(len(responses))
        for idx, (response, answer) in enumerate(zip(responses, answers)):
            m = re.findall(r"\d+", response)
            if m and m[-1] != "" and m[-1] == answer:
                scores[idx] = 1.0
        self._logger.store_list.remote("scores", scores)
        return scores


@ray.remote
class ReferenceWorker(TorchDistActor):

    def __init__(self, model_path):
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        super().__init__(self._model)

    def process(
        self,
        input_output_ids,
        input_output_mask,
        temperature=1.0,
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
        log_probs = F.log_softmax(result.logits.detach() / temperature, dim=-1)
        ref_log_probs = torch.gather(
            input=log_probs,
            dim=-1,
            index=input_output_ids_torch.unsqueeze(-1),
        ).squeeze(-1)
        return ref_log_probs.numpy()


@ray.remote
class ReferenceDispatcher:

    def __init__(self, reference_workers):
        self._pool = ActorPool(reference_workers)

    def process(
        self, input_output_ids, input_output_mask, temperature=1.0, batch_size=1
    ):
        assert len(input_output_ids) == len(input_output_mask)
        assert len(input_output_ids) % batch_size == 0
        fn = lambda a, v: a.process.remote(v[0], v[1], temperature)
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


@ray.remote
class GRPOLearner(TorchDistActor):

    def __init__(self, model_path, learning_rate=1e-6, logger=None):
        self._model = AutoModelForCausalLM.from_pretrained(model_path)
        super().__init__(self._model)
        self._optimizer = torch.optim.AdamW(
            params=self._model.parameters(),
            lr=learning_rate,
        )
        self._logger = logger if logger else NoLogger.remote()

    def distribute(self, group_name):
        assert group_name in self._groups
        pg = self._groups[group_name]
        self._model = DistributedDataParallel(self._model, process_group=pg)

    def process(
        self,
        num_generations,
        input_output_ids,
        input_output_mask,
        output_mask,
        output_log_probs,
        ref_log_probs,
        scores,
        ratios_clip_eps=0.2,
        scores_std_eps=1e-4,
        kl_loss_coef=0.05,
        temperature=1.0,
    ):
        assert len(input_output_ids) == len(input_output_mask)
        assert len(input_output_ids) == len(output_mask)
        assert len(input_output_ids) == len(output_log_probs)
        assert len(input_output_ids) == len(ref_log_probs)
        assert len(input_output_ids) == len(scores)
        assert len(input_output_ids) % num_generations == 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            input_output_ids_torch = torch.from_numpy(input_output_ids)
            input_output_mask_torch = torch.from_numpy(input_output_mask)
            output_mask_torch = torch.from_numpy(output_mask)
            output_log_probs_torch = torch.from_numpy(output_log_probs)
            ref_log_probs_torch = torch.from_numpy(ref_log_probs)
            scores_torch = torch.from_numpy(scores)

        # policy forward pass
        result = self._model(
            input_ids=input_output_ids_torch,
            attention_mask=input_output_mask_torch,
        )
        log_softmax_torch = F.log_softmax(result.logits / temperature, dim=-1)
        log_probs_torch = torch.gather(
            input=log_softmax_torch,
            dim=-1,
            index=input_output_ids_torch.unsqueeze(-1),
        ).squeeze(-1)

        # calculate loss
        policy_loss, kl_loss = _grpo_loss(
            num_generations=num_generations,
            log_probs=log_probs_torch,
            ref_log_probs=ref_log_probs_torch,
            output_log_probs=output_log_probs_torch,
            output_mask=output_mask_torch,
            scores=scores_torch,
            ratios_clip_eps=ratios_clip_eps,
            scores_std_eps=scores_std_eps,
        )
        loss = policy_loss + kl_loss_coef * kl_loss
        loss.backward()

        self._logger.store.remote("policy_loss", policy_loss.item())
        self._logger.store.remote("kl_loss", kl_loss.item())
        self._logger.store.remote("loss", loss.item())
        return loss.item()

    def update(self, loss_):
        self._optimizer.step()
        self._optimizer.zero_grad()


def _grpo_loss(
    num_generations,
    log_probs,
    ref_log_probs,
    output_log_probs,
    output_mask,
    scores,
    ratios_clip_eps,
    scores_std_eps,
):
    # advantage estimation
    means = (
        scores.view(-1, num_generations)
        .mean(dim=-1)
        .repeat_interleave(num_generations, axis=0)
    )
    stds = (
        scores.view(-1, num_generations)
        .std(dim=-1)
        .repeat_interleave(num_generations, axis=0)
    )
    advantages = (scores - means) / (stds + scores_std_eps)

    # policy loss
    ratios = torch.exp(log_probs - output_log_probs)
    clipped_ratios = torch.clamp(
        input=ratios,
        min=1 - ratios_clip_eps,
        max=1 + ratios_clip_eps,
    )
    per_token_policy_loss = torch.minimum(
        ratios * advantages.unsqueeze(-1),
        clipped_ratios * advantages.unsqueeze(-1),
    )
    policy_loss = (
        (per_token_policy_loss[:, :-1] * output_mask[:, 1:]).sum(dim=-1)
        / output_mask[:, 1:].sum(dim=-1)
    ).mean()

    # kl loss
    per_token_kl_loss = (
        torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1.0
    )
    kl_loss = (
        (per_token_kl_loss[:, :-1] * output_mask[:, 1:]).sum(dim=-1)
        / output_mask[:, 1:].sum(dim=-1)
    ).mean()

    # loss
    return policy_loss, kl_loss


@ray.remote
class GRPODispatcher:

    def __init__(self, grpo_learners):
        self._pool = ActorPool(grpo_learners)
        self._num_learners = len(grpo_learners)

    def process(
        self,
        num_generations,
        input_output_ids,
        input_output_mask,
        output_mask,
        output_log_probs,
        ref_log_probs,
        scores,
        ratios_clip_eps=0.2,
        scores_std_eps=1e-4,
        kl_loss_coef=0.05,
        temperature=1.0,
        batch_size=1,
    ):
        assert len(input_output_ids) == len(input_output_mask)
        assert len(input_output_ids) == len(output_mask)
        assert len(input_output_ids) == len(output_log_probs)
        assert len(input_output_ids) == len(ref_log_probs)
        assert len(input_output_ids) == len(scores)
        assert len(input_output_ids) == batch_size * self._num_learners
        assert batch_size % num_generations == 0

        fn = lambda a, kwargs: a.process.remote(**kwargs)
        args = []
        for bgn in range(0, len(input_output_ids), batch_size):
            kwargs = dict(
                num_generations=num_generations,
                input_output_ids=input_output_ids[bgn : bgn + batch_size],
                input_output_mask=input_output_mask[bgn : bgn + batch_size],
                output_mask=output_mask[bgn : bgn + batch_size],
                output_log_probs=output_log_probs[bgn : bgn + batch_size],
                ref_log_probs=ref_log_probs[bgn : bgn + batch_size],
                scores=scores[bgn : bgn + batch_size],
                ratios_clip_eps=ratios_clip_eps,
                scores_std_eps=scores_std_eps,
                kl_loss_coef=kl_loss_coef,
                temperature=temperature,
            )
            args.append(kwargs)
        losses = np.empty(self._num_learners)
        for idx, loss in enumerate(self._pool.map(fn, args)):
            losses[idx] = loss
        return losses

    def update(self, loss_):
        args = [loss_ for _ in range(self._num_learners)]
        fn = lambda a, v: a.update.remote(v)
        self._pool.map(fn, args)
