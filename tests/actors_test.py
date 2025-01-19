import ray
import numpy as np

from rlvr.actors import TokenizeActor, RolloutActor, DeTokenizeActor, LastIntRewardActor


def test_tokenize_actor():
    actor = TokenizeActor.remote("openai-community/gpt2")
    item = {"text": "This is a pen"}
    item = ray.get(actor.map.remote(item))
    assert np.all(item["tokens"] == np.array([1212, 318, 257, 3112]))


def test_rollout_actor():
    actor = RolloutActor.remote("openai-community/gpt2", 50256)
    batch = [{"query": np.array([1212, 318])}, {"query": np.array([2504, 318])}]
    batch = ray.get(actor.map_batch.remote(batch, max_new_tokens=2))
    assert set(batch[0].keys()) == {"query", "query_response"}
    assert np.all(batch[0]["query_response"] == np.array([1212, 318, 257, 845]))


def test_detokenize_actor():
    actor = DeTokenizeActor.remote("openai-community/gpt2")
    item = {"tokens": np.array([1212, 318, 257, 3112])}
    item = ray.get(actor.map.remote(item))
    assert np.all(item["text"] == "This is a pen")


def test_last_int_reward_actor():
    actor = LastIntRewardActor.remote()
    item = {"response": "The answer is 32", "answer": "32"}
    item = ray.get(actor.map.remote(item))
    assert item["reward"] == 1
