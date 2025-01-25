import ray
import torch
import numpy as np
from torch.nn import functional as F

from rlvr import actors


def test_replicator():
    actor = actors.Replicator.remote()
    inpt = np.array([[1, 2], [3, 4]])
    output = ray.get(actor.process.remote(inpt, 2))
    np.testing.assert_equal(output, np.array([[1, 2], [1, 2], [3, 4], [3, 4]]))


def test_tokenize():
    actor = actors.Tokenizer.remote("sbintuitions/tiny-lm-chat")

    texts = np.array(["This is a pen", "That is also a pen"])
    ref = actor.process.remote(texts, apply_chat_template=False)
    input_ids, attention_mask = ray.get(ref)
    expected_input_ids = np.array(
        [[3, 489, 310, 287, 8926, 2], [3252, 310, 354, 287, 8926, 2]]
    )
    np.testing.assert_equal(input_ids, expected_input_ids)
    expected_attention_mask = np.array([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    np.testing.assert_equal(attention_mask, expected_attention_mask)

    texts = np.array(["Hi!", "Hello?"])
    ref = actor.process.remote(texts, apply_chat_template=True)
    input_ids, attention_mask = ray.get(ref)
    expected_input_ids = np.array(
        [[3, 51202, 7182, 749, 2], [51202, 271, 28498, 1690, 2]]
    )
    np.testing.assert_equal(input_ids, expected_input_ids)
    expected_attention_mask = np.array([[0, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    np.testing.assert_equal(attention_mask, expected_attention_mask)


def test_detokenizer():
    actor = actors.DeTokenizer.remote("sbintuitions/tiny-lm-chat")
    tokens = np.array([[3, 489, 310, 287, 8926, 2], [3252, 310, 354, 287, 8926, 2]])
    attention_mask = np.array([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]])
    texts = ray.get(actor.process.remote(tokens, attention_mask))
    assert texts[0] == "This is a pen"
    assert texts[1] == "That is also a pen"


def test_rollout_worker():
    actor = actors.RolloutWorker.remote("sbintuitions/tiny-lm")
    input_ids = np.array([[3, 489, 310, 287, 8926]])
    attention_mask = np.array([[0, 1, 1, 1, 1]])

    input_outputs, input_output_mask, output_mask = ray.get(
        actor.process.remote(input_ids, attention_mask, 6)
    )
    np.testing.assert_equal(input_outputs, np.array([[3, 489, 310, 287, 8926, 5477]]))
    np.testing.assert_equal(input_output_mask, np.array([[0, 1, 1, 1, 1, 1]]))
    np.testing.assert_equal(output_mask, np.array([[0, 0, 0, 0, 0, 1]]))

    input_outputs, input_output_mask, output_mask = ray.get(
        actor.process.remote(input_ids, attention_mask, 6, True, 1.0, 2)
    )
    assert input_outputs.shape == (2, 6)
    assert input_output_mask.shape == (2, 6)
    assert output_mask.shape == (2, 6)


def test_rollout_postprocess():
    input_outputs = np.array(
        [
            [2, 10, 3, 11, 12, 3, 13, 3, 14],  # <pad> p <eos> p r <eos> r <eos> r
            [10, 3, 11, 12, 13, 14, 3, 15, 16],  # p <eos> p p r r <eos> r r
        ]
    )
    input_mask = np.array([[0, 1, 1, 1], [1, 1, 1, 1]])
    expected_input_output_mask = np.array(
        [
            [0, 1, 1, 1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
        ]
    )
    expected_output_mask = np.array(
        [
            [0, 0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 0, 0],
        ]
    )
    eos_token_id = 3
    input_output_mask, output_mask = actors._rollout_postprocess(
        input_outputs, input_mask, eos_token_id
    )
    np.testing.assert_equal(input_output_mask, expected_input_output_mask)
    np.testing.assert_equal(output_mask, expected_output_mask)


def test_rollout_dispatcher():
    actor1 = actors.RolloutWorker.remote("sbintuitions/tiny-lm")
    actor2 = actors.RolloutWorker.remote("sbintuitions/tiny-lm")
    actor3 = actors.RolloutDispatcher.remote([actor1, actor2])
    input_ids = np.array([[489, 310], [3252, 310]])
    attention_mask = np.ones((2, 2), dtype=np.int64)
    ref = actor3.process.remote(input_ids, attention_mask, batch_size=1, max_length=3)
    input_outputs, input_output_mask, output_mask = ray.get(ref)
    assert input_outputs.shape == (2, 3)
    assert input_output_mask.shape == (2, 3)
    assert output_mask.shape == (2, 3)


def test_last_int_scorer():
    actor = actors.LastIntScorer.remote()
    responses = np.array(
        ["This is 42", "A: 2", "That is a pen", "The answer is \\boxed{3}"]
    )
    answers = np.array(["42", "1", "3", "3"])
    scores = ray.get(actor.process.remote(responses, answers))
    np.testing.assert_equal(scores, np.array([1.0, 0.0, 0.0, 1.0]))


def test_reference_worker():
    actor = actors.ReferenceWorker.remote("sbintuitions/tiny-lm")
    input_output_ids = np.array([[3, 489, 310, 287, 8926]])
    input_output_mask = np.array([[0, 1, 1, 1, 1]])
    ref_probs = ray.get(actor.process.remote(input_output_ids, input_output_mask))
    assert ref_probs.shape == (1, 5)


def test_reference_dispatcher():
    actor1 = actors.ReferenceWorker.remote("sbintuitions/tiny-lm")
    actor2 = actors.ReferenceWorker.remote("sbintuitions/tiny-lm")
    actor = actors.ReferenceDispatcher.remote([actor1, actor2])
    input_output_ids = np.array([[489, 310], [3252, 310]])
    input_output_mask = np.ones((2, 2), dtype=np.int64)
    ref_log_probs = ray.get(
        actor.process.remote(input_output_ids, input_output_mask, batch_size=1)
    )
    assert ref_log_probs.shape == (2, 2)


def test_grpo_loss():
    log_probs = F.log_softmax(torch.randn(3 * 2, 5), dim=-1)
    output_mask = torch.ones((3 * 2, 5))
    scores = torch.randn(3 * 2)
    loss = actors._grpo_loss(
        num_generations=2,
        log_probs=log_probs,
        old_log_probs=log_probs,
        ref_log_probs=log_probs,
        output_mask=output_mask,
        scores=scores,
        ratios_clip_eps=0.2,
        scores_std_eps=1e-4,
        kl_loss_coef=0.0,
    )
    torch.testing.assert_close(loss, torch.zeros_like(loss))
