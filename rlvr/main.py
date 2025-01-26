import ray
import numpy as np
from rlvr.loaders import get_gsm8k
from rlvr.actors import (
    Tokenizer,
    DeTokenizer,
    RolloutWorker,
    RolloutDispatcher,
    Replicator,
    LastIntScorer,
    ReferenceWorker,
    ReferenceDispatcher,
    GRPOLearner,
    GRPODispatcher,
)


def main():
    ray.init()

    tokenizer = Tokenizer.remote("sbintuitions/tiny-lm")
    rollout_workers = [RolloutWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    rollout_dispatcher = RolloutDispatcher.remote(rollout_workers)
    detokenizer = DeTokenizer.remote("sbintuitions/tiny-lm")
    replicator = Replicator.remote()
    scorer = LastIntScorer.remote()
    ref_workers = [ReferenceWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    ref_dispatcher = ReferenceDispatcher.remote(ref_workers)
    grpo_workers = [GRPOLearner.remote("sbintuitions/tiny-lm") for _ in range(2)]
    grpo_dispatcher = GRPODispatcher.remote(grpo_workers)

    global_dist_group = grpo_workers + rollout_workers
    host, port = ray.get(global_dist_group[0].get_addr.remote())
    for rank, worker in enumerate(global_dist_group):
        worker.init_process_group.remote(host, port, len(global_dist_group), rank)

    weight_share_group = [grpo_workers[0]] + rollout_workers
    weight_share_ranks = ray.get([w.get_rank.remote() for w in weight_share_group])
    for rank, worker in enumerate(weight_share_group):
        worker.new_group.remote(weight_share_ranks, group_name="weight-share")

    ddp_ranks = ray.get([w.get_rank.remote() for w in grpo_workers])
    for rank, worker in enumerate(grpo_workers):
        worker.new_group.remote(ddp_ranks, group_name="ddp")
        worker.distribute.remote(group_name="ddp")

    dataloader = get_gsm8k()
    for batch in dataloader.iter_batches(batch_size=24):
        loss_refs = []
        for bgn in range(0, 12, 6):
            for i in range(0, 12, 6):
                questions = batch["question"][bgn : bgn + 6]
                answers = batch["answer"][bgn : bgn + 6]
                input_ids_ref, attention_mask_ref = tokenizer.process.remote(
                    texts=questions,
                    apply_chat_template=False,
                )
                (
                    input_outputs_ref,
                    input_output_mask_ref,
                    output_mask_ref,
                    output_log_probs_ref,
                ) = rollout_dispatcher.process.remote(
                    input_ids=input_ids_ref,
                    attention_mask=attention_mask_ref,
                    batch_size=2,
                    max_length=512,
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=2,
                )
                output_texts_ref = detokenizer.process.remote(
                    tokens=input_outputs_ref,
                    attention_mask=output_mask_ref,
                )
                repl_answers_ref = replicator.process.remote(answers, 2)
                scores_ref = scorer.process.remote(
                    responses=output_texts_ref,
                    answers=repl_answers_ref,
                )
                ref_log_probs_ref = ref_dispatcher.process.remote(
                    input_output_ids=input_outputs_ref,
                    input_output_mask=input_output_mask_ref,
                    batch_size=2,
                )
                loss_ref = grpo_dispatcher.process.remote(
                    num_generations=2,
                    input_output_ids=input_outputs_ref,
                    input_output_mask=input_output_mask_ref,
                    output_log_probs=output_log_probs_ref,
                    output_mask=output_mask_ref,
                    ref_log_probs=ref_log_probs_ref,
                    scores=scores_ref,
                    batch_size=6,
                )
                loss_refs.append(loss_ref)
            grpo_dispatcher.update.remote(loss_ref)
        losses = ray.get(loss_refs)
        print("loss =", np.concatenate(losses).mean())

        src_rank = ray.get(weight_share_group[0].get_rank.remote())
        for worker in weight_share_group:
            worker.sync.remote(src_rank, "weight-share")

    print(ray.get(loss_ref))

    ray.shutdown()


if __name__ == "__main__":
    main()
