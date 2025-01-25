import ray
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
    old_workers = [ReferenceWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    old_dispatcher = ReferenceDispatcher.remote(old_workers)
    grpo_workers = [GRPOLearner.remote("sbintuitions/tiny-lm") for _ in range(2)]
    grpo_dispatcher = GRPODispatcher.remote(grpo_workers)

    global_dist_group = grpo_workers + old_workers + rollout_workers
    host, port = ray.get(global_dist_group[0].get_addr.remote())
    for rank, worker in enumerate(global_dist_group):
        worker.init_process_group.remote(host, port, len(global_dist_group), rank)

    dataloader = get_gsm8k()
    for batch in dataloader.iter_batches(batch_size=6):
        input_ids_ref, attention_mask_ref = tokenizer.process.remote(
            texts=batch["question"],
            apply_chat_template=False,
        )
        input_outputs_ref, input_output_mask_ref, output_mask_ref = (
            rollout_dispatcher.process.remote(
                input_ids=input_ids_ref,
                attention_mask=attention_mask_ref,
                batch_size=2,
                max_length=512,
                do_sample=True,
                temperature=1.0,
                num_return_sequences=2,
            )
        )
        output_texts_ref = detokenizer.process.remote(
            tokens=input_outputs_ref,
            attention_mask=output_mask_ref,
        )
        repl_answers_ref = replicator.process.remote(batch["answer"], 2)
        scores_ref = scorer.process.remote(
            responses=output_texts_ref,
            answers=repl_answers_ref,
        )
        ref_log_probs_ref = ref_dispatcher.process.remote(
            input_output_ids=input_outputs_ref,
            input_output_mask=input_output_mask_ref,
            batch_size=2,
        )
        old_log_probs_ref = old_dispatcher.process.remote(
            input_output_ids=input_outputs_ref,
            input_output_mask=input_output_mask_ref,
            batch_size=2,
        )
        loss_ref = grpo_dispatcher.process.remote(
            num_generations=2,
            input_output_ids=input_outputs_ref,
            input_output_mask=input_output_mask_ref,
            output_mask=output_mask_ref,
            ref_log_probs=ref_log_probs_ref,
            old_log_probs=old_log_probs_ref,
            scores=scores_ref,
            batch_size=6,
        )

    print(ray.get(loss_ref))

    ray.shutdown()


if __name__ == "__main__":
    main()
