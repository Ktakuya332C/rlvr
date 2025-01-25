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
)


def main():
    ray.init()

    tokenizer = Tokenizer.remote("sbintuitions/tiny-lm")
    rollout_dispatcher = RolloutDispatcher.remote(
        [RolloutWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    )
    detokenizer = DeTokenizer.remote("sbintuitions/tiny-lm")
    replicator = Replicator.remote()
    scorer = LastIntScorer.remote()
    ref_log_prob_dispatcher = ReferenceDispatcher.remote(
        [ReferenceWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    )
    old_log_prob_dispatcher = ReferenceDispatcher.remote(
        [ReferenceWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    )

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
        ref_log_probs_ref = ref_log_prob_dispatcher.process.remote(
            input_output_ids=input_outputs_ref,
            input_output_mask=input_output_mask_ref,
            batch_size=2,
        )
        old_log_probs_ref = old_log_prob_dispatcher.process.remote(
            input_output_ids=input_outputs_ref,
            input_output_mask=input_output_mask_ref,
            batch_size=2,
        )

    print(ray.get(scores_ref), ray.get(ref_log_probs_ref))

    ray.shutdown()


if __name__ == "__main__":
    main()
