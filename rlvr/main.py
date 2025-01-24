import ray
from rlvr.loaders import get_gsm8k
from rlvr.actors import (
    Tokenizer,
    DeTokenizer,
    RolloutWorker,
    RolloutDispatcher,
    LastIntScorer,
)


def main():
    ray.init()

    tokenizer = Tokenizer.remote("sbintuitions/tiny-lm")
    rollout_dispatcher = RolloutDispatcher.remote(
        [RolloutWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    )
    detokenizer = DeTokenizer.remote("sbintuitions/tiny-lm")
    scorer = LastIntScorer.remote()

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
            )
        )
        output_texts_ref = detokenizer.process.remote(
            tokens=input_outputs_ref,
            attention_mask=output_mask_ref,
        )
        scores_ref = scorer.process.remote(
            responses=output_texts_ref,
            answers=batch["answer"],
        )
    print(ray.get(scores_ref))

    ray.shutdown()


if __name__ == "__main__":
    main()
