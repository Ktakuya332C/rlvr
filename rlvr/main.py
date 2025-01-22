import ray
from rlvr.loaders import get_gsm8k
from rlvr.actors import (
    Tokenizer,
    DeTokenizer,
    RolloutWorker,
    RolloutDispatcher,
)


def main():
    ray.init()

    tokenizer = Tokenizer.remote("sbintuitions/tiny-lm")
    rollout_dispatcher = RolloutDispatcher.remote(
        [RolloutWorker.remote("sbintuitions/tiny-lm") for _ in range(2)]
    )
    detokenizer = DeTokenizer.remote("sbintuitions/tiny-lm")

    dataloader = get_gsm8k()
    for batch in dataloader.iter_batches(batch_size=6):
        input_ids_ref, attention_mask_ref = tokenizer.process.remote(
            texts=batch["question"],
            apply_chat_template=False,
        )
        outputs_ref, output_mask_ref = rollout_dispatcher.process.remote(
            input_ids=input_ids_ref,
            attention_mask=attention_mask_ref,
            batch_size=2,
            max_length=512,
        )
        texts_ref = detokenizer.process.remote(
            tokens=outputs_ref,
            attention_mask=output_mask_ref,
        )
    print(ray.get(texts_ref))

    ray.shutdown()


if __name__ == "__main__":
    main()
