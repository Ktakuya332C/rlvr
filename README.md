# simple-rlvr
A simple toy implementation of GRPO with LLMs.

## Usage
```bash
$ git clone https://github.com/Ktakuya332C/simple-rlvr.git
$ cd simple-rlvr
$ poetry install
$ poetry run python -m rlvr.main \
  --model=sbintuitions/tiny-lm-chat \
  --num-rollout-workers=2 \
  --num-reference-workers=2 \
  --num-grpo-learners=2 \
  --batch-size-sync=16 \
  --batch-size-update=8 \
  --batch-size-backward=4 \
  --batch-size-rollout=2 \
  --batch-size-reference=2 \
  --num-generations=2 \
  --max-length=512 \
  --temperature=1.0
```

## Development
```bash
$ poetry run black .
$ poetry run pytest -xsvv tests
```

## Note
- You may need to set `GLOO_SOCKET_IFNAME=lo0` to run this script on Mac.
- This design is largely influenced by [RLlib Flow](https://arxiv.org/abs/2011.12719).

## ToDo
- missing eos penalty
- bf16 (numpy does not support bf16) 
- FSDP (DDP works without GPUs, but FSDP does not)
- vllm integration (`collective_rpc` support is not yet released)
- large scale experiments
- tensor parallel
