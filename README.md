# simple-rlvr
A simple implementation of reinforcement learning with verifiable rewards.

## Usage
```bash
$ git clone https://github.com/Ktakuya332C/simple-rlvr.git
$ cd simple-rlvr
$ poetry install
$ potry run python -m rlvr.main
```

## Development
```bash
$ poetry run black .
$ poetry run pytest -xsvv tests
```

## Notes
- You may need to set `GLOO_SOCKET_IFNAM` to loopback (e.g. lo0)
