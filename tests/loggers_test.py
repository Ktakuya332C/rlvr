import ray
import contextlib
from rlvr import loggers


def test_stdout_logger():
    t = loggers.StdoutLogger.remote()
    t.store.remote(0, "key1", 1)
    t.store.remote(0, "key2", 2)
    ray.get(t.log.remote())


def test_tensorboard_logger(tmp_path):
    t = loggers.TensorboardLogger.remote(tmp_path)
    t.store.remote(0, "key1", 1)
    t.store.remote(0, "key2", 2)
    ray.get(t.log.remote())
