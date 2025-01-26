import ray
import contextlib
from rlvr import loggers


def test_stdout_logger():
    t = loggers.StdoutLogger.remote()
    t.store.remote("key1", 1)
    t.store_list.remote("key2", [2, 3])
    ray.get(t.log.remote(0))


def test_tensorboard_logger(tmp_path):
    t = loggers.TensorboardLogger.remote(tmp_path)
    t.store.remote("key1", 1)
    t.store_list.remote("key2", [2, 3])
    ray.get(t.log.remote(0))
