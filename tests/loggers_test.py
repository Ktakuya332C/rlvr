import ray
from rlvr import loggers


def test_metrics_collector():
    c = loggers.MetricsCollector()
    c.register("key1", 1)
    c.register("key1", 2)
    c.register("key2", 3)

    expected = {
        "key1/mean": 1.5,
        "key1/max": 2,
        "key1/min": 1,
        "key2/mean": 3,
        "key2/max": 3,
        "key2/min": 3,
    }
    assert c.get_metrics() == expected

    c2 = loggers.MetricsCollector()
    c2.register("key1", 3)
    c2.register("key3", 4)

    c.merge(c2)

    expected = {
        "key1/mean": 2,
        "key1/max": 3,
        "key1/min": 1,
        "key2/mean": 3,
        "key2/max": 3,
        "key2/min": 3,
        "key3/mean": 4,
        "key3/max": 4,
        "key3/min": 4,
    }
    assert c.get_metrics() == expected


def test_tensorboard_logger(tmp_path):
    t = loggers.TensorboardLogger.remote(tmp_path)
    ray.get(t.log_dict.remote({"key1": 1.0}, 0))
