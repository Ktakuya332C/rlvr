import os
import ray
import pytest


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    os.environ["RAY_DEDUP_LOGS"] = str(0)
    os.environ["TORCH_CPP_LOG_LEVEL"] = "INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    ray.init()
    try:
        yield
    finally:
        ray.shutdown()
