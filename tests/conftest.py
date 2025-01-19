import ray
import pytest


@pytest.fixture(scope="session", autouse=True)
def ray_init():
    ray.init()
    try:
        yield
    finally:
        ray.shutdown()
