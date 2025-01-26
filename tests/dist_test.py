import ray
import time
import torch
import numpy as np
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rlvr.dist import TorchDistActor


def test_getaddr():
    inst = TorchDistActor()
    _ = inst.get_addr()


class SampleActor(TorchDistActor):

    def __init__(self):
        super().__init__(model=torch.nn.Linear(3, 4))

    def get_weight(self):
        return self._model.weight.data.numpy()


def test_global_group():
    pg = placement_group([{"CPU": 1} for _ in range(2)])
    _ = ray.get(pg.ready())
    st = PlacementGroupSchedulingStrategy(placement_group=pg)

    actors = []
    for rank in range(2):
        actor = ray.remote(SampleActor).options(scheduling_strategy=st).remote()
        if rank == 0:
            master_host, master_port = ray.get(actor.get_addr.remote())
        actor.init_process_group.remote(master_host, master_port, 2, rank)
        actors.append(actor)

    assert ray.get(actors[0].get_rank.remote()) == 0
    assert ray.get(actors[1].get_rank.remote()) == 1

    for rank in range(2):
        actors[rank].sync.remote(0, group_name=None)

    val0, val1 = ray.get(
        [
            actors[0].get_weight.remote(),
            actors[1].get_weight.remote(),
        ]
    )
    np.testing.assert_equal(val0, val1)


def test_local_group():
    pg = placement_group([{"CPU": 1} for _ in range(4)])
    _ = ray.get(pg.ready())
    st = PlacementGroupSchedulingStrategy(placement_group=pg)

    actors = []
    for rank in range(4):
        actor = ray.remote(SampleActor).options(scheduling_strategy=st).remote()
        if rank == 0:
            master_host, master_port = ray.get(actor.get_addr.remote())
        actor.init_process_group.remote(master_host, master_port, 4, rank)
        actors.append(actor)

    for rank in range(4):
        actors[rank].new_group.remote(ranks=[0, 1], group_name="local")

    for rank in range(2):
        actors[rank].sync.remote(0, group_name="local")

    val0, val1 = ray.get(
        [
            actors[0].get_weight.remote(),
            actors[1].get_weight.remote(),
        ]
    )
    np.testing.assert_equal(val0, val1)
