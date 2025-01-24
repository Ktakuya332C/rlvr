import ray
import time
import torch
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from rlvr.dist import TorchDistActor


def test_getaddr():
    inst = TorchDistActor()
    _ = inst.get_addr()


class SampleActor(TorchDistActor):

    def broadcast(self, number, src_rank, group_name=None):
        tensor = torch.tensor(number)
        self._broadcast(tensor, src_rank, group_name)
        return tensor.item()


def test_global_group():
    pg = placement_group([{"CPU": 1} for _ in range(2)])
    _ = ray.get(pg.ready())
    st = PlacementGroupSchedulingStrategy(placement_group=pg)

    actors = []
    for rank in range(2):
        actor = ray.remote(SampleActor).options(scheduling_strategy=st).remote()
        if rank == 0:
            master_host = "localhost"
            _, master_port = ray.get(actor.get_addr.remote())
        actor.init_process_group.remote(master_host, master_port, 2, rank)
        actors.append(actor)

    # wait for the init_process_group to end without using ray's functionalities
    time.sleep(1.0)

    assert ray.get(actors[0].get_rank.remote()) == 0
    assert ray.get(actors[1].get_rank.remote()) == 1

    val0, val1 = ray.get(
        [
            actors[0].broadcast.remote(0, 0),
            actors[1].broadcast.remote(1, 0),
        ]
    )
    assert val0 == 0
    assert val1 == 0


def test_local_group():
    pg = placement_group([{"CPU": 1} for _ in range(4)])
    _ = ray.get(pg.ready())
    st = PlacementGroupSchedulingStrategy(placement_group=pg)

    actors = []
    for rank in range(4):
        actor = ray.remote(SampleActor).options(scheduling_strategy=st).remote()
        if rank == 0:
            master_host = "localhost"
            _, master_port = ray.get(actor.get_addr.remote())
        actor.init_process_group.remote(master_host, master_port, 4, rank)
        actors.append(actor)

    for rank in range(4):
        actors[rank].new_group.remote(ranks=[0, 1], group_name="local")

    val0, val1 = ray.get(
        [
            actors[0].broadcast.remote(0, 0, "local"),
            actors[1].broadcast.remote(1, 0, "local"),
        ]
    )
    assert val0 == 0
    assert val1 == 0
