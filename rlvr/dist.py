import os
import torch
import socket


class TorchDistActor:

    def __init__(self):
        self._groups = {}

    def get_addr(self):
        s = socket.socket()
        s.bind(("", 0))
        host, port = s.getsockname()
        return host, port

    def init_process_group(self, master_host, master_port, world_size, rank):
        os.environ["MASTER_ADDR"] = master_host
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group("gloo", world_size=world_size, rank=rank)
        self._groups[None] = None

    def get_rank(self, group_name=None):
        group = self._groups[group_name]
        return torch.distributed.get_rank(group)

    def new_group(self, ranks, group_name):
        group = torch.distributed.new_group(ranks)
        self._groups[group_name] = group

    def _broadcast(self, tensor, src_rank, group_name=None):
        group = self._groups[group_name]
        rank = torch.distributed.get_rank(group)
        torch.distributed.broadcast(tensor, src_rank, group)
