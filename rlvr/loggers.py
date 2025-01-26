import ray
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


class MetricsCollector:

    def __init__(self):
        self._metrics = defaultdict(list)

    def merge(self, other):
        for key, value_list in other._metrics.items():
            self._metrics[key].extend(value_list)

    def register(self, name, value):
        self._metrics[name].append(value)

    def get_metrics(self):
        ret = {}
        for name, value_list in self._metrics.items():
            ret[f"{name}/mean"] = sum(value_list) / len(value_list)
            ret[f"{name}/max"] = max(value_list)
            ret[f"{name}/min"] = min(value_list)
        return ret


@ray.remote
class TensorboardLogger:

    def __init__(self, log_dir):
        self._writer = SummaryWriter(log_dir)

    async def log_dict(self, data, step):
        for key, value in data.items():
            self._writer.add_scalar(key, value, step)
