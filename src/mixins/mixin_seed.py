import os
import random

import numpy as np
import torch


class FixSeedMixin:
    def _fix_seed(self, seed: int) -> None:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        # torch.use_deterministic_algorithms(True)

    def seed_worker(self, worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
