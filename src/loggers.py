import os

import wandb


class BaseLogger:
    def log(self, log_label: str, log_value: float, step: int) -> None:
        pass


class WandbLogger(BaseLogger):
    def __init__(self, project: str, name: str) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.local_rank == 0:
            wandb.init(project=project, name=name)

    def log(self, log_label: str, log_value: float, step: int) -> None:
        if self.local_rank == 0:
            wandb.log({log_label: log_value}, step=step)
