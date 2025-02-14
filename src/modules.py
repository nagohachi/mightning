# import ABCMeta, abstractmethod
import inspect
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class OptimClass:
    optimizer: torch.optim.Optimizer  # type: ignore
    scheduler: torch.optim.lr_scheduler.LRScheduler | None


class MightningModule(metaclass=ABCMeta):
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        # int -> str -> list
        self.log_train_dict = defaultdict(lambda: defaultdict(list))
        # int -> str -> str -> list
        self.log_train_dict_sub = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.log_valid_dict = defaultdict(lambda: defaultdict(list))
        self.log_valid_dict_sub = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        self.log_test_dict = defaultdict(lambda: defaultdict(list))
        self.log_test_dict_sub = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        optims = self.configure_optimizers()
        self.optimizer = optims.optimizer
        self.scheduler = optims.scheduler

    @abstractmethod
    def configure_optimizers(self) -> OptimClass:
        pass

    @abstractmethod
    def training_step(self, batch: dict | list | tuple, batch_idx: int) -> torch.Tensor:
        pass

    @abstractmethod
    def validation_step(self, batch: dict | list | tuple, batch_idx: int) -> None:
        pass

    @abstractmethod
    def test_step(self, batch: dict | list | tuple, batch_idx: int) -> None:
        pass

    def on_validation_end(self) -> None:
        pass

    def on_test_end(self) -> None:
        pass

    def log(
        self,
        log_label: str,
        log_value: float,
        log_interval: int = -1,
        take_mean_on: str | None = None,
    ) -> None:
        """
        Log a value to the training, validation, or test set.

        Args:
            log_label (str): logged label name
            log_value (float): logged value
            log_interval (int): logging interval. if set to more than 0, the value will be logged every `log_interval` steps and take the mean of the values logged in that interval. \
                If set to -1, the mean of all values logged in the dataloader will be taken. \
                Defaults to -1.
            take_mean_on (str, optional): take the mean of the values logged in the interval by this key. Defaults to None.

        Raises:
            ValueError: log function not recognized
        """
        func_name = inspect.stack()[1].function
        if isinstance(log_value, torch.Tensor):
            log_value = log_value.item()
        match func_name:
            case "training_step":
                if take_mean_on is not None:
                    self.log_train_dict_sub[log_interval][log_label][
                        take_mean_on
                    ].append(log_value)
                else:
                    self.log_train_dict[log_interval][log_label].append(log_value)
            case "validation_step" | "on_validation_end":
                if take_mean_on is not None:
                    self.log_valid_dict_sub[log_interval][log_label][
                        take_mean_on
                    ].append(log_value)
                else:
                    self.log_valid_dict[log_interval][log_label].append(log_value)
            case "test_step" | "on_test_end":
                if take_mean_on is not None:
                    self.log_test_dict_sub[log_interval][log_label][
                        take_mean_on
                    ].append(log_value)
                else:
                    self.log_test_dict[log_interval][log_label].append(log_value)
            case _:
                raise ValueError(f"log function ({func_name}) not recognized")
