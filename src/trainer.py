import datetime
import math
import os
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers.feature_extraction_utils import BatchFeature

from src.loggers import BaseLogger
from src.mixins.mixin_loggers import LoggersMixin
from src.mixins.mixin_samplers import SamplersMixin
from src.mixins.mixin_seed import FixSeedMixin
from src.model_checkpoints import ModelCheckpoint
from src.modules import MightningModule


class Trainer(LoggersMixin, SamplersMixin, FixSeedMixin):
    def __init__(
        self,
        loggers: list[BaseLogger] | BaseLogger,
        ckpt_dir: str | Path,
        max_epochs: int,
        val_check_interval: float | int = 1,
        callbacks: list[ModelCheckpoint] | None = None,
        strategy: Literal["ddp_nccl"] | None = None,
        precision: Literal["16-mixed"] | None = None,
        fix_seed: bool = True,
    ) -> None:
        """custom trainer.

        Args:
            hyper_parameters (dict | None): Hyper parameters.
            loggers (list[BaseLogger] | BaseLogger): Instance of logger or list of loggers
            ckpt_dir (str | Path): Directory to save checkpoints and logs
            max_epochs (int): Number of maximum epochs
            val_check_interval (float | int, optional): How often valid metrics are calculated. If set to 0~1, metrics are calculated more than once in an epoch. Defaults to 1.
            strategy (Literal["ddp_nccl"] | None, optional): Training strategy. Defaults to None.
            precision (Literal["16-mixed"], optional): Automatic Mixed Precision strategy. Defaults to None.
            fix_seed (bool, optional): Whether to fix seed. Defaults to True.
        """
        self.loggers = loggers if isinstance(loggers, list) else [loggers]
        self.ckpt_dir = Path(ckpt_dir)
        self.max_epochs = max_epochs
        self.val_check_interval = val_check_interval
        self.callbacks = callbacks if callbacks is not None else []
        self.strategy: Literal["ddp_nccl"] | None = strategy
        self.precision = precision
        self.fix_seed = fix_seed

        if self.strategy == "ddp_nccl":
            dist.init_process_group("nccl")
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.global_rank = int(os.environ["RANK"])
            torch.cuda.set_device(self.local_rank)
            torch.cuda.empty_cache()

        if self.precision == "16-mixed":
            self.scaler = GradScaler()

        self.metrics_dict: dict[str, float] = {}  # metrics name -> value
        self.ckpt_path_dict: dict[str, Path] = {}  # metrics name -> ckpt path

    def callable_after_optimizer_step(self) -> None:
        pass

    def __send_to_device(self, batch: tuple | list | dict) -> tuple | list | dict:
        if self.strategy == "ddp_nccl":
            if isinstance(batch, tuple):
                return tuple(value.to(f"cuda:{self.local_rank}") for value in batch)
            elif isinstance(batch, dict):
                return {
                    key: value.to(f"cuda:{self.local_rank}")
                    for key, value in batch.items()
                }
            elif isinstance(batch, list):
                return [value.to(f"cuda:{self.local_rank}") for value in batch]
            elif isinstance(batch, BatchFeature):
                return BatchFeature(
                    {
                        key: value.to(f"cuda:{self.local_rank}")
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in batch.items()  # type: ignore
                    }
                )  # type: ignore
            else:
                raise NotImplementedError(
                    f"Please use tuple, list or dict for Dataset. current batch type: {type(batch)}"
                )
        raise NotImplementedError("Please use DDP strategy")

    def fit(
        self,
        model: MightningModule,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader | None,
    ) -> None:
        current_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        ckpt_dir_fit = self.ckpt_dir / current_datetime
        ckpt_dir_fit.mkdir(parents=True, exist_ok=True)

        if self.fix_seed:
            self.seed_everything(42)
            self.fix_seed(42)
            self.seed_everything(42)

        if self.strategy == "ddp_nccl":
            train_sampler = DistributedSampler(
                train_dataloader.dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=True,
                seed=42,
            )
            train_dataloader = self.get_dataloader_with_distributed_sampler(
                dataloader=train_dataloader, sampler=train_sampler
            )

            if valid_dataloader is not None:
                valid_sampler = DistributedSampler(
                    valid_dataloader.dataset,
                    num_replicas=dist.get_world_size(),
                    rank=dist.get_rank(),
                    shuffle=False,
                    seed=42,
                )
                valid_dataloader = self.get_dataloader_with_distributed_sampler(
                    dataloader=valid_dataloader, sampler=valid_sampler
                )

            model.model = model.model.to(f"cuda:{self.local_rank}")
            model.model = DistributedDataParallel(
                module=model.model, device_ids=[self.local_rank]
            )

        model.model.train()
        self.global_step = 0

        for epoch in range(self.max_epochs):
            if self.strategy != "ddp_nccl" or self.global_rank == 0:
                train_dataloader = tqdm(
                    train_dataloader,
                    desc=f"epoch: {epoch}",
                    dynamic_ncols=True,
                    leave=True,
                )  # type: ignore
                train_sampler.set_epoch(epoch)

            for train_idx, train_batch in enumerate(train_dataloader):
                self.global_step += 1
                model.optimizer.zero_grad()  # type: ignore

                # update scheduler
                if model.scheduler is not None:
                    model.scheduler.step()

                # move data to GPU if using DDP
                train_batch = self.__send_to_device(train_batch)

                # forward + update optimizer
                if self.precision == "16-mixed":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        loss = model.training_step(
                            batch=train_batch, batch_idx=train_idx
                        )
                    self.scaler.scale(loss).backward()
                    self.scaler.step(model.optimizer)
                    self.scaler.update()
                else:
                    loss = model.training_step(batch=train_batch, batch_idx=train_idx)
                    loss.backward()
                    model.optimizer.step()

                self.log_learning_rate(model=model, step=self.global_step)

                self.callable_after_optimizer_step()

                self.log_train(
                    model=model, step=self.global_step, global_step=self.global_step
                )

                eval_log_step = max(
                    1,
                    int(len(train_dataloader) * self.val_check_interval),
                )

                # validation
                if (
                    self.global_step % eval_log_step == 0
                    and valid_dataloader is not None
                ):
                    model.model.eval()
                    if self.strategy != "ddp_nccl" or self.global_rank == 0:
                        valid_dataloader = tqdm(
                            valid_dataloader,
                            desc="validation",
                            dynamic_ncols=True,
                            leave=True,
                        )  # type: ignore
                        valid_sampler.set_epoch(epoch)

                    assert valid_dataloader is not None
                    for valid_idx, valid_batch in enumerate(valid_dataloader):
                        with torch.inference_mode():
                            valid_batch = self.__send_to_device(valid_batch)
                            model.validation_step(
                                batch=valid_batch, batch_idx=valid_idx
                            )

                    model.on_validation_end()
                    valid_metrics_dict = self.log_valid(
                        model=model, step=-1, global_step=self.global_step
                    )

                    # save best model
                    for callback in self.callbacks:
                        greater_is_better = callback.greater_is_better
                        best_metric_value = self.metrics_dict.get(
                            callback.on, math.inf * (-1 if greater_is_better else 1)
                        )
                        current_metric_value = valid_metrics_dict[callback.on]

                        # skip if not improved
                        if (
                            greater_is_better
                            and current_metric_value < best_metric_value
                        ) or (
                            (not greater_is_better)
                            and current_metric_value > best_metric_value
                        ):
                            continue

                        # update best metric
                        self.metrics_dict[callback.on] = valid_metrics_dict[callback.on]
                        filename = f"step={self.global_step}_{callback.on}={valid_metrics_dict[callback.on]:.4f}.pth"

                        # update best ckpt path
                        self.ckpt_path_dict[callback.on] = ckpt_dir_fit / filename

                        # save best ckpt
                        if self.strategy == "ddp_nccl":
                            torch.save(
                                model.model.module.state_dict(),
                                ckpt_dir_fit / filename,
                            )
                        else:
                            torch.save(
                                model.model.state_dict(),
                                ckpt_dir_fit / filename,
                            )

                    if self.strategy != "ddp_nccl" or self.global_rank == 0:
                        # remove old ckpt
                        for ckpt_path in ckpt_dir_fit.glob("*.pth"):
                            if ckpt_path not in self.ckpt_path_dict.values():
                                if not ckpt_path.exists():
                                    continue
                                ckpt_path.unlink()

                    model.model.train()

            # torch.save(
            #     model.model.module.state_dict(),
            #     ckpt_dir_fit / f"epoch={epoch:03d}.pth",
            # )
            # on epoch end
            self.log_train(model=model, step=-1, global_step=self.global_step)

        torch.save(
            model.model.module.state_dict(),
            ckpt_dir_fit / "final.pth",
        )

    def test(
        self, model: MightningModule, test_dataloader: DataLoader
    ) -> dict[str, float]:
        self.global_step = 0
        if self.strategy == "ddp_nccl":
            if self.strategy == "ddp_nccl":
                if self.local_rank != 0:
                    return {}

            test_sampler = DistributedSampler(
                test_dataloader.dataset,
                shuffle=False,
                seed=42,
            )
            test_dataloader = self.get_dataloader_with_distributed_sampler(
                dataloader=test_dataloader, sampler=test_sampler
            )

            model.model = model.model.to(f"cuda:{self.local_rank}")
            model.model = DistributedDataParallel(
                module=model.model, device_ids=[self.local_rank]
            )

        model.model.eval()
        with torch.inference_mode():
            if self.strategy != "ddp_nccl" or self.global_rank == 0:
                test_dataloader = tqdm(
                    test_dataloader,
                    desc="testing",
                    dynamic_ncols=True,
                )  # type: ignore
            test_sampler.set_epoch(0)
            for test_idx, test_batch in enumerate(test_dataloader):
                test_batch = self.__send_to_device(test_batch)
                model.test_step(batch=test_batch, batch_idx=test_idx)

        model.on_test_end()
        return self.log_test(model=model, step=-1, global_step=self.global_step)
