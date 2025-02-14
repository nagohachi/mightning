from collections import defaultdict
from typing import Literal, Protocol, TypeAlias

# import icecream
import icecream
import numpy as np
import torch.distributed as dist

from mightning.loggers import BaseLogger
from mightning.modules import MightningModule

LabelType: TypeAlias = str

LoggedValuesType: TypeAlias = list[float]
StepType: TypeAlias = int

LogDictType: TypeAlias = dict[LabelType, float]
LogDictEachType: TypeAlias = dict[LabelType, LoggedValuesType]

RawLogDictType: TypeAlias = dict[StepType, LogDictEachType]
RawLogSubDictType: TypeAlias = dict[StepType, dict[LabelType, LogDictEachType]]


class LoggersMixinProtocol(Protocol):
    @property
    def loggers(self) -> list[BaseLogger]: ...

    @property
    def strategy(self) -> Literal["ddp_nccl"] | None: ...

    def __filter_dict_by_step(self, *args, **kwargs) -> dict[int, dict]: ...

    def __filter_log_dict_by_step(self, *args, **kwargs) -> RawLogDictType: ...

    def __filter_log_sub_dict_by_step(self, *args, **kwargs) -> RawLogSubDictType: ...

    def __merge_log_dict_each(self, *args, **kwargs) -> LogDictType: ...

    def __log_from_log_dict_each(self, *args, **kwargs) -> LogDictType: ...

    def __log_from_dict(self, *args, **kwargs) -> LogDictType: ...


class LoggersMixin:
    def __filter_dict_by_step(
        self: LoggersMixinProtocol, log_dict: dict[int, dict], step: int
    ) -> dict[int, dict]:
        if step == -1:
            return {-1: log_dict[-1]}
        return {
            log_interval: log_dict_each
            for log_interval, log_dict_each in log_dict.items()
            if log_interval > 0 and step % log_interval == 0
        }

    def __filter_log_dict_by_step(
        self: LoggersMixinProtocol,
        log_dict: RawLogDictType,
        step: int,
    ) -> RawLogDictType:
        return self.__filter_dict_by_step(log_dict, step)

    def __filter_log_sub_dict_by_step(
        self: LoggersMixinProtocol,
        log_dict_sub: RawLogSubDictType,
        step: int,
    ) -> RawLogSubDictType:
        return self.__filter_dict_by_step(log_dict_sub, step)

    def __merge_log_dict_each(
        self: LoggersMixinProtocol, log_dict_each: LogDictEachType
    ) -> LogDictType:
        dist.barrier()
        # list for each process
        gathered_dicts = [None for _ in range(dist.get_world_size())]
        # gather log_dict_each from all processes
        dist.all_gather_object(gathered_dicts, log_dict_each)
        # merge gathered_dicts
        merged_dict = defaultdict(list)
        for gathered_dict in gathered_dicts:
            for label, value in gathered_dict.items():  # type: ignore
                merged_dict[label].extend(value)

        # if self.local_rank == 0:
        #     icecream.ic(merged_dict)

        merged_dict_mean = {
            label: float(np.mean(value)) for label, value in merged_dict.items()
        }

        return merged_dict_mean

    def __log_from_log_dict_each(
        self: LoggersMixinProtocol,
        log_dict_each: LogDictEachType,
        step: int,
        global_step: int,
    ) -> LogDictType:
        result = {}
        mean_log_dict_each = self.__merge_log_dict_each(log_dict_each)
        for logger in self.loggers:
            for label, value in mean_log_dict_each.items():
                if step == -1:
                    logger.log(label, value, global_step)
                else:
                    logger.log(label, value, step)

        for label, value in mean_log_dict_each.items():
            result[label] = value

        return result

    def log_learning_rate(
        self: LoggersMixinProtocol, model: MightningModule, step: int
    ) -> None:
        for logger in self.loggers:
            for param_group in model.optimizer.param_groups:
                logger.log("learning_rate", param_group["lr"], step=step)

    def __log_from_dict(
        self: LoggersMixinProtocol,
        log_dict: RawLogDictType,
        log_dict_sub: RawLogSubDictType,
        step: int,
        global_step: int,
    ) -> LogDictType:
        result = {}
        ## process log dict
        log_dict_filtered = self.__filter_log_dict_by_step(log_dict, step)
        filtered_steps = log_dict_filtered.keys()

        for log_dict_each in log_dict_filtered.values():
            tmp_result = self.__log_from_log_dict_each(log_dict_each, step, global_step)
            result |= tmp_result

        # clear log_dict for step interval
        for log_step in filtered_steps:
            log_dict[log_step] = defaultdict(list)

        ## process log dict sub
        log_dict_sub_filtered = self.__filter_log_sub_dict_by_step(log_dict_sub, step)
        filtered_steps = log_dict_sub_filtered.keys()

        for log_sub_dict_each in log_dict_sub_filtered.values():
            for log_label, log_dict_each in log_sub_dict_each.items():
                result[log_label] = 0
                tmp_result_sub = self.__merge_log_dict_each(log_dict_each)

                # if self.local_rank == 0:
                #     icecream.ic(tmp_result_sub.values())

                for value in tmp_result_sub.values():
                    result[log_label] += value

                log_sub_dict_each[log_label] = defaultdict(list)
                result[log_label] /= len(tmp_result_sub)

                for logger in self.loggers:
                    if step == -1:
                        logger.log(log_label, result[log_label], global_step)
                    else:
                        logger.log(log_label, result[log_label], step)

        return result

    def log_train(
        self: LoggersMixinProtocol, model: MightningModule, step: int, global_step: int
    ) -> LogDictType:
        return self.__log_from_dict(
            model.log_train_dict, model.log_train_dict_sub, step, global_step
        )

    def log_valid(
        self: LoggersMixinProtocol, model: MightningModule, step: int, global_step: int
    ) -> LogDictType:
        return self.__log_from_dict(
            model.log_valid_dict, model.log_valid_dict_sub, step, global_step
        )

    def log_test(
        self: LoggersMixinProtocol, model: MightningModule, step: int, global_step: int
    ) -> LogDictType:
        return self.__log_from_dict(
            model.log_test_dict, model.log_test_dict_sub, step, global_step
        )
