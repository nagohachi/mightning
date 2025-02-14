import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SamplersMixin:
    def get_dataloader_with_distributed_sampler(
        self, dataloader: DataLoader, sampler: DistributedSampler
    ) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(42)

        return DataLoader(
            dataset=dataloader.dataset,
            batch_size=dataloader.batch_size // dist.get_world_size(),  # type: ignore
            sampler=sampler,
            collate_fn=dataloader.collate_fn,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            drop_last=dataloader.drop_last,
            worker_init_fn=self.seed_worker,  # type: ignore
            generator=g,
        )
