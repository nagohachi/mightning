from argparse import ArgumentParser
from pathlib import Path

import torch
from examples.resnet.module import MightningResNet
from src.mixins.mixin_seed import FixSeedMixin
from src.model_checkpoints import ModelCheckpoint
from src.trainer import Trainer
from src.loggers import WandbLogger
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms


def retrieve_dataloaders(
    total_dev_batch_size: int,
    total_train_batch_size: int,
    dataset_root: str | Path = ".",
) -> tuple[DataLoader, DataLoader]:
    """get train & dev dataloaders

    Parameters
    ----------
    total_dev_batch_size : int
        dev batch_size (on all GPUs)
    total_train_batch_size : int
        train batch_size (on all GPUs)
    dataset_root : str | Path, optional
        dataset to download MNIST, by default "."

    Returns
    -------
    tuple[DataLoader, DataLoader]
        (train_dataloader, dev_dataloader)
    """
    dataset_train = torchvision.datasets.MNIST(
        root=dataset_root, train=True, download=True, transform=transforms.ToTensor()
    )
    dataset_dev = torchvision.datasets.MNIST(
        root=dataset_root, train=False, download=True, transform=transforms.ToTensor()
    )

    def collate_fn(
        features: list[tuple[torch.Tensor, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([image for (image, _) in features])
        labels = torch.tensor([label for (_, label) in features], dtype=torch.long)
        return images, labels

    train_dataloader = DataLoader(
        dataset=dataset_train,
        batch_size=total_train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_dataloader = DataLoader(dataset=dataset_dev, batch_size=total_dev_batch_size)

    return train_dataloader, dev_dataloader


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--dataset_download_path", type=str)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--total_train_batch_size", type=int)
    parser.add_argument("--total_dev_batch_size", type=int)

    args = parser.parse_args()

    resnet_module = MightningResNet(args.model_type, in_chans=1, num_classes=10)

    FixSeedMixin().fix_seed(seed=42)

    train_dataloader, dev_dataloder = retrieve_dataloaders(
        total_train_batch_size=args.total_train_batch_size,
        total_dev_batch_size=args.total_dev_batch_size,
        dataset_root=args.dataset_download_path,
    )

    trainer = Trainer(
        loggers=WandbLogger(project="test_mightning", name="test_resnet"),
        ckpt_dir=".",
        max_epochs=args.epochs,
        strategy="ddp_nccl",
        precision="16-mixed",
        fix_seed=True,
        val_check_interval=0.2,
        callbacks=[ModelCheckpoint(on="valid_acc", greater_is_better=True)],
    )

    trainer.fit(
        model=resnet_module,
        train_dataloader=train_dataloader,
        valid_dataloader=dev_dataloder,
    )


if __name__ == "__main__":
    main()
