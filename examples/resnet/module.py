from typing import Literal
import timm
import torch
from torch._tensor import Tensor
from src.modules import MightningModule, OptimClass
from torch.optim import AdamW
from torch import nn


class MightningResNet(MightningModule):
    def __init__(
        self,
        model_type: Literal["resnet18", "resnet50"],
        in_chans: int,
        num_classes: int,
    ) -> None:
        """ResNet のラッパーモデル

        Parameters
        ----------
        model_type : Literal["resnet18", "resnet50"]
            画像認識モデルの形式
        in_chans : int
            入力チャンネル数
        num_classes : int
            出力クラス数
        """
        self.model = timm.create_model(  # type: ignore
            model_type, pretrained=True, in_chans=in_chans, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss()
        super().__init__(self.model)

    def configure_optimizers(self) -> OptimClass:
        return OptimClass(optimizer=AdamW(self.model.parameters()), scheduler=None)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        images, labels = batch
        outputs = self.model(images)

        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, log_interval=-1)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, labels = batch
        outputs = self.model(images)
        preds = torch.argmax(outputs, dim=-1)

        loss = self.criterion(outputs, labels)
        self.log("valid_loss", loss, log_interval=-1)

        accuracy = (preds == labels).to(torch.float32).mean()
        self.log("valid_acc", accuracy, log_interval=-1)

    def test_step(self, batch: dict, batch_idx: int) -> None:
        pass
