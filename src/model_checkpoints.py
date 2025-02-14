from dataclasses import dataclass


@dataclass
class ModelCheckpoint:
    on: str
    greater_is_better: bool
