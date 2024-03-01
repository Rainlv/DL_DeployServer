from pydantic import BaseModel


class TrainingArgsModel(BaseModel):
    lr: float = 0.01
    batch_size: int = 1
    epoch_num: int = 100
