from pydantic import BaseModel


class TrainingArgsModel(BaseModel):
    lr: float = 0.01
    batch_size: int = 1
    epoch_num: int = 100


class CreateVersionRequestModel(BaseModel):
    model_id: int
    version: str
    sample_set_id: int
    description: None | str


class CreateDeployRequestModel(BaseModel):
    display_name: str
