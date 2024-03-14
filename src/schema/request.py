from pydantic import BaseModel, Field


class TrainingArgsModel(BaseModel):
    lr: float = 0.01
    batch_size: int = 1
    epoch_num: int = 100


class CreateVersionRequestModel(BaseModel):
    model_id: int
    version: str
    sample_set_id: int
    description: None | str
    lr: float = Field(..., title="学习率", description="模型训练学习率", ge=0)
    batch_size: int = Field(..., title="批次大小", description="模型训练批次大小", ge=1)
    epoch_num: int = Field(..., title="训练轮次", description="模型训练轮次", ge=1)


class CreateDeployRequestModel(BaseModel):
    display_name: str
    description: None | str
