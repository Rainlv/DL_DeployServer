from datetime import datetime

from pydantic import BaseModel, Field


class BaseEntity(BaseModel):
    id: int
    create_time: datetime
    user_id: int

    class Config:
        orm_mode = True


class DLTaskTypeEntity(BaseEntity):
    name: str = Field(..., title="任务类型名称", description="任务类型名称")


class DLModelEntity(BaseEntity):
    name: str = Field(..., title="模型名称", description="模型名称")
    description: None | str = Field(None, title="模型描述", description="模型描述")
    task_type_item: DLTaskTypeEntity = Field(..., alias="task_type_item", title="任务类型信息", )


class DLModelVersionEntity(BaseEntity):
    model_id: int = Field(..., title="模型ID", description="模型ID")
    version: str = Field(..., title="模型版本", description="模型版本，如v1.0.0")
    sample_set_id: int = Field(..., title="样本集ID", description="样本集ID")
    deploy_status: bool = Field(..., title="部署状态", description="部署状态 true已部署 false未部署")
    train_status: int = Field(..., title="训练状态",
                              description="训练状态 0训练完成 1未开始 -1训练失败 2准备中 3训练中")
    model_mar_path: str | None = Field(None, title="模型路径", description="模型路径")
    description: None | str = Field(None, title="模型版本描述", description="模型版本描述")
    lr: float = Field(..., title="学习率", description="模型训练学习率", ge=0)
    batch_size: int = Field(..., title="批次大小", description="模型训练批次大小", ge=1)
    epoch_num: int = Field(..., title="训练轮次", description="模型训练轮次", ge=1)


class DLModelDeployEntity(BaseEntity):
    version_id: int = Field(..., title="模型版本ID", description="模型版本ID")
    display_name: str
    description: None | str
