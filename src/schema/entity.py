from datetime import datetime

from pydantic import BaseModel, Field


class BaseEntity(BaseModel):
    id: int
    create_time: datetime
    user_id: int

    class Config:
        orm_mode = True


class DLTaskTypeEntity(BaseEntity):
    name: str


class DLModelEntity(BaseEntity):
    name: str
    description: None | str
    task_type_item: DLTaskTypeEntity = Field(..., alias="task_type_item", title="任务类型信息", )


class DLModelVersionEntity(BaseEntity):
    model_id: int
    version: str
    sample_set_id: int
    deploy_status: bool
    train_status: int
    model_mar_path: str | None
    description: None | str


class DLModelDeployEntity(BaseEntity):
    version_id: int
    display_name: str
    description: None | str
