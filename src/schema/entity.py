from datetime import datetime

from pydantic import BaseModel


class BaseEntity(BaseModel):
    id: int
    create_time: datetime
    user_id: int

    class Config:
        orm_mode = True


class DLModelEntity(BaseEntity):
    name: str
    description: None | str
    task_type_id: int


class DLModelVersionEntity(BaseEntity):
    model_id: int
    version: str
    sample_set_id: int
    deploy_status: bool
    model_mar_path: str
    description: None | str


class DLModelDeployEntity(BaseEntity):
    version_id: int
    display_name: str
    description: None | str
