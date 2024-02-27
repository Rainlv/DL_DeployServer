from datetime import datetime

from pydantic import BaseModel


class BaseEntity(BaseModel):
    id: int
    create_time: datetime
    user_id: int


class DLModelEntity(BaseEntity):
    model_id: int
    version: str
    sample_set_id: int
    deploy_status: bool
    model_mar_path: str
