from enum import Enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    func, Boolean, ForeignKey
)


class TableName(Enum):
    dl_model = 'dl_model'
    dl_model_version = 'dl_model_version'
    dl_model_deploy = 'dl_model_deploy'


class BaseDBModel:
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)  # 主键
    create_time = Column(DateTime(), server_default=func.now())  # 创建时间
    user_id = Column(Integer, nullable=False, index=True)  # 创建者


class DLModel(BaseDBModel):
    __tablename__ = TableName.dl_model.value
    name = Column(String(50), nullable=False, index=True)
    description = Column(String(255), nullable=False, index=True)


class DLModelVersion(BaseDBModel):
    __tablename__ = TableName.dl_model_version.value
    model_id = Column(Integer, ForeignKey(f'{TableName.dl_model.value}.id'), nullable=False, index=True, )
    version = Column(String(50), nullable=False, index=True)
    sample_set_id = Column(Integer, nullable=False, index=True)
    deploy_status = Column(Boolean, nullable=False, index=True)
    model_mar_path = Column(String(512), nullable=False)


class DLModelDeploy(BaseDBModel):
    __tablename__ = TableName.dl_model_deploy.value
    version_id = Column(Integer, ForeignKey(f'{TableName.dl_model_version.value}.id'), nullable=False, index=True)
    display_name = Column(String(50), nullable=False)
