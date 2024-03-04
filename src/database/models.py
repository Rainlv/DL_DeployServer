from enum import Enum

from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    func, Boolean, ForeignKey
)
from sqlalchemy.orm import DeclarativeMeta, declarative_base, relationship

Base: DeclarativeMeta = declarative_base()


class TableName(Enum):
    dl_model = 'dl_model'
    dl_model_version = 'dl_model_version'
    dl_model_deploy = 'dl_model_deploy'
    dl_task_type = 'dl_task_type'


class BaseDBModel:
    __allow_unmapped__ = True
    id = Column(Integer, primary_key=True, autoincrement=True, index=True)  # 主键
    create_time = Column(DateTime(), server_default=func.now())  # 创建时间
    user_id = Column(Integer, nullable=False, index=True, default=1)  # 创建者


class DLTaskTypeInDB(BaseDBModel, Base):
    __tablename__ = TableName.dl_task_type.value
    name = Column(String(50), nullable=False, index=True, unique=True)


class DLModelInDB(BaseDBModel, Base):
    __tablename__ = TableName.dl_model.value
    name = Column(String(50), nullable=False, index=True, unique=True)
    register_name = Column(String(50), nullable=False, index=True, unique=True)
    description = Column(String(255), nullable=True, index=True)
    task_type_id = Column(Integer, ForeignKey(f'{TableName.dl_task_type.value}.id'), nullable=False, index=True, )

    model_version_items = relationship('DLModelVersionInDB', back_populates='model_item')


class DLModelVersionInDB(BaseDBModel, Base):
    __tablename__ = TableName.dl_model_version.value
    model_id = Column(Integer, ForeignKey(f'{TableName.dl_model.value}.id'), nullable=False, index=True, )
    version = Column(String(50), nullable=False, index=True)
    sample_set_id = Column(Integer, nullable=False, index=True)
    deploy_status = Column(Boolean, nullable=False, index=True, default=False)
    train_status = Column(Integer, nullable=False, index=True, default=1)
    model_mar_path = Column(String(1024), nullable=True)
    description = Column(String(512), nullable=True)

    model_deploy_item: "DLModelDeployInDB" = relationship('DLModelDeployInDB', back_populates='model_version_item')
    model_item: DLModelInDB = relationship('DLModelInDB', back_populates='model_version_items')


class DLModelDeployInDB(BaseDBModel, Base):
    __tablename__ = TableName.dl_model_deploy.value
    version_id = Column(Integer, ForeignKey(f'{TableName.dl_model_version.value}.id'), nullable=False, index=True)
    display_name = Column(String(50), nullable=False)
    model_version_item: DLModelVersionInDB = relationship('DLModelVersionInDB', back_populates='model_deploy_item')
    description = Column(String(512), nullable=True)
