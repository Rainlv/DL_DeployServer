from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import sessionmaker, relationship

from config import config
from .models import DLModel, DLModelVersion, DLModelDeploy
from .utils import get_db_uri

DATABASE_URL = get_db_uri(db_name=config.DB_NAME, is_async=True)
Base: DeclarativeMeta = declarative_base()

engine = create_async_engine(DATABASE_URL)

async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class DLModelInDB(DLModel, Base):
    model_version_items = relationship('DLModelVersionInDB', back_populates='model_item')


class DLModelVersionInDB(DLModelVersion, Base):
    model_deploy_item = relationship('DLModelDeployInDB', back_populates='model_version_item')
    model_item = relationship('DLModelInDB', back_populates='model_version_items')


class DLModelDeployInDB(DLModelDeploy, Base):
    model_version_item = relationship('DLModelVersionInDB', back_populates='model_deploy_item')


async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session
