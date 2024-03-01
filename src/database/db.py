from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from config import config
from .models import Base
from .utils import get_db_uri

DATABASE_URL = get_db_uri(db_name=config.DB_NAME, is_async=False)

# engine = create_async_engine(DATABASE_URL)
engine = create_engine(DATABASE_URL)

# async_session_maker = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
session_maker = sessionmaker(engine, expire_on_commit=False)


def create_db_and_tables():
    Base.metadata.create_all(engine)


# async def create_db_and_tables():
#     async with engine.begin() as conn:
#         await conn.run_sync(Base.metadata.create_all)


# async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
#     async with session_maker() as session:
#         yield session
def get_session() -> Generator[Session, None, None]:
    with session_maker() as session:
        yield session
