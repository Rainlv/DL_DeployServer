"""
读取.env.{environment}配置文件信息
"""
from pathlib import Path

from pydantic import BaseSettings

rootDir = Path(__file__).resolve().parents[1]


class Env(BaseSettings):
    ENVIRONMENT: str = "prod"

    class Config:
        env_file = rootDir / ".env"
        env_file_encoding = 'utf-8'


class Config(BaseSettings):
    _env_file: str

    # 是否以调试模式运行
    DEBUG: bool = False
    # 运行端口
    PORT: int = 5000

    # 数据库配置相关
    DB_HOST: str = "localhost"
    DB_PORT: int = 3306
    DB_USER: str
    DB_PASSWD: str
    DB_NAME: str

    TORCH_SERVER_URL: str
    TRAIN_SERVER_URL: str
    MODEL_STORE_DIR: str

    SECRET: str

    class Config:
        extra = "allow"
        env_file = rootDir / '.env.prod'


__env = Env()
config = Config(_env_file=rootDir / f".env.{__env.ENVIRONMENT}")
