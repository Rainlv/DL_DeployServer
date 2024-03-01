import requests
from loguru import logger

from config import config
from database.models import DLModelVersionInDB
from schema.request import TrainingArgsModel


def deploy_model_version(obj: DLModelVersionInDB):
    model_name = obj.model_item.register_name
    mar_path = obj.model_mar_path
    resp = requests.post(f"{config.TORCH_SERVER_URL}/models?url={mar_path}&model_name={model_name}")
    logger.info(
        f"Deploy model {model_name} version {obj.version} to torch server, status code: {resp.status_code}, response: {resp.text}")
    if resp.status_code == 200:
        return True
    return False


def train_model(obj: DLModelVersionInDB, training_arg: TrainingArgsModel):
    resp = requests.put(f"{config.TRAIN_SERVER_URL}/model/train/{obj.model_item.register_name}/{obj.id}",
                        json=training_arg.dict())
    if resp.status_code == 200:
        return True
    logger.error(
        f"Failed to start training model {obj.model_item.register_name} version {obj.version}, details: {resp.text}")
    return False
