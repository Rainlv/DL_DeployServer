import requests
from loguru import logger

from config import config
from database.models import DLModelVersionInDB


def deploy_model_version(obj: DLModelVersionInDB):
    model_name = obj.model_item.name
    mar_path = obj.model_mar_path
    resp = requests.post(f"{config.TORCH_SERVER_URL}/models?url={mar_path}&model_name={model_name}")
    logger.info(
        f"Deploy model {model_name} version {obj.version} to torch server, status code: {resp.status_code}, response: {resp.text}")
    if resp.status_code == 200:
        return True
    return False
