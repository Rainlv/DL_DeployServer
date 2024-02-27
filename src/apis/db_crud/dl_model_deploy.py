from apis.db_crud import BaseMapper
from database.db import DLModelDeployInDB


class DLModelDeployMapper(BaseMapper):
    def __init__(self, session):
        super().__init__(session, DLModelDeployInDB)
