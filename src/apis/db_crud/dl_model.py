from apis.db_crud import BaseMapper
from database.db import DLModelInDB


class DLModelMapper(BaseMapper):
    def __init__(self, session):
        super().__init__(session, DLModelInDB)
