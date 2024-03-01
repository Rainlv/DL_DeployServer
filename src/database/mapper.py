from sqlalchemy.orm import Session

from database.models import DLModelInDB, DLModelDeployInDB, DLTaskTypeInDB, DLModelVersionInDB


class BaseMapper:
    def __init__(self, session, db_schema):
        self.session: Session = session
        self.db_schema = db_schema

    def list(self):
        return self.session.query(self.db_schema).all()

    def add(self, obj):
        self.session.add(obj)
        self.session.commit()
        return obj


class DLModelMapper(BaseMapper):
    def __init__(self, session):
        super().__init__(session, DLModelInDB)

    def query(self, name: str | None, task_type_name: str | None):
        conditions = []
        if name:
            # FIXME 模糊查询
            conditions.append(DLModelInDB.name == name)
        if task_type_name:
            task_type_obj = self.session.query(DLTaskTypeInDB).filter(
                DLTaskTypeInDB.name == task_type_name).one_or_none()
            if task_type_obj:
                conditions.append(DLModelInDB.task_type_id == task_type_obj.id)

        return self.session.query(self.db_schema).filter(*conditions).all()


class DLModelVersionMapper(BaseMapper):
    def __init__(self, session):
        super().__init__(session, DLModelVersionInDB)

    def query(self,
              model_id: int,
              version: str | None,
              deploy_status: bool | None):
        conditions = [DLModelVersionInDB.model_id == model_id]
        # FIXME 模糊查询
        if version:
            conditions.append(DLModelVersionInDB.version == version)
        if deploy_status:
            conditions.append(DLModelVersionInDB.deploy_status == deploy_status)

        return self.session.query(self.db_schema).filter(*conditions).all()


class DLModelDeployMapper(BaseMapper):
    def __init__(self, session):
        super().__init__(session, DLModelDeployInDB)
