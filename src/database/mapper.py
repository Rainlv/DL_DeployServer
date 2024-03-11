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

    def get_by_id(self, version_id: int) -> DLModelVersionInDB | None:
        return self.session.query(self.db_schema).get(version_id)

    def create(self,
               model_id: int,
               sample_set_id: int,
               description: None | str,
               version: str
               ):
        version_obj = DLModelVersionInDB()
        version_obj.version = version
        version_obj.model_id = model_id
        version_obj.sample_set_id = sample_set_id
        version_obj.description = description
        self.session.add(version_obj)
        self.session.commit()
        return version_obj

    def deploy(self, model_version_obj: DLModelVersionInDB, display_name: str):
        model_version_obj.deploy_status = True
        obj = DLModelDeployInDB()
        obj.version_id = model_version_obj.id
        obj.display_name = display_name
        self.session.add(obj)
        self.session.commit()

    def edit(self,
             model_version_obj: DLModelVersionInDB,
             train_status: int | None = None,
             mar_file_path: str | None = None,
             description: str | None = None,
             ):
        if train_status is not None:
            model_version_obj.train_status = train_status
        if description:
            model_version_obj.description = description
        if mar_file_path:
            model_version_obj.model_mar_path = mar_file_path
        self.session.commit()
        return model_version_obj

    def delete(self, version_id: int):
        version_obj = self.get_by_id(version_id)
        if version_obj:
            self.session.delete(version_obj)
            self.session.commit()
            return True
        return False


class DLModelDeployMapper(BaseMapper):
    def __init__(self, session):
        super().__init__(session, DLModelDeployInDB)

    def query(self, version_id: int | None):
        conditions = []
        if version_id:
            conditions.append(DLModelDeployInDB.version_id == version_id)

        return self.session.query(self.db_schema).filter(*conditions).all()

    def get_by_version_id(self, version_id: int) -> DLModelDeployInDB | None:
        return self.session.query(self.db_schema).filter(DLModelDeployInDB.version_id == version_id).one_or_none()
