class BaseMapper:
    def __init__(self, session, db_schema):
        self.session = session
        self.db_schema = db_schema

    def list(self):
        return self.session.query(self.db_schema).all()

    def add(self, obj):
        self.session.add(obj)
        self.session.commit()
        return obj
