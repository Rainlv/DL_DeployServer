from config import config


def get_db_uri(db_name: str, is_async: bool = False) -> str:
    """
    配置文件中拼接数据库URI
    :param db_name: 数据库名称
    :param is_async: 是否异步，异步则使用asyncpg
    :return:
    """
    host = config.DB_HOST
    port = config.DB_PORT
    user = config.DB_USER
    passwd = config.DB_PASSWD
    return f"mysql+pymysql://{user}:{passwd}@{host}:{port}/{db_name}" if not is_async else f"mysql+aiomysql://{user}:{passwd}@{host}:{port}/{db_name}"


def orm_to_dict(obj):
    dic = {}
    dic_columns = obj.__table__.columns
    # 保证都是字符串和数字
    types = [str, int, float, bool]
    # 注意，obj.__dict__会在commit后被作为过期对象清空dict，所以保险的办法还是用columns
    for k, tmp in dic_columns.items():
        # k=nick,tmp=user.nick
        v = getattr(obj, k, None)
        dic[k] = str(v) if v and type(v) not in types else v
    return dic
