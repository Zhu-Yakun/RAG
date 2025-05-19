class Config(object):
    # 数据库配置
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:123456@localhost:3306/RAG'
    SQLALCHEMY_TRACK_MODIFICATIONS = True
    SQLALCHEMY_ECHO = True
    # 密钥
    SECRET_KEY = '123456'
    JWT_SECRET_KEY = '123456'
