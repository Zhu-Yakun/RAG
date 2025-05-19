from flask_sqlalchemy import SQLAlchemy
# 实例SQLAlchemy类
db = SQLAlchemy()

# 创建
def create():
    db.create_all()

# 删除
def drop():
    db.drop_all()

