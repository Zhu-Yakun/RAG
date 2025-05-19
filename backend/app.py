from flask import Flask, jsonify,request,session
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, get_jwt_identity, jwt_required, verify_jwt_in_request
from werkzeug.datastructures import Headers
from models import *
from models.modelConfig import db
from config import Config
from flask_socketio import SocketIO
from extensions import jwt, socketio


def create_app():
    app = Flask(__name__)  
    app.config.from_object(Config)
    CORS(app,supports_credentials=True) # 解决跨域服务问题

    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")
    jwt.init_app(app)

    from views.chat import chat_api

    app.register_blueprint(chat_api)

    @app.before_request
    def load_token():
        headers = Headers(request.headers)
        headers.add('Access-Control-Allow-Origin', 'http://localhost:8080')
        headers.add('Access-Control-Allow-Credentials', 'true')
        token = request.cookies.get('access_token')
        if token:
            headers['Authorization'] = f'Bearer {token}'
            request.headers = headers

    return app

app = create_app()

if __name__ == '__main__':
    # socketio.run(app, debug=True)
    socketio.run(app, host='0.0.0.0', port=5000)  # 监听所有网络接口
    # app.run(host='0.0.0.0', port=5000)