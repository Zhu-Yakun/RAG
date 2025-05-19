from flask import Blueprint, request, jsonify
from zhipuai import ZhipuAI
from models.Message import Message, Conversation
from models.modelConfig import db
import uuid  # 生成唯一ID
from extensions import socketio
from flask_socketio import emit
from flask_jwt_extended import jwt_required, get_jwt_identity, decode_token

# 初始化 ZhipuAI 客户端
client = ZhipuAI(api_key="64661646507f4886af8bda4d2928e21b.RIQSS2ebNaX4yVAr")  # 请填写您自己的APIKey

chat_api = Blueprint("chat", __name__, url_prefix="/api/chat")

def get_history_by_ID(conversation_id=None, user_id=None):
    """
    根据 conversation_id 或 user_id 获取聊天记录
    """
    if conversation_id is None and user_id is not None:
        # 查询用户的所有对话
        conversations = Conversation.query.filter_by(user_id=user_id).all()
        conversation_history = [
            {
                "conversation_id": conv.id,
                "messages": [{"role": msg.role, "content": msg.content} for msg in Message.query.filter_by(conversation_id=conv.id).order_by(Message.timestamp).all()]
            }
            for conv in conversations
        ]
    elif conversation_id is not None:
        # 查询特定对话的所有消息
        messages = Message.query.filter_by(conversation_id=conversation_id).order_by(Message.timestamp).all()
        conversation_history = [{"role": msg.role, "content": msg.content} for msg in messages]
    else:
        conversation_history = []

    return conversation_history

@socketio.on('chat')
def handle_chat(data):
    """
    处理 WebSocket 方式的聊天请求
    """
    # print("handle_chat")
    # try:
    user_id = 1  # 直接为1，不需要登录，用户唯一
    use_rag = data['if_use_rag']
    user_message = data['userMessage']['content']
    conversation_id = data['userMessage']['conversation_id']

    # 获取历史对话记录
    conversation_history = get_history_by_ID(conversation_id)

    if(use_rag):
        print("use_rag")
        # TODO

    # 调用 ZhipuAI API
    response = client.chat.completions.create(
        model="glm-4-plus",
        messages=conversation_history + [{"role": "user", "content": user_message}],
        stream=True,  # WebSocket 使用流式传输
    )

    ai_reply = ''.join(chunk.choices[0].delta.content for chunk in response)

    # 记录消息
    user_message_record = Message(conversation_id=conversation_id, role='user', user_id=user_id, content=user_message)
    ai_message_record = Message(conversation_id=conversation_id, role='assistant', user_id=user_id, content=ai_reply)

    db.session.add(user_message_record)
    db.session.add(ai_message_record)
    db.session.commit()

    emit('chat_response', {'response': ai_reply, 'status': 'success'})

    # except Exception as e:
        # db.session.rollback()
        # emit('chat_response', {'error': str(e), 'status': 'fail'})

@chat_api.route('/history', methods=['POST'])
def get_history():
    """
    获取用户的历史对话
    """
    try:
        conversation_id = request.args.get('conversation_id', None)
        # print("conversation_id", conversation_id)
        user_id = 1
        conversation = get_history_by_ID(conversation_id, user_id)
        return jsonify({'history': conversation, 'status': 'success'})

    except Exception as e:
        return jsonify({'error': str(e), 'status': 'fail'}), 400

@chat_api.route('/conversations', methods=['GET'])
def get_conversations():
    """
    获取用户所有的会话
    """
    # print("get_conversations")
    # try:
    user_id = 1
    # print("user_id", user_id)
    conversations = Conversation.query.filter_by(user_id=user_id).order_by(Conversation.created_at.desc()).all()

    # print("conversations", conversations)

    conversation_list = [
        {
            'id': conv.id,
            'preview': Message.query.filter_by(conversation_id=conv.id).order_by(Message.timestamp).first().content[:10] if Message.query.filter_by(conversation_id=conv.id).count() > 0 else '历史对话'
        }
        for conv in conversations
    ]
    return jsonify({'conversations': conversation_list, 'status': 'success'})

    # except Exception as e:
        # return jsonify({'error': str(e), 'status': 'fail'}), 400


@chat_api.route('/new_conversations', methods=['POST'])
def create_conversation():
    """
    创建新会话
    """
    # print("create_conversation")
    try:
        user_id = 1
        # print("user_id",user_id)
        # conversation_id = str(uuid.uuid4())
        new_conversation = Conversation(user_id=user_id)
        db.session.add(new_conversation)
        db.session.commit()
    
        return jsonify({'conversation_id': new_conversation.id, 'status': 'success'})

    except Exception as e:
        print('create_conversation failed:',e)
        db.session.rollback()
        return jsonify({'error': str(e), 'status': 'fail'}), 400

@chat_api.route('/delete_conversation', methods=['DELETE'])
def delete_conversation():
    try:
        conversation_id = request.args.get('conversation_id')
        conversation = Conversation.query.filter_by(id=conversation_id).first()
        
        messages = Message.query.filter_by(conversation_id=conversation_id).all()
        for message in messages:
            db.session.delete(message)
        db.session.delete(conversation)
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': '删除对话失败'}), 500
