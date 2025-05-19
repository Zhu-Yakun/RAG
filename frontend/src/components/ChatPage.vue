<template>
    <div class="flex-col justify-start items-center relative page">
        <div class="background"></div>
        <span class="text text_2 pos">RAG&LLM测试</span>

        <div class="box content-wrapper">
            <el-container style="height: 100%; width: 100%;">
                <!-- 左侧历史记录和新建对话区域 -->
                <el-aside class="left-contain">
                    <el-button type="primary" icon="el-icon-plus" @click="startNewConversation"
                        style="width: 100%; margin-bottom: 20px;">
                        新建对话
                    </el-button>
                    <el-menu :default-active="activeConversationId" @select="selectConversation"
                        style="border-right: none;">
                        <el-menu-item v-for="conversation in conversations" :key="conversation.id"
                            :index="conversation.id"
                            style="display: flex; justify-content: space-between; align-items: center;">
                            <span>{{ conversation.preview }}</span>
                            <el-button type="text" @click="deleteConversation(conversation.id)">
                                <el-icon><Delete /></el-icon>
                            </el-button>
                        </el-menu-item>
                    </el-menu>
                </el-aside>
                <!-- 右侧聊天区域 -->
                <el-container>
                    <el-header class="chat-title">
                        对话
                    </el-header>
                    <el-main class="chat-main" ref="chatMessages" style="overflow-y: auto">
                        <div v-for="(message, index) in activeConversation.messages" :key="index" class="message-row"
                            :class="{ 'user-message': message.role === 'user', 'assistant-message': message.role === 'assistant' }">
                            <div class="chat-bubble" v-html="renderMarkdown(message.content)"></div>
                        </div>
                    </el-main>

                    <el-footer class="send-message">
                        <div class="input-container">
                            <!-- RAG切换按钮 -->
                            <el-button 
                                type="primary" 
                                :class="{ active: use_rag }"
                                @click="use_rag = !use_rag"
                                style="margin-right: 10px; display: flex; justify-content: center; align-items: center; padding: 0 15px;"
                            >
                                {{ use_rag ? '关闭RAG' : '启用RAG' }}
                            </el-button>

                            <!-- 消息输入框 -->
                            <el-input 
                                class="" 
                                v-model="newMessage" 
                                placeholder="输入消息..." 
                                @keyup.enter="sendMessage"
                                clearable
                            >
                                <template #append>
                                    <el-button type="primary" @click="sendMessage">发送</el-button>
                                </template>
                            </el-input>
                        </div>
                    </el-footer>
                </el-container>
            </el-container>
        </div>
    </div>
</template>

<script>
import axios from 'axios';
import { io } from 'socket.io-client';
import MarkdownIt from 'markdown-it';

const md = new MarkdownIt();
export default {
    data() {
        return {
            socket: null,  // WebSocket 连接
            conversations: [], // 存储所有对话
            activeConversationId: null, // 当前激活的对话ID
            newMessage: '', // 新消息内容
            use_rag: false,
            // userId: '', // 当前用户ID（自动从 JWT 获取）
        };
    },
    watch: {
        // 监听 activeConversation.messages 数组的变化
        'activeConversation.messages': {
            handler() {
                // 当消息数组更新后，等待 DOM 渲染完成再滚动到底部
                this.$nextTick(() => {
                    const chatMessagesEl = this.$refs.chatMessages.$el || this.$refs.chatMessages;
                    chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
                });
            },
            deep: true  // 如果 messages 是一个数组或对象，需要使用 deep 选项
        }
    },
    computed: {
        // 当前激活的对话
        activeConversation() {
            return this.conversations.find(
                (conv) => conv.id === this.activeConversationId
            ) || { messages: [] };
        },
    },
    methods: {
        // 转换 markdown 为 HTML
        renderMarkdown(content) {
            return md.render(content);
        },

        // 开始新对话
        async startNewConversation() {
            // console.log('开始新对话');
            try {
                const response = await axios.post(this.$baseUrl + '/api/chat/new_conversations');
                const newConv = response.data;

                this.conversations.unshift({
                    id: newConv.conversation_id,
                    preview: '新对话',
                    messages: [],
                });

                this.activeConversationId = newConv.conversation_id;
                this.$message.success({ message: '创建成功', duration: 1000 });
            } catch (error) {
                this.$message.warning({ message: '创建失败', duration: 1000 });
            }
        },

        // 选择历史对话
        async selectConversation(id) {
            // console.log('选择对话:', id);
            this.activeConversationId = id;
            const conversation = this.conversations.find(conv => conv.id === id);
            // console.log('conversation:', conversation);

            // 设置加载状态
            // this.$set(conversation, 'loading', true);
            conversation.loading = true; // 直接赋值即可

            if (!conversation.messages || conversation.messages.length === 0) {
                try {
                    const response = await axios.post(this.$baseUrl + '/api/chat/history', null, {
                        params: { conversation_id: id },
                    });

                    const historyMessages = response.data.history || [];

                    // 数据获取完成后，再更新响应式属性
                    // this.$set(conversation, 'messages', historyMessages);
                    conversation.messages = historyMessages;
                    // console.log('historyMessages:', historyMessages);
                    if (historyMessages.length > 0 && historyMessages[0].content)
                        conversation.preview = historyMessages[0].content.slice(0, 10);
                    else
                        conversation.preview = '历史对话';
                } catch (error) {
                    this.$message.warning({ message: "获取历史记录失败", duration: 1000 });
                } finally {
                    // 清除加载状态
                    // this.$set(conversation, 'loading', false);
                    conversation.loading = false;
                }
            } else {
                // 如果已经有数据，也确保加载状态为 false
                // this.$set(conversation, 'loading', false);
                conversation.loading = false;
            }

            // 等待 DOM 更新完成后，再执行滚动操作
            await this.$nextTick();
            const chatMessagesEl = this.$refs.chatMessages.$el || this.$refs.chatMessages;
            chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
        },

        // 发送消息 (WebSocket)
        sendMessage() {
            console.log('发送消息:', this.newMessage);
            if (!this.newMessage.trim()) {
                this.$message.warning({ message: "消息不能为空", duration: 1000 });
                return;
            }

            if (!this.activeConversationId) {
                this.$message.warning({ message: "请先选择或创建对话", duration: 1000 });
                return;
            }

            const userMessage = {
                role: 'user',
                content: this.newMessage,
                conversation_id: this.activeConversationId,
                user_id: this.userId,
            };

            // 先添加用户消息到前端
            const conversation = this.activeConversation;
            conversation.messages.push(userMessage);
            if (conversation.preview === '新对话')
                conversation.preview = this.newMessage.slice(0, 10);
            this.newMessage = '';

            // 滚动到底部
            this.$nextTick(() => {
                const chatMessagesEl = this.$refs.chatMessages.$el || this.$refs.chatMessages;
                chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
            });

            // 通过 WebSocket 发送消息
            // const token = localStorage.getItem('token');
            const if_use_rag = this.use_rag;
            this.socket.emit('chat', { userMessage,  if_use_rag});
        },

        // 监听 AI 回复
        setupSocketListeners() {
            this.socket.on('chat_response', (data) => {
                // console.log('收到 AI 回复:', data);

                if (data.status === 'success') {
                    const conversation = this.activeConversation;

                    // 确保 message 数组是响应式的，更新到页面
                    conversation.messages.push({
                        role: 'assistant',
                        content: data.response,
                    });

                    // 滚动到底部
                    this.$nextTick(() => {
                        const chatMessagesEl = this.$refs.chatMessages.$el || this.$refs.chatMessages;
                        chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
                    });
                } else {
                    this.$message.warning({ message: '回复失败', duration: 1000 });
                }
            });

            this.socket.on('connect', () => {
                console.log('WebSocket 连接成功');
            });

            this.socket.on('disconnect', () => {
                console.log('WebSocket 连接断开');
            });
        },

        // 删除对话
        async deleteConversation(id) {
            try {
                const { data } = await axios.delete(this.$baseUrl + '/api/chat/delete_conversation', {
                    params: { conversation_id: id }
                });
                if (data.status !== 'success') throw new Error();
                this.conversations = this.conversations.filter(conv => conv.id !== id);
                this.$message.success({ message: '对话已删除', duration: 1000 });
                if (this.activeConversationId === id) {
                    this.activeConversationId = this.conversations[0].id;
                    await this.selectConversation(this.activeConversationId);
                }
            } catch (error) {
                this.$message.error({ message: '删除失败', duration: 1000 });
            }
        },
    },
    async mounted() {
        try {
            // 连接 WebSocket
            this.socket = io(this.$baseUrl); // 确保 WebSocket 服务器地址正确
            this.setupSocketListeners();
            // 获取历史对话列表
            // console.log('初始化对话列表');
            const response = await axios.get(this.$baseUrl + '/api/chat/conversations');
            this.conversations = response.data.conversations || [];

            // console.log('对话列表长度:', this.conversations.length);

            if (this.conversations.length > 0) {
                this.activeConversationId = this.conversations[0].id;
                await this.selectConversation(this.activeConversationId);
            } else {
                await this.startNewConversation();
            }
        } catch (error) {
            this.$message.error({ message: '初始化失败', duration: 1000 });
        }
    },
    beforeUnmount() {
        if (this.socket) {
            this.socket.disconnect();
        }
    },
};
</script>

<style scoped>
/* 设置输入框光标颜色为黑色（清晰可见） */
>>>.el-input__inner {
    caret-color: #000 !important;
    color: #000 !important;
}

.text {
    text-shadow:
        0.31rem 0.31rem 0.31rem #1e1008a3,
        -0.3px -0.3px 0 #1e1008a3,
        0.3px -0.3px 0 #1e1008a3,
        -0.3px 0.3px 0 #1e1008a3,
        0.3px 0.3px 0 #1e1008a3;
    /* 黑色描边 */
}

.text_2 {
    color: #ffffff;
    font-size: 4.3rem;
    font-family: "hongleixingshu";
    line-height: 3.3rem;
}

.pos {
    position: absolute;
    left: 50%;
    top: 1.99rem;
    transform: translateX(-50%);
}

.page {
    width: 100%;
    height: 100%;
    overflow: auto;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: start;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* 背景 */
.background {
    background-image: url("../assets/bg1.png");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    opacity: 0.5;
}

/* 内容区域：使用flex布局并且允许换行 */
.box {
    position: absolute;
    top: 15%;
    display: flex;
    justify-content: center;
}

/* 透明内容框样式 */
.content-wrapper {
    width: 90vw;
    height: 80vh;
    margin: 0 auto;
    border-radius: 20px;
    transition: var(--transition);
    overflow: hidden;
    background-color: rgba(255, 255, 255, 0.5);
}

.container {
    display: flex;
    flex-direction: column;
    height: 100%;
}

.left-contain {
    width: 20%;
    background-color: rgba(255, 255, 255, 0.6);
    /* background-color: #f2f2f2; */
    padding: 20px;
}

.chat-title {
    background-color: rgba(255, 255, 255, 0.8);
    /* background-color: #ffffff; */
    padding: 20px;
    text-align: center;
}

.chat-main {
    padding: 20px;
    overflow-y: auto;
    background: rgba(255, 255, 255, 0.1);
    /* background: #f5f5f5; */
    flex: 1;
}

.el-input__inner {
    caret-color: #000;
    /* 黑色光标，更明显 */
}

.send-message {
    padding: 10px;
    background: rgba(255, 255, 255, 0.8);
    /* background-color: #f9f9f9; */
}

.input-container {
    display: flex;
    align-items: center;
    gap: 10px; /* 按钮和输入框之间的间距 */
}

/* RAG按钮的默认样式 */
.send-message .el-button {
    padding: 10px 15px;
    font-size: 14px;
    /* background-color: #409EFF;
    border-color: #409EFF; */
}

/* RAG按钮的激活状态样式 */
.send-message .el-button.active {
    background-color: #ffe240;
    color: white;
    border-color: #ffe240;
}

/* 每条消息所在行 */
.message-row {
    display: flex;
    margin-bottom: 10px;
}

/* 用户消息靠右，AI 消息靠左 */
.user-message {
    justify-content: flex-end;
}

.assistant-message {
    justify-content: flex-start;
}

/* 气泡样式 */
.chat-bubble {
    max-width: 70%;
    padding: 10px 15px;
    border-radius: 15px;
    word-break: break-word;
    line-height: 1.5;
    text-align: left;
}

/* 用户消息样式 */
.user-message .chat-bubble {
    background-color: #409eff;
    color: #fff;
    border-bottom-right-radius: 0;
}

/* AI 消息样式 */
.assistant-message .chat-bubble {
    background-color: #e4e4e4;
    color: #333;
    border-bottom-left-radius: 0;
}
</style>