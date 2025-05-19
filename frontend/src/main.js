import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import ElementPlus from 'element-plus'
import 'element-plus/dist/index.css'
import axios from 'axios'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'


// 配置 axios，解决跨域请求携带凭证
const app = createApp(App)
app.config.globalProperties.$axios = axios
axios.defaults.withCredentials = true

// 注册所有图标组件
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
  app.component(key, component)
}

// 设置本地运行的基地址
app.config.globalProperties.$baseUrl = 'http://localhost:5000'

app.use(router)
app.use(ElementPlus)
app.mount('#app')