import { createRouter, createWebHistory } from 'vue-router';
import ChatPage from '@/components/ChatPage';

const routes = [
  {
    path: '/',
    name: 'chatPage',
    component: ChatPage,
  },
];

const router = createRouter({
  history: createWebHistory(),
  routes,
});

export default router;