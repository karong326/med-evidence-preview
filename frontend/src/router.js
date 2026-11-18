// 定义登录、聊天与管理员页面的客户端路由，并在进入前执行最小 JWT 导航守卫。
// 守卫只改善前端体验；受保护 API 的用户/管理员权限始终由后端重新校验。
import { createRouter, createWebHistory } from "vue-router"

import { getAuthenticatedHomePath, getToken, getTokenRole } from "./api.js"
import ChatView from "./views/ChatView.vue"
import LoginView from "./views/LoginView.vue"
import AdminView from "./views/AdminView.vue"

// 路由关系：
//    /       → /login
//    /login  → 登录页
//    /chat   → 聊天页
//    /admin  → 管理后台
const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", redirect: "/login" },
    { path: "/login", component: LoginView },
    { path: "/chat", component: ChatView, meta: { requiresAuth: true } },
    { path: "/admin", component: AdminView, meta: { requiresAuth: true, requiresAdmin: true } },
  ],
})

// beforeEach() 是前端导航保护（只负责用户体验，不是安全边界。真正权限由后端 require_user() 和 require_admin() 验证）：
//    没有 JWT → 回登录页
//    不是管理员 → 不能进入 /admin
//    已经登录 → 不再显示登录页
router.beforeEach((to) => {
  if (to.meta.requiresAuth && !getToken()) return "/login"
  if (to.meta.requiresAdmin && getTokenRole() !== "admin") return "/chat"
  if (to.path === "/login" && getToken()) return getAuthenticatedHomePath()
  return true
})

export default router
