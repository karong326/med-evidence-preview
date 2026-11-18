// Vue 浏览器入口：创建根组件、安装客户端路由并挂载到 index.html 的 #app。
// API、身份校验和具体页面逻辑分别位于 api.js、router.js 与 views 目录。
import { createApp } from "vue"

import App from "./App.vue"
import router from "./router.js"

// 创建 Vue 应用、安装 Router、挂载到 HTML 的 #app：
createApp(App).use(router).mount("#app")
