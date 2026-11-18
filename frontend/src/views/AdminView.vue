<!-- 管理后台：按标签页管理共享基础资料、三段 Agent 提示词与访问令牌用户。
     进入某标签时再请求对应管理 API；提示词保存携带版本时间以接收后端的并发覆盖冲突提示。 -->
<script setup>
import { onMounted, ref } from "vue"
import { useRouter } from "vue-router"

import { clearToken, request } from "../api.js"

const router = useRouter()
// 后台按当前标签页惰性加载，避免进入页面时同时请求三组管理数据。
const activeTab = ref("documents")
const documents = ref([])
const users = ref([])
const systemPrompt = ref("")
const queryRewritePrompt = ref("")
const todoPrompt = ref("")
const promptsUpdatedAt = ref("")
const message = ref("")
const busy = ref(false)

async function loadDocuments() {
  documents.value = await (await request("/api/admin/base-documents")).json()
}

async function loadPrompts() {
  const result = await (await request("/api/admin/prompts")).json()
  systemPrompt.value = result.system_prompt
  queryRewritePrompt.value = result.query_rewrite_prompt
  todoPrompt.value = result.todo_prompt
  promptsUpdatedAt.value = result.updated_at
}

async function loadUsers() {
  users.value = await (await request("/api/admin/access-users")).json()
}

async function loadActiveTab() {
  message.value = ""
  try {
    if (activeTab.value === "documents") await loadDocuments()
    if (activeTab.value === "prompts") await loadPrompts()
    if (activeTab.value === "users") await loadUsers()
  } catch (error) {
    message.value = error.message
  }
}

async function uploadDocument(event) {
  const [file] = event.target.files
  if (!file) return
  busy.value = true
  try {
    const form = new FormData()
    form.append("file", file)
    const result = await (await request("/api/admin/base-documents", { method: "POST", body: form })).json()
    message.value = `基础资料已入库：${result.filename}，切块数=${result.chunks}`
    await loadDocuments()
  } catch (error) { message.value = error.message } finally { busy.value = false; event.target.value = "" }
}

async function deleteDocument(filename) {
  if (!window.confirm(`确认删除基础资料「${filename}」及其向量吗？`)) return
  try { await request(`/api/admin/base-documents/${encodeURIComponent(filename)}`, { method: "DELETE" }); await loadDocuments() } catch (error) { message.value = error.message }
}

async function savePrompts() {
  // 携带读取时的版本；后端发现 updated_at 过期会拒绝覆盖。
  busy.value = true
  try { const result = await (await request("/api/admin/prompts", { method: "PUT", body: JSON.stringify({ system_prompt: systemPrompt.value, query_rewrite_prompt: queryRewritePrompt.value, todo_prompt: todoPrompt.value, updated_at: promptsUpdatedAt.value }) })).json(); promptsUpdatedAt.value = result.updated_at; message.value = result.saved ? "提示词已保存，下一轮问答将使用新配置" : "提示词内容未变化" } catch (error) { message.value = error.message } finally { busy.value = false }
}

async function deleteUser(id) {
  if (!window.confirm("确认删除该访问令牌用户的全部会话、资料和记忆吗？")) return
  try { await request(`/api/admin/access-users/${id}`, { method: "DELETE" }); await loadUsers() } catch (error) { message.value = error.message }
}

async function clearUsers() {
  if (!window.confirm("确认清空全部访问令牌用户吗？邮箱用户不会受影响。")) return
  try { const result = await (await request("/api/admin/access-users", { method: "DELETE" })).json(); message.value = `已删除 ${result.deleted} 位访问令牌用户`; await loadUsers() } catch (error) { message.value = error.message }
}

function logout() { clearToken(); router.push("/login") }
onMounted(loadActiveTab)
</script>

<template>
  <main class="admin-page">
    <header><div><h1>MedEvidence 管理后台</h1><p>基础资料、Agent 提示词与访问令牌用户管理</p></div><button @click="logout">退出登录</button></header>
    <nav><button :class="{ active: activeTab === 'documents' }" @click="activeTab = 'documents'; loadActiveTab()">基础资料</button><button :class="{ active: activeTab === 'prompts' }" @click="activeTab = 'prompts'; loadActiveTab()">提示词</button><button :class="{ active: activeTab === 'users' }" @click="activeTab = 'users'; loadActiveTab()">访问令牌用户</button></nav>
    <p class="status">{{ message }}</p>
    <section v-if="activeTab === 'documents'"><label class="upload">添加基础资料<input type="file" :disabled="busy" @change="uploadDocument"></label><ul><li v-for="item in documents" :key="item.source"><span>{{ item.filename }}</span><button @click="deleteDocument(item.filename)">删除</button></li></ul></section>
    <section v-if="activeTab === 'prompts'" class="prompts"><label>主系统提示词（全局边界 / 面）<textarea v-model="systemPrompt" rows="16"></textarea></label><label>查询拆分与优化提示词（点）<textarea v-model="queryRewritePrompt" rows="10"></textarea></label><label>复杂任务流程与安排提示词（线 / 面）<textarea v-model="todoPrompt" rows="10"></textarea></label><button :disabled="busy" @click="savePrompts">保存配置</button></section>
    <section v-if="activeTab === 'users'"><button class="danger" @click="clearUsers">清空全部访问令牌用户</button><ul><li v-for="item in users" :key="item.id"><span>创建：{{ new Date(item.created_at).toLocaleString() }}｜最近登录：{{ new Date(item.last_login_at).toLocaleString() }}｜会话：{{ item.conversation_count }}</span><button @click="deleteUser(item.id)">删除</button></li></ul></section>
  </main>
</template>

<style scoped>
.admin-page { height: 100dvh; overflow-y: auto; overflow-x: hidden; max-width: 1100px; margin: 0 auto; padding: 28px; }.admin-page header { display: flex; justify-content: space-between; gap: 16px; align-items: start; }.admin-page h1, p { margin: 0; }.admin-page p { color: #64748b; }.admin-page nav { display: flex; gap: 8px; margin: 24px 0 12px; }.admin-page button { border: 0; border-radius: 8px; padding: 9px 12px; background: #e0e7ff; color: #1e3a8a; font-weight: 600; }.admin-page button.active { background: #2563eb; color: #fff; }.admin-page button.danger, li button { background: #fee2e2; color: #b91c1c; }.status { min-height: 22px; }.upload { display: inline-block; margin-bottom: 16px; padding: 9px 12px; border-radius: 8px; background: #2563eb; color: #fff; cursor: pointer; }.upload input { display: none; } ul { display: grid; gap: 8px; padding: 0; list-style: none; } li { display: flex; justify-content: space-between; gap: 12px; padding: 12px; border: 1px solid #e2e8f0; border-radius: 8px; } .prompts { display: grid; gap: 16px; }.prompts label { display: grid; gap: 7px; font-weight: 600; }.prompts textarea { width: 100%; padding: 10px; border: 1px solid #cbd5e1; border-radius: 8px; font-family: inherit; line-height: 1.5; }
</style>
