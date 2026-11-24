<!-- 主聊天页：协调会话历史、资料管理、流式问答和右侧证据/执行记录面板。
     它调用 /api/conversations、/api/documents 与 /api/ask；后者的 NDJSON 事件被分别累计为
     回答、证据、轨迹和错误状态，回答文本再按渲染帧小块追加。 -->
<script setup>
import DOMPurify from "dompurify"
import { marked } from "marked"
import { computed, nextTick, onMounted, reactive, ref } from "vue"
import { useRouter } from "vue-router"

import ChatSidebar from "../components/ChatSidebar.vue"
import RightPanel from "../components/RightPanel.vue"
import EmailConfirmations from "../components/EmailConfirmations.vue"
import { clearToken, getToken, getTokenEmail, request, streamAsk } from "../api.js"
import {
  appendAnswerForDisplay,
  clonePanelItems,
  evidencePanelState,
} from "../chatStream.js"

const router = useRouter()
// 服务端正式数据、当前输入和右侧面板状态分别维护，避免相互耦合。
const conversationId = ref(null)
const conversations = ref([])
const turns = ref([])
const question = ref("")
const sending = ref(false)
const documents = ref([])
const uploadStatus = ref("")
const rightOpen = ref(false)
const activeTab = ref("evidence")
const currentEvidences = ref([])
const currentTrace = ref([])
const messages = ref(null)
const composerInput = ref(null)
const emailActions = ref([])
const emailBusy = ref(false)
const emailActionMessage = ref("")

const email = computed(getTokenEmail)

const renderMarkdown = (text) => DOMPurify.sanitize(marked.parse(text || ""))
const formatTime = (value) => value ? new Date(value).toLocaleString() : ""

async function loadConversations() {
  const response = await request("/api/conversations")
  conversations.value = await response.json()
}

async function loadDocuments() {
  const response = await request("/api/documents")
  documents.value = await response.json()
}

async function selectConversation(id, { preservePanel = false } = {}) {
  conversationId.value = id
  const response = await request(`/api/conversations/${id}/turns`)
  turns.value = await response.json()
  await loadEmailActions(id)
  if (!preservePanel) {
    currentEvidences.value = []
    currentTrace.value = []
  }
  await nextTick()
  if (messages.value) messages.value.scrollTop = messages.value.scrollHeight
}

function newConversation() {
  conversationId.value = null
  turns.value = []
  currentEvidences.value = []
  currentTrace.value = []
  emailActions.value = []
  emailActionMessage.value = ""
}

async function loadEmailActions(id = conversationId.value) {
  if (!id) return
  const response = await request(`/api/email-actions?conversation_id=${encodeURIComponent(id)}`)
  const actions = await response.json()
  if (conversationId.value === id) emailActions.value = actions
}

async function decideEmail(actionId, decision, delivered) {
  if (emailBusy.value) return
  const id = conversationId.value
  emailBusy.value = true
  emailActionMessage.value = ""
  try {
    const options = { method: "POST" }
    if (decision === "reconcile") options.body = JSON.stringify({ delivered })
    const response = await request(`/api/email-actions/${actionId}/${decision}`, options)
    const action = await response.json()
    if (conversationId.value !== id) return
    const messages = { sent: "邮件已被发送服务接受。", rejected: "已取消此邮件。",
      executing: "邮件正在发送，请稍后刷新会话查看。", failed: "发送服务未接受邮件。", unknown: "发送未能确认成功，请核实收件情况，系统不会自动重发。" }
    emailActionMessage.value = decision === "reconcile"
      ? "已记录核实结果，未发起邮件发送。"
      : messages[action.status] || "确认状态已更新。"
    await loadEmailActions(id)
  } catch (error) {
    if (conversationId.value === id) emailActionMessage.value = error.message
  }
  finally { emailBusy.value = false }
}

async function deleteConversation(id) {
  if (!window.confirm("确定删除这条历史对话及其全部上下文吗？此操作无法恢复。")) return
  try {
    await request(`/api/conversations/${id}`, { method: "DELETE" })
    conversations.value = conversations.value.filter((item) => item.id !== id)
    if (conversationId.value === id) newConversation()
  } catch (error) {
    window.alert(`删除失败：${error.message}`)
  }
}

async function ensureConversation() {
  // 空白页面不创建数据库记录；第一次真正发送问题时才创建会话。
  if (conversationId.value) return conversationId.value
  const response = await request("/api/conversations", { method: "POST" })
  conversationId.value = (await response.json()).id
  return conversationId.value
}

function openTurnPanel(turn, tab) {
  currentEvidences.value = clonePanelItems(turn.evidence_snapshot)
  currentTrace.value = clonePanelItems(turn.execution_trace)
  activeTab.value = tab
  rightOpen.value = true
}

function scrollWhenFollowing() {
  const node = messages.value
  if (!node || node.scrollHeight - node.scrollTop - node.clientHeight > 80) return
  nextTick(() => { node.scrollTop = node.scrollHeight })
}

function resizeQuestion(event) {
  const node = event.target
  const maximumHeight = window.innerHeight * 0.4
  node.style.height = "auto"
  node.style.height = `${Math.min(node.scrollHeight, maximumHeight)}px`
  node.style.overflowY = node.scrollHeight > maximumHeight ? "auto" : "hidden"
}

function resetQuestionHeight() {
  if (!composerInput.value) return
  composerInput.value.style.height = ""
  composerInput.value.style.overflowY = "hidden"
}

// ================================================================
async function sendQuestion() {
  // 1,防止空问题和重复发送:
  const text = question.value.trim()
  if (!text || sending.value) return

  sending.value = true
  currentEvidences.value = []
  currentTrace.value = []

  // 2,创建前端临时问答记录:
  // 草稿先进入页面，随后由 NDJSON 事件增量更新，流结束后再以数据库记录替换。
  const draft = reactive({
    id: `draft-${Date.now()}`,
    question: text,
    question_at: new Date().toISOString(),
    answer: "",
    answer_at: null,
    status: "processing",
    failure_code: null,
    failure_message: null,
    evidence_snapshot: [],
    execution_trace: [],
  })
  turns.value.push(draft)
  question.value = ""
  nextTick(resetQuestionHeight)
  scrollWhenFollowing()
  // ================================================================

  try {
    // 3,确保存在会话:
    const id = await ensureConversation()
    // 4,发起流式请求:
    await streamAsk({ question: text, conversation_id: id }, async (event) => {
      // evidence、trace、answer 和 error 是后端约定的四类增量事件。
      if (event.type === "evidence") {
        const evidence = event.evidences.map((item) => ({ ...item, query: event.query }))
        draft.evidence_snapshot.push(...evidence)
        currentEvidences.value.push(...evidence)
        const panelState = evidencePanelState(rightOpen.value, activeTab.value)
        activeTab.value = panelState.activeTab
        rightOpen.value = panelState.rightOpen
      }
      if (event.type === "trace") {
        draft.execution_trace.push(event.trace)
        currentTrace.value.push(event.trace)
      }
      if (event.type === "answer") {
        await appendAnswerForDisplay(event.content, (part) => {
          draft.answer += part
          scrollWhenFollowing()
        })
      }
      if (event.type === "error") {
        draft.status = "failed"
        draft.failure_code = event.code
        draft.failure_message = event.message
        scrollWhenFollowing()
      }
    })
    await selectConversation(id, { preservePanel: true })
    await loadConversations()
  } catch (error) {
    draft.answer = `请求失败：${error.message}`
    if (!getToken()) await router.push("/login")
  } finally {
    sending.value = false
    try { await loadEmailActions() } catch { /* The main response error is already shown. */ }
  }
}

async function uploadDocument(file) {
  uploadStatus.value = "正在解析并入库..."
  const form = new FormData()
  form.append("file", file)
  try {
    const response = await request("/api/documents/upload", { method: "POST", body: form })
    const result = await response.json()
    uploadStatus.value = `上传完成：${result.filename}，${result.chunks} 个文本块`
    await loadDocuments()
  } catch (error) {
    uploadStatus.value = error.message
    if (!getToken()) await router.push("/login")
  }
}

async function deleteDocument(id) {
  try {
    await request(`/api/documents/${id}`, { method: "DELETE" })
    await loadDocuments()
  } catch (error) {
    uploadStatus.value = error.message
  }
}

async function logout() {
  clearToken()
  await router.push("/login")
}

onMounted(async () => {
  try {
    await Promise.all([loadConversations(), loadDocuments()])
  } catch (error) {
    if (!getToken()) await router.push("/login")
  }
})
</script>

<template>
  <main class="chat-shell" :class="{ 'right-open': rightOpen }">
    <ChatSidebar :conversations="conversations" :conversation-id="conversationId" :email="email" @new-conversation="newConversation" @select-conversation="selectConversation" @delete-conversation="deleteConversation" @open-documents="rightOpen = true; activeTab = 'documents'" @logout="logout" />
    <section class="chat-main">
      <header><strong>{{ conversations.find((item) => item.id === conversationId)?.title || "新对话" }}</strong><button @click="rightOpen = !rightOpen">{{ rightOpen ? "收起侧栏" : "证据与资料" }}</button></header>
      <section ref="messages" class="messages" aria-live="polite"><p v-if="!turns.length" class="empty">输入医学问题开始对话；医学内容将优先检索本地证据。</p><article v-for="turn in turns" :key="turn.id" class="turn"><div class="user"><small>用户 {{ formatTime(turn.question_at) }}</small><p>{{ turn.question }}</p></div><div class="assistant"><small>MedEvidence {{ formatTime(turn.answer_at) }}</small><div class="answer" v-html="renderMarkdown(turn.answer)" /><p v-if="turn.status === 'failed' && turn.failure_message" class="failure-message">本轮未完成：{{ turn.failure_message }} 输入“重试”可重新执行刚才的问题。</p><div class="turn-actions"><button v-if="turn.evidence_snapshot?.length" @click="openTurnPanel(turn, 'evidence')">原文证据</button><button v-if="turn.execution_trace?.length" @click="openTurnPanel(turn, 'trace')">执行记录</button></div></div></article></section>
      <EmailConfirmations :actions="emailActions" :busy="emailBusy" :message="emailActionMessage" @confirm="decideEmail($event, 'confirm')" @reject="decideEmail($event, 'reject')" @reconcile="(id, delivered) => decideEmail(id, 'reconcile', delivered)" />
      <form class="composer" @submit.prevent="sendQuestion"><textarea ref="composerInput" v-model="question" rows="2" placeholder="请输入医学问题" :disabled="sending" @input="resizeQuestion" @keydown.enter.exact.prevent="sendQuestion" /><button type="submit" :disabled="sending || !question.trim()">{{ sending ? "回答中" : "发送" }}</button></form>
    </section>
    <RightPanel v-if="rightOpen" :active-tab="activeTab" :evidences="currentEvidences" :trace="currentTrace" :documents="documents" :upload-status="uploadStatus" @select-tab="activeTab = $event" @upload="uploadDocument" @delete-document="deleteDocument" />
  </main>
</template>

<style scoped>
.chat-shell { height: 100dvh; min-height: 0; display: grid; grid-template-columns: minmax(168px, 18%) minmax(0, 1fr); overflow: hidden; background: #fff; }.chat-shell.right-open { grid-template-columns: minmax(168px, 18%) minmax(0, 1fr) minmax(320px, 32%); }.chat-main { min-width: 0; min-height: 0; display: flex; flex-direction: column; overflow: hidden; }.chat-main header { height: 58px; flex: none; display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 0 20px; border-bottom: 1px solid #e2e8f0; }.chat-main header button, .turn-actions button { border: 1px solid #cbd5e1; border-radius: 7px; padding: 6px 8px; background: #fff; color: #334155; font-size: 12px; }.messages { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; padding: 24px max(7%, 20px); scroll-behavior: smooth; }.empty { color: #64748b; }.turn { display: grid; gap: 15px; margin-bottom: 24px; }.turn p { margin: 6px 0 0; white-space: pre-wrap; }.user { justify-self: end; max-width: 78%; padding: 10px 13px; border-radius: 12px 12px 2px 12px; background: #e0e7ff; }.assistant { max-width: 88%; padding: 12px 14px; border-radius: 12px 12px 12px 2px; background: #f8fafc; }.turn small { color: #64748b; font-size: 11px; }.answer { line-height: 1.75; word-break: break-word; }.failure-message { color: #b42318; font-size: 13px; }.answer :deep(pre) { overflow: auto; padding: 10px; background: #eef2f7; border-radius: 7px; }.answer :deep(code) { font-family: ui-monospace, monospace; }.turn-actions { display: flex; gap: 6px; margin-top: 10px; }.composer { flex: none; display: flex; gap: 10px; align-items: end; padding: 14px max(7%, 20px); border-top: 1px solid #e2e8f0; }.composer textarea { min-height: 52px; max-height: 40dvh; flex: 1; resize: none; overflow-y: hidden; padding: 11px; border: 1px solid #cbd5e1; border-radius: 10px; outline-color: #2563eb; }.composer button { border: 0; border-radius: 9px; padding: 11px 16px; background: #2563eb; color: #fff; font-weight: 700; }.composer button:disabled { opacity: .55; cursor: wait; } @media (max-width: 900px) { .chat-shell.right-open { grid-template-columns: 180px minmax(0, 1fr); }.chat-shell.right-open :deep(.right-panel) { position: absolute; z-index: 3; right: 0; height: 100dvh; width: min(90vw, 370px); box-shadow: -8px 0 24px #0f172a18; } }
</style>
