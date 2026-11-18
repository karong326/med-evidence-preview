<!-- 聊天右侧可切换面板：显示本轮原文证据、个人资料上传/删除和 Agent 执行记录。
     父页面提供数据与上传删除回调；组件仅维护选中文件和面板切换后的滚动位置。 -->
<script setup>
import { nextTick, ref, watch } from "vue"

const props = defineProps({
  activeTab: { type: String, required: true },
  evidences: { type: Array, required: true },
  trace: { type: Array, required: true },
  documents: { type: Array, required: true },
  uploadStatus: { type: String, default: "" },
})

const emit = defineEmits(["select-tab", "upload", "delete-document"])
const selectedFile = ref(null)
const content = ref(null)
const formatTime = (value) => value ? new Date(value).toLocaleString() : ""
const filename = (source) => source?.split(/[\\/]/).pop() || "未知来源"

function chooseFile(event) { selectedFile.value = event.target.files[0] || null }
function upload() { if (selectedFile.value) emit("upload", selectedFile.value) }

function scrollToTop() { nextTick(() => { if (content.value) content.value.scrollTop = 0 }) }
watch(() => props.activeTab, scrollToTop)
watch(() => props.evidences.length, scrollToTop)
</script>

<template>
  <aside class="right-panel">
    <nav><button :class="{ active: activeTab === 'evidence' }" @click="emit('select-tab', 'evidence')">原文证据</button><button :class="{ active: activeTab === 'documents' }" @click="emit('select-tab', 'documents')">我的资料</button><button :class="{ active: activeTab === 'trace' }" @click="emit('select-tab', 'trace')">Agent 执行记录</button></nav>
    <section v-if="activeTab === 'evidence'" ref="content" class="content"><p v-if="!evidences.length" class="empty">本轮尚无可展示的原文证据。</p><article v-for="item in evidences" :key="item.citation_id" class="evidence"><strong>[{{ item.citation_id }}] 分数={{ Number(item.score).toFixed(4) }}</strong><small>{{ item.title }}｜{{ filename(item.source) }}<br>检索问题：{{ item.query || "本轮检索" }}</small><pre>{{ item.content }}</pre></article></section>
    <section v-else-if="activeTab === 'documents'" ref="content" class="content"><label class="upload"><input type="file" @change="chooseFile"><span>{{ selectedFile?.name || "选择资料文件" }}</span></label><button class="primary" :disabled="!selectedFile" @click="upload">上传并入库</button><p class="status">{{ uploadStatus }}</p><p v-if="!documents.length" class="empty">还没有上传资料。</p><article v-for="item in documents" :key="item.id" class="document"><div><b>{{ item.filename }}</b><small>{{ item.status }}｜{{ item.chunk_count }} 个文本块｜{{ formatTime(item.created_at) }}</small></div><button @click="emit('delete-document', item.id)">删除</button></article></section>
    <section v-else ref="content" class="content"><p v-if="!trace.length" class="empty">本轮尚无执行记录。</p><ol v-else class="trace"><li v-for="item in trace" :key="item.step"><b>{{ item.title }}</b><p>{{ item.detail }}</p><small>{{ formatTime(item.created_at) }}</small></li></ol></section>
  </aside>
</template>

<style scoped>
.right-panel { min-width: 320px; min-height: 0; display: flex; flex-direction: column; overflow: hidden; border-left: 1px solid #e2e8f0; background: #fff; }.right-panel nav { flex: none; display: flex; gap: 4px; padding: 10px; border-bottom: 1px solid #e2e8f0; }.right-panel nav button { border: 0; border-radius: 7px; padding: 7px 8px; background: transparent; font-size: 12px; }.right-panel nav button.active { background: #e0e7ff; color: #1d4ed8; font-weight: 700; }.content { flex: 1; min-height: 0; overflow-y: auto; overflow-x: hidden; padding: 14px; font-size: 13px; }.empty, .status { color: #64748b; }.evidence, .document { padding: 0 0 14px; margin-bottom: 14px; border-bottom: 1px solid #e2e8f0; }.evidence strong, .document b, .document small { display: block; }.evidence small, .document small, .trace small { color: #64748b; font-size: 11px; }.evidence pre { margin: 9px 0 0; white-space: pre-wrap; font: 12px/1.65 ui-monospace, monospace; }.upload { display: block; border: 1px dashed #94a3b8; border-radius: 8px; padding: 11px; color: #475569; }.upload input { display: block; width: 100%; margin-bottom: 7px; }.primary { width: 100%; margin-top: 8px; border: 0; border-radius: 8px; padding: 9px; background: #2563eb; color: #fff; }.document { display: flex; justify-content: space-between; gap: 8px; }.document button { height: fit-content; border: 0; border-radius: 6px; padding: 6px; color: #b91c1c; background: #fee2e2; }.trace { display: grid; gap: 0; padding: 0; margin: 0; list-style: none; }.trace li { position: relative; padding: 0 0 18px 25px; border-left: 1px solid #cbd5e1; }.trace li::before { content: ""; position: absolute; left: -5px; top: 3px; width: 9px; height: 9px; border-radius: 50%; background: #2563eb; }.trace p { margin: 4px 0; color: #475569; line-height: 1.55; }
</style>
