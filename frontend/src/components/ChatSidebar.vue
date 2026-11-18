<!-- 聊天左侧栏：展示会话列表、当前登录邮箱和资料入口，并向 ChatView 发出导航/删除事件。
     它不直接请求 API；父页面持有会话数据并处理所有副作用。 -->
<script setup>
defineProps({
  conversations: { type: Array, required: true },
  conversationId: { type: String, default: null },
  email: { type: String, default: "当前用户" },
})

defineEmits(["new-conversation", "select-conversation", "delete-conversation", "open-documents", "logout"])

const formatTime = (value) => value ? new Date(value).toLocaleString() : ""
</script>

<template>
  <aside class="sidebar">
    <div><strong class="brand">MedEvidence</strong><p>医学证据智能问答</p></div>
    <button class="new-button" @click="$emit('new-conversation')">＋ 开启新对话</button>
    <section class="history">
      <span>历史对话</span>
      <div class="conversation-list">
        <div
          v-for="item in conversations"
          :key="item.id"
          class="conversation-item"
          :class="{ active: item.id === conversationId }"
        >
          <button class="conversation-select" @click="$emit('select-conversation', item.id)">
            <b class="conversation-title">{{ item.title }}</b>
          </button>
          <div class="conversation-meta">
            <small>{{ formatTime(item.updated_at) }}</small>
            <button class="delete-button" :aria-label="`删除会话：${item.title}`" @click.stop="$emit('delete-conversation', item.id)">删除</button>
          </div>
        </div>
      </div>
    </section>
    <div class="bottom"><button class="plain-button" @click="$emit('open-documents')">资料管理</button><small>{{ email }}</small><button class="plain-button" @click="$emit('logout')">退出登录</button></div>
  </aside>
</template>

<style scoped>
 .sidebar { min-width: 0; min-height: 0; display: flex; flex-direction: column; gap: 16px; padding: 18px 13px; overflow: hidden; background: #f8fafc; border-right: 1px solid #e2e8f0; }.brand { font-size: 18px; }.sidebar p, small { margin: 5px 0 0; color: #64748b; font-size: 12px; }.new-button { border: 0; border-radius: 8px; padding: 10px; background: #2563eb; color: #fff; font-weight: 600; }.history { min-height: 0; display: grid; gap: 8px; flex: 1; font-size: 12px; color: #475569; }.conversation-list { min-height: 0; display: grid; align-content: start; gap: 6px; overflow-y: auto; overflow-x: hidden; }.conversation-item { display: grid; gap: 4px; padding: 8px; border-radius: 7px; }.conversation-select, .plain-button { min-width: 0; border: 0; border-radius: 7px; padding: 0; text-align: left; background: transparent; color: #334155; }.conversation-select { width: 100%; }.conversation-item:hover, .conversation-item.active { background: #e0e7ff; color: #1d4ed8; }.conversation-title { display: -webkit-box; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-height: 1.4; }.conversation-meta { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 8px; align-items: center; }.conversation-meta small { min-width: 0; margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 13px; }.delete-button { position: static; min-width: 38px; border: 0; border-radius: 7px; padding: 5px 6px; background: #fee2e2; color: #b91c1c; font-size: 12px; font-weight: 700; }.delete-button:hover { background: #fecaca; color: #991b1b; }.bottom { flex: none; display: grid; gap: 8px; padding-top: 12px; border-top: 1px solid #e2e8f0; }.plain-button:hover { background: #e2e8f0; }
</style>
