<script setup>
defineProps({ actions: { type: Array, default: () => [] }, busy: Boolean, message: String })
defineEmits(["confirm", "reject"])
</script>

<template>
  <aside v-if="actions.length || message" class="email-confirmations" aria-label="邮件确认">
    <p v-if="message" role="status">{{ message }}</p>
    <article v-for="action in actions" :key="action.id">
      <strong>待确认邮件 · 尚未发送</strong>
      <p>收件人：{{ action.recipient }}</p>
      <p>主题：{{ action.subject }}</p>
      <pre>{{ action.body }}</pre>
      <button :disabled="busy || action.status !== 'pending'" @click="$emit('confirm', action.id)">
        {{ action.status === 'executing' ? '发送中' : '确认发送以上邮件' }}
      </button>
      <button :disabled="busy || action.status !== 'pending'" @click="$emit('reject', action.id)">取消</button>
    </article>
  </aside>
</template>

<style scoped>
.email-confirmations { flex: none; max-height: 35vh; overflow: auto; padding: 10px 20px; border-top: 1px solid #dbe3ed; background: #fffbea; }
article + article { border-top: 1px solid #ddd; margin-top: 12px; padding-top: 12px; }
p { margin: 5px 0; word-break: break-word; }
pre { white-space: pre-wrap; font: inherit; max-height: 150px; overflow: auto; }
button { margin-right: 8px; padding: 6px 10px; cursor: pointer; }
</style>
