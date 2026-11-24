<script setup>
defineProps({ actions: { type: Array, default: () => [] }, busy: Boolean, message: String })
defineEmits(["confirm", "reject", "reconcile"])
</script>

<template>
  <aside v-if="actions.length || message" class="email-confirmations" aria-label="邮件确认">
    <p v-if="message" role="status">{{ message }}</p>
    <article v-for="action in actions" :key="action.id">
      <strong v-if="action.status === 'unknown'">发送结果未知，请先核实收件情况。系统不会自动重发。</strong>
      <strong v-else-if="action.status === 'executing'">邮件发送中</strong>
      <strong v-else>待确认邮件 · 尚未发送</strong>
      <p>收件人：{{ action.recipient }}</p>
      <p>主题：{{ action.subject }}</p>
      <pre>{{ action.body }}</pre>
      <button v-if="action.status === 'pending'" :disabled="busy" @click="$emit('confirm', action.id)">
        确认发送以上邮件
      </button>
      <button v-if="action.status === 'pending'" :disabled="busy" @click="$emit('reject', action.id)">取消</button>
      <template v-if="action.status === 'unknown'">
        <button :disabled="busy" @click="$emit('reconcile', action.id, true)">已核实收到，记录结果</button>
        <button :disabled="busy" @click="$emit('reconcile', action.id, false)">已核实未发送，记录结果</button>
      </template>
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
