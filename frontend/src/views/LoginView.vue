<!-- 登录页：支持邮箱验证码和访问令牌两种身份入口，并在发送/登录前复用一次性图片人机验证。
     页面维护验证码冷却与按钮状态，成功后只保存 JWT 并按其角色跳转；认证判断由后端完成。 -->
<script setup>
import { onBeforeUnmount, ref } from "vue"
import { useRouter } from "vue-router"

import { request, setToken } from "../api.js"

const router = useRouter()
// 邮箱、访问令牌和图片验证码共用当前页面，但各自保留独立忙碌状态。
const email = ref("")
const code = ref("")
const accessCode = ref("")
const challengeId = ref("")
const captchaVisible = ref(false)
const captchaImage = ref("")
const captchaAnswer = ref("")
const captchaAction = ref("")
const accessReady = ref(false)
const accessHumanToken = ref("")
const message = ref("")
const sending = ref(false)
const loggingIn = ref(false)
const accessLoggingIn = ref(false)
const humanChecking = ref(false)
const cooldownSeconds = ref(0)
let cooldownTimer

function startCooldown() {
  cooldownSeconds.value = 60
  cooldownTimer = window.setInterval(() => {
    cooldownSeconds.value -= 1
    if (cooldownSeconds.value <= 0) window.clearInterval(cooldownTimer)
  }, 1000)
}

// 两种登录入口都先完成同一套一次性人机验证。
async function openCaptcha(action) {
  captchaAction.value = action
  captchaVisible.value = true
  captchaAnswer.value = ""
  message.value = ""
  const response = await request("/api/auth/captcha")
  const result = await response.json()
  challengeId.value = result.challenge_id
  captchaImage.value = result.image_data_url
}

function closeCaptcha() {
  captchaVisible.value = false
  captchaAnswer.value = ""
  captchaImage.value = ""
  challengeId.value = ""
}

async function verifyCaptcha() {
  humanChecking.value = true
  message.value = ""
  try {
    const response = await request("/api/auth/human-check", {
      method: "POST",
      body: JSON.stringify({ challenge_id: challengeId.value, answer: captchaAnswer.value }),
    })
    const humanToken = (await response.json()).human_token
    captchaVisible.value = false
    if (captchaAction.value === "email") await sendCode(humanToken)
    else {
      accessHumanToken.value = humanToken
      accessReady.value = true
    }
  } catch (error) {
    const errorMessage = error.message
    await openCaptcha(captchaAction.value)
    message.value = errorMessage
  } finally {
    humanChecking.value = false
  }
}

async function sendCode(humanToken) {
  sending.value = true
  message.value = ""
  try {
    const response = await request("/api/auth/email-code", {
      method: "POST",
      body: JSON.stringify({ email: email.value, human_token: humanToken }),
    })
    message.value = (await response.json()).message
    startCooldown()
  } catch (error) {
    message.value = error.message
  } finally {
    sending.value = false
  }
}

async function login() {
  loggingIn.value = true
  message.value = ""
  try {
    const response = await request("/api/auth/login/email", {
      method: "POST",
      body: JSON.stringify({ email: email.value, code: code.value }),
    })
    const result = await response.json()
    setToken(result.access_token)
    await router.push("/chat")
  } catch (error) {
    message.value = error.message
  } finally {
    loggingIn.value = false
  }
}

function visitorId() {
  // 同一浏览器复用稳定标识；后端只保存它的 SHA-256，不保存原值。
  const key = "med_evidence_visitor_id"
  let value = localStorage.getItem(key)
  if (!value) {
    value = crypto.randomUUID()
    localStorage.setItem(key, value)
  }
  return value
}

async function loginWithAccessCode() {
  accessLoggingIn.value = true
  message.value = ""
  try {
    const response = await request("/api/auth/login/access-token", {
      method: "POST",
      body: JSON.stringify({ access_code: accessCode.value, visitor_id: visitorId(), human_token: accessHumanToken.value }),
    })
    const result = await response.json()
    setToken(result.access_token)
    await router.push(result.role === "admin" ? "/admin" : "/chat")
  } catch (error) {
    message.value = error.message
  } finally {
    accessLoggingIn.value = false
  }
}

onBeforeUnmount(() => window.clearInterval(cooldownTimer))
</script>

<template>
  <main class="login-page">
    <form class="login-card" @submit.prevent="login">
      <div class="brand-mark">＋</div>
      <h1>MedEvidence</h1>
      <p>医学证据智能问答</p>
      <label>邮箱<input v-model="email" type="email" autocomplete="email" required></label>
      <div class="code-row">
        <label>验证码<input v-model="code" inputmode="numeric" maxlength="6" required></label>
        <button type="button" :disabled="sending || cooldownSeconds > 0" @click="openCaptcha('email')">{{ sending ? "发送中" : cooldownSeconds ? `${cooldownSeconds}秒后重试` : "获取验证码" }}</button>
      </div>
      <button class="login-button" type="submit" :disabled="loggingIn">{{ loggingIn ? "登录中" : "登录" }}</button>
      <button class="demo-entry" type="button" @click="openCaptcha('demo')">访问令牌登录</button>
      <div v-if="accessReady" class="demo-login">
        <span>访问令牌</span>
        <input v-model="accessCode" type="password" autocomplete="off" placeholder="请输入访问令牌">
        <button type="button" :disabled="accessLoggingIn" @click="loginWithAccessCode">{{ accessLoggingIn ? "验证中" : "访问登录" }}</button>
      </div>
      <p class="status" aria-live="polite">{{ message }}</p>
    </form>
    <div v-if="captchaVisible" class="captcha-mask">
      <form class="captcha-card" @submit.prevent="verifyCaptcha">
        <button class="captcha-close" type="button" aria-label="关闭安全验证" @click="closeCaptcha">×</button>
        <h2>安全验证</h2>
        <p>请输入图片中的字符后继续。</p>
        <img :src="captchaImage" alt="图片验证码">
        <input v-model="captchaAnswer" autocomplete="off" maxlength="5" placeholder="验证码" required>
        <div class="captcha-actions">
          <button type="button" @click="openCaptcha(captchaAction)">换一张</button>
          <button class="login-button" type="submit" :disabled="humanChecking">{{ humanChecking ? "验证中" : "继续" }}</button>
        </div>
      </form>
    </div>
  </main>
</template>

<style scoped>
.login-page { min-height: 100vh; display: grid; place-items: center; padding: 24px; background: linear-gradient(145deg, #eff6ff, #f8fafc); }
.login-card { width: min(100%, 390px); display: grid; gap: 14px; padding: 34px; border: 1px solid #e2e8f0; border-radius: 18px; background: #fff; box-shadow: 0 18px 50px #0f172a12; }
.brand-mark { width: 42px; height: 42px; display: grid; place-items: center; border-radius: 12px; background: #2563eb; color: #fff; font-size: 28px; }
h1, p { margin: 0; } h1 { font-size: 25px; } p { color: #64748b; font-size: 14px; }
label { display: grid; gap: 7px; color: #334155; font-size: 13px; font-weight: 600; }
input { width: 100%; padding: 10px 11px; border: 1px solid #cbd5e1; border-radius: 8px; outline-color: #2563eb; }
.code-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: end; }.code-row button { height: 39px; }
button { border: 0; border-radius: 8px; padding: 10px 12px; background: #e0e7ff; color: #1e3a8a; font-weight: 600; } button:disabled { opacity: .55; cursor: wait; }
.login-button { background: #2563eb; color: #fff; }.status { min-height: 20px; color: #475569; }
.demo-login { display: grid; gap: 7px; padding-top: 4px; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 13px; }.demo-login button { width: 100%; }
.demo-entry { background: #f1f5f9; color: #334155; }
.captcha-mask { position: fixed; inset: 0; display: grid; place-items: center; padding: 24px; background: #0f172a80; }.captcha-card { position: relative; width: min(100%, 320px); display: grid; gap: 12px; padding: 24px; border-radius: 14px; background: #fff; }.captcha-card h2 { margin: 0; font-size: 20px; }.captcha-card img { width: 180px; height: 70px; border: 1px solid #cbd5e1; border-radius: 6px; }.captcha-close { position: absolute; top: 9px; right: 9px; padding: 2px 8px; background: transparent; color: #64748b; font-size: 22px; line-height: 1; }.captcha-actions { display: flex; gap: 8px; }.captcha-actions button { flex: 1; }
</style>
