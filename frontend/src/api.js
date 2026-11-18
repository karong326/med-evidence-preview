// 前端 HTTP 与流式协议工具：统一管理 JWT、本地请求错误和 /api/ask 的 NDJSON 解析。
// 页面只能用 JWT 载荷决定导航；每次真实权限校验仍由 FastAPI Bearer 依赖完成。
// readNdjson 保留网络分块残片并按行串行交付事件，避免 UTF-8 字符或事件 JSON 被截断。
const tokenKey = "access_token"

// ！！！ JWT 只在这里统一读写；角色解码用于导航，权限仍由后端验证。
export const getToken = () => localStorage.getItem(tokenKey)  //getTokenRole() 解码 JWT 的 role，只用于页面跳转，不验证签名

export function getTokenPayload() {
  // 页面显示与跳转共用同一个 Base64URL 解码；真实权限仍由后端 JWT 校验。
  const token = getToken()
  if (!token) return null
  try {
    const payload = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/")
    return JSON.parse(atob(payload.padEnd(Math.ceil(payload.length / 4) * 4, "=")))
  } catch { return null }
}

export const getTokenRole = () => getTokenPayload()?.role || (getToken() ? "user" : "")
export const getTokenEmail = () => getTokenPayload()?.email || "当前用户"
export const getAuthenticatedHomePath = () => getTokenRole() === "admin" ? "/admin" : "/chat"

export const setToken = (token) => localStorage.setItem(tokenKey, token)

export const clearToken = () => localStorage.removeItem(tokenKey)

// formatErrorMessage()兼容两类后端错误：
//    FastAPI 普通错误：
//    {"detail":"验证码错误"}
//
//    Pydantic 参数校验错误：
//    {"detail":[{"msg":"Value error, 邮箱格式不正确"}]}
// 最终统一转换成人能阅读的字符串。
export function formatErrorMessage(body) {
  const detail = body?.detail
  if (Array.isArray(detail) && detail[0]?.msg) {
    return detail[0].msg.replace(/^Value error,\s*/, "")
  }
  return detail || body?.message || "请求失败"
}

// HTTP 响应块可能截断 JSON 行，因此先缓存残片，再逐行交给页面处理。
// readNdjson() 解决网络分块不等于 JSON 行的问题，逐行恢复事件并调用：
//    onEvent(event)
export async function readNdjson(stream, onEvent) {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { value, done } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const lines = buffer.split("\n")
    buffer = lines.pop()
    for (const line of lines) {
      if (line) await onEvent(JSON.parse(line))
    }
    if (done) break
  }

  if (buffer) await onEvent(JSON.parse(buffer))
}


// 所有普通请求都经过 request() ：
//    读取 JWT
//    → 添加 Authorization
//    → JSON 请求自动添加 Content-Type
//    → 调用 fetch()
//    → 401 时清除 Token
//    → 非成功状态转换成 Error
// 因此页面方法不需要反复编写认证头和错误解析。
export async function request(path, options = {}) {
  // 普通请求统一注入认证头，并把后端错误转换为页面可直接展示的 Error。
  const headers = new Headers(options.headers)
  const token = getToken()
  if (token) headers.set("Authorization", `Bearer ${token}`)
  if (options.body && !headers.has("Content-Type") && !(options.body instanceof FormData)) {
    headers.set("Content-Type", "application/json")
  }

  const response = await fetch(path, { ...options, headers })
  if (response.status === 401) clearToken()
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(formatErrorMessage(body))
  }
  return response
}

// streamAsk() 专门负责 /api/ask
export async function streamAsk(payload, onEvent, signal) {
  const response = await request("/api/ask", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  })
  if (!response.body) throw new Error("服务未返回流式内容")
  await readNdjson(response.body, onEvent)
}
