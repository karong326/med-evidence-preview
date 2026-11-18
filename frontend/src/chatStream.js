// 聊天页面的纯展示辅助：把后端已经到达的回答进一步拆成短 Unicode 块并等待渲染帧。
// 它不改变 NDJSON 协议、证据或执行记录，只减少浏览器把长回答整段跳出的视觉问题。
const DEFAULT_DISPLAY_CHUNK_SIZE = 12

export function splitTextForDisplay(text, chunkSize = DEFAULT_DISPLAY_CHUNK_SIZE) {
  const characters = Array.from(text)
  const chunks = []
  for (let index = 0; index < characters.length; index += chunkSize) {
    chunks.push(characters.slice(index, index + chunkSize).join(""))
  }
  return chunks
}

function waitForAnimationFrame() {
  return new Promise((resolve) => window.requestAnimationFrame(() => resolve()))
}

export async function appendAnswerForDisplay(
  text,
  append,
  waitForFrame = waitForAnimationFrame,
  chunkSize = DEFAULT_DISPLAY_CHUNK_SIZE,
) {
  for (const part of splitTextForDisplay(text, chunkSize)) {
    append(part)
    await waitForFrame()
  }
}

export function clonePanelItems(items) {
  return [...(items || [])]
}

export function evidencePanelState(rightOpen, activeTab) {
  return {
    rightOpen: true,
    activeTab: rightOpen ? activeTab : "evidence",
  }
}
