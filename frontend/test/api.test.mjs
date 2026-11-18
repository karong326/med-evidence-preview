import test from 'node:test'
import assert from 'node:assert/strict'
import { readNdjson, request, getTokenPayload, setToken, getToken, formatErrorMessage } from '../src/api.js'

const storage = new Map()
globalThis.localStorage = { getItem: key => storage.get(key) ?? null,
  setItem: (key, value) => storage.set(key, value), removeItem: key => storage.delete(key) }

test('NDJSON restores split UTF-8, multiple lines and a final line without newline', async () => {
  const bytes = new TextEncoder().encode('{"type":"answer","content":"中文😀"}\n{"type":"done"}')
  const stream = new ReadableStream({ start(controller) {
    for (const byte of bytes) controller.enqueue(Uint8Array.of(byte))
    controller.close()
  } })
  const events = []
  await readNdjson(stream, async event => events.push(event))
  assert.deepEqual(events, [{ type: 'answer', content: '中文😀' }, { type: 'done' }])
})

test('malformed stream fails instead of silently reporting success', async () => {
  const stream = new ReadableStream({ start(c) { c.enqueue(new TextEncoder().encode('invalid\n')); c.close() } })
  await assert.rejects(readNdjson(stream, () => {}), SyntaxError)
})

test('401 clears saved credentials and forwards authorization header', async () => {
  const previous = globalThis.fetch
  setToken('test-token')
  globalThis.fetch = async (path, options) => {
    assert.equal(options.headers.get('Authorization'), 'Bearer test-token')
    return new Response(JSON.stringify({ detail: 'expired' }), { status: 401 })
  }
  try { await assert.rejects(request('/api/conversations'), /expired/); assert.equal(getToken(), null) }
  finally { globalThis.fetch = previous; storage.clear() }
})

test('JSON body gets content type; multipart preserves browser boundary', async () => {
  const previous = globalThis.fetch
  const observed = []
  globalThis.fetch = async (path, options) => { observed.push(options.headers.get('Content-Type')); return new Response('{}') }
  try {
    await request('/api/test', { method: 'POST', body: '{}' })
    await request('/api/test', { method: 'POST', body: new FormData() })
    assert.deepEqual(observed, ['application/json', null])
  } finally { globalThis.fetch = previous }
})

test('malformed JWT is handled and validation errors have readable messages', () => {
  setToken('invalid')
  try { assert.equal(getTokenPayload(), null) } finally { storage.clear() }
  assert.equal(formatErrorMessage({ detail: [{ msg: 'Value error, invalid email' }] }), 'invalid email')
})
