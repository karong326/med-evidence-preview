import test from 'node:test'
import assert from 'node:assert/strict'
import { splitTextForDisplay, appendAnswerForDisplay, clonePanelItems } from '../src/chatStream.js'

test('display splitting preserves Unicode characters', () => {
  assert.deepEqual(splitTextForDisplay('中😀文', 1), ['中', '😀', '文'])
})

test('answer rendering preserves order while waiting for frames', async () => {
  let result = ''; let frames = 0
  await appendAnswerForDisplay('abcdef', chunk => { result += chunk }, async () => { frames++ }, 2)
  assert.equal(result, 'abcdef'); assert.equal(frames, 3)
})

test('panel list cloning does not append into the original list', () => {
  const original = [{ content: 'evidence' }]
  const copy = clonePanelItems(original); copy.push({ content: 'second' })
  assert.equal(original.length, 1)
})
