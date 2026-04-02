/**
 * Trajectory Logger — captures complete API request/response pairs for training data.
 *
 * Writes to ~/.claude/projects/<cwd>/trajectory.jsonl
 * Each line is a JSON object with:
 *   - type: "api_request" | "api_response"
 *   - timestamp: ISO string
 *   - sessionId: session UUID
 *   - turnIndex: which API call in this session (0-indexed)
 *   - For requests: { system, messages, tools, model, thinkingConfig }
 *   - For responses: { content } (full assistant message content blocks)
 */
import { join } from 'node:path'
import { appendFile, mkdir } from 'node:fs/promises'
import { getSessionId } from '../bootstrap/state.js'
import type { SystemPrompt } from './systemPromptType.js'
import type { Message, AssistantMessage } from '../types/message.js'
import { normalizeMessagesForAPI } from './messages.js'

let turnIndex = 0

function getTrajectoryPath(): string {
  const homeDir = process.env.HOME || process.env.USERPROFILE || '.'
  const cwd = process.cwd().replace(/\//g, '-')
  return join(homeDir, '.claude', 'projects', cwd, 'trajectory.jsonl')
}

async function appendEntry(entry: Record<string, unknown>): Promise<void> {
  const filePath = getTrajectoryPath()
  try {
    await appendFile(filePath, JSON.stringify(entry) + '\n', 'utf-8')
  } catch (e: any) {
    if (e?.code === 'ENOENT') {
      const dir = filePath.substring(0, filePath.lastIndexOf('/'))
      await mkdir(dir, { recursive: true })
      await appendFile(filePath, JSON.stringify(entry) + '\n', 'utf-8')
    }
    // Silently ignore other errors — don't break the main flow
  }
}

export async function logAPIRequest(opts: {
  systemPrompt: SystemPrompt
  messages: Message[]
  tools?: { name: string }[]
  model?: string
  thinkingConfig?: unknown
}): Promise<void> {
  const normalized = normalizeMessagesForAPI(opts.messages)
  const serializedMessages = normalized.map(m => ({
    role: m.message.role,
    content: m.message.content,
  }))

  await appendEntry({
    type: 'api_request',
    timestamp: new Date().toISOString(),
    sessionId: getSessionId(),
    turnIndex,
    system: opts.systemPrompt,
    messages: serializedMessages,
    tools: opts.tools?.map(t => t.name),
    model: opts.model,
    thinkingConfig: opts.thinkingConfig,
  })
}

export async function logAPIResponse(
  assistantMessages: AssistantMessage[],
): Promise<void> {
  const contentBlocks = assistantMessages.flatMap(m => m.message.content)

  await appendEntry({
    type: 'api_response',
    timestamp: new Date().toISOString(),
    sessionId: getSessionId(),
    turnIndex,
    content: contentBlocks,
  })

  turnIndex++
}

export function resetTurnIndex(): void {
  turnIndex = 0
}
