import { z } from 'zod';

import { ToolInputDisplaySchema } from './display';
import { isoDateTimeSchema } from './time';

export const messageRoleSchema = z.enum(['user', 'assistant', 'tool', 'system']);
export type MessageRole = z.infer<typeof messageRoleSchema>;

/**
 * Execution outcome of a tool call — the only structured field on the result
 * side. Orthogonal to `is_error` (e.g. a revised plan exit is `not_run` but
 * `is_error: false` so the turn continues). `not_run` means a gate stopped
 * the call before execution; pair it with the `approval_results` map to tell
 * user rejections from policy denials and hook blocks.
 */
export const toolOutcomeSchema = z.enum([
  'completed',
  'failed',
  'not_run',
  'invalid',
  'skipped',
  'cancelled',
  'interrupted',
]);
export type ToolOutcome = z.infer<typeof toolOutcomeSchema>;

export const textContentSchema = z.object({
  type: z.literal('text'),
  text: z.string(),
});
export type TextContent = z.infer<typeof textContentSchema>;

export const toolUseContentSchema = z.object({
  type: z.literal('tool_use'),
  tool_call_id: z.string().min(1),
  tool_name: z.string().min(1),
  input: z.unknown(),
  // Structured client-only rendering payload carried over from
  // `tool.call.started` (`toolData`). Only whitelisted kinds that cannot be
  // rebuilt from `input` are persisted (e.g. `plan_review` — the plan body).
  tool_data: ToolInputDisplaySchema.optional(),
});
export type ToolUseContent = z.infer<typeof toolUseContentSchema>;

export const toolResultContentSchema = z.object({
  type: z.literal('tool_result'),
  tool_call_id: z.string().min(1),
  output: z.unknown(),
  is_error: z.boolean().optional(),
  outcome: toolOutcomeSchema.optional(),
});
export type ToolResultContent = z.infer<typeof toolResultContentSchema>;

export const imageSourceSchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('url'), url: z.string().min(1) }),
  z.object({
    kind: z.literal('base64'),
    media_type: z.string().min(1),
    data: z.string().min(1),
  }),
  z.object({ kind: z.literal('file'), file_id: z.string().min(1) }),
]);
export type ImageSource = z.infer<typeof imageSourceSchema>;

export const imageContentSchema = z.object({
  type: z.literal('image'),
  source: imageSourceSchema,
});
export type ImageContent = z.infer<typeof imageContentSchema>;

// Video uses the same source shape as image (url / base64 / uploaded file id).
export const videoContentSchema = z.object({
  type: z.literal('video'),
  source: imageSourceSchema,
});
export type VideoContent = z.infer<typeof videoContentSchema>;

export const fileContentSchema = z.object({
  type: z.literal('file'),
  file_id: z.string().min(1),
  name: z.string(),
  media_type: z.string().min(1),
  size: z.number().int().nonnegative(),
});
export type FileContent = z.infer<typeof fileContentSchema>;

export const thinkingContentSchema = z.object({
  type: z.literal('thinking'),
  thinking: z.string(),
  signature: z.string().optional(),
});
export type ThinkingContent = z.infer<typeof thinkingContentSchema>;

export const messageContentSchema = z.discriminatedUnion('type', [
  textContentSchema,
  toolUseContentSchema,
  toolResultContentSchema,
  imageContentSchema,
  videoContentSchema,
  fileContentSchema,
  thinkingContentSchema,
]);
export type MessageContent = z.infer<typeof messageContentSchema>;

export const messageSchema = z.object({
  id: z.string().min(1),
  session_id: z.string().min(1),
  role: messageRoleSchema,
  content: z.array(messageContentSchema),
  created_at: isoDateTimeSchema,
  prompt_id: z.string().min(1).optional(),
  parent_message_id: z.string().min(1).optional(),
  metadata: z.record(z.string(), z.unknown()).optional(),
});

export type Message = z.infer<typeof messageSchema>;
