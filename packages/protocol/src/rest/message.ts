/**
 *   GET /v1/sessions/{session_id}/messages
 *     Query: cursorQuery + role
 *     Default page_size=50; max 100 (per SCHEMAS §1.3).
 *     Response data: Page<Message>
 *
 *   GET /v1/sessions/{session_id}/messages/{message_id}
 *     Response data: Message
 *     Errors: 40401 (session.not_found) + 40403 (message.not_found)
 */

import { z } from 'zod';

import { approvalResultSchema } from '../approval';
import { messageRoleSchema, messageSchema } from '../message';
import { cursorQuerySchema } from '../pagination';

export const listMessagesQuerySchema = cursorQuerySchema.and(
  z.object({
    role: messageRoleSchema.optional(),
  }),
);
export type ListMessagesQuery = z.infer<typeof listMessagesQuerySchema>;

export const listMessagesResponseSchema = z.object({
  items: z.array(messageSchema),
  has_more: z.boolean(),
  // Approval decisions keyed by tool_call_id, collected from persisted
  // `permission.record_approval_result` records covering the returned page.
  // Optional for cross-version tolerance: older servers do not send it.
  approval_results: z.record(z.string(), approvalResultSchema).optional(),
});
export type ListMessagesResponse = z.infer<typeof listMessagesResponseSchema>;

export const getMessageResponseSchema = messageSchema;
export type GetMessageResponse = z.infer<typeof getMessageResponseSchema>;
