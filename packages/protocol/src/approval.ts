import { z } from 'zod';

import { isoDateTimeSchema } from './time';

export const approvalDecisionSchema = z.enum(['approved', 'rejected', 'cancelled']);
export type ApprovalDecision = z.infer<typeof approvalDecisionSchema>;

export const approvalScopeSchema = z.enum(['session']);
export type ApprovalScope = z.infer<typeof approvalScopeSchema>;

export const approvalRequestSchema = z.object({
  approval_id: z.string().min(1),
  session_id: z.string().min(1),
  turn_id: z.number().int().nonnegative().optional(),
  tool_call_id: z.string().min(1),
  tool_name: z.string().min(1),
  action: z.string(),
  /** @deprecated Use `approval_data`. Legacy (v1) emitters still send this. */
  tool_input_display: z.unknown().optional(),
  approval_data: z.unknown().optional(),
  created_at: isoDateTimeSchema,
  expires_at: isoDateTimeSchema,
});
export type ApprovalRequest = z.infer<typeof approvalRequestSchema>;

export const approvalResponseSchema = z.object({
  decision: approvalDecisionSchema,
  scope: approvalScopeSchema.optional(),
  feedback: z.string().optional(),
  selected_label: z.string().optional(),
});
export type ApprovalResponse = z.infer<typeof approvalResponseSchema>;

/**
 * A persisted approval decision, keyed by tool_call_id in the
 * `approval_results` side map on message-list responses and snapshots.
 * `source` tells who made the decision: a prompted user, a deny policy, or
 * auto-approval. `decision: 'denied'` only occurs with `source: 'policy'`
 * (a policy denial never produced a user-facing response).
 */
export const approvalResultSchema = z.object({
  decision: z.enum(['approved', 'rejected', 'cancelled', 'denied']),
  source: z.enum(['user', 'policy', 'auto']),
  scope: approvalScopeSchema.optional(),
  feedback: z.string().optional(),
  selected_label: z.string().optional(),
});
export type ApprovalResult = z.infer<typeof approvalResultSchema>;
