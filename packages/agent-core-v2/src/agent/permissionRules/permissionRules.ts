import { createDecorator } from "#/_base/di/instantiation";
import type { ApprovalResponse } from "@moonshot-ai/protocol";

/** Who produced an approval decision: a prompted user, a deny policy, or
 *  auto-mode approval. */
export type PermissionApprovalSource = 'user' | 'policy' | 'auto';

/** A policy denial recorded as an approval result. Never crosses the
 *  client-facing `ApprovalResponse` contract — it only exists on persisted
 *  approval-result records and the `permission.approval.resolved` event. */
export interface PermissionPolicyDenial {
  readonly decision: 'denied';
}

export interface PermissionApprovalResultRecord {
  readonly turnId: number;
  readonly toolCallId: string;
  readonly toolName: string;
  readonly action: string;
  readonly sessionApprovalRule?: string;
  readonly source: PermissionApprovalSource;
  readonly result: ApprovalResponse | PermissionPolicyDenial;
}

export type PermissionRuleDecision = 'allow' | 'deny' | 'ask';

export type PermissionRuleScope = 'turn-override' | 'session-runtime' | 'project' | 'user';

export interface PermissionRule {
  readonly decision: PermissionRuleDecision;
  readonly scope: PermissionRuleScope;
  readonly pattern: string;
  readonly reason?: string;
}

export interface IAgentPermissionRulesService {
  readonly _serviceBrand: undefined;

  readonly rules: readonly PermissionRule[];
  readonly sessionApprovalRulePatterns: readonly string[];
  addRules(rules: readonly PermissionRule[]): void;
  recordApprovalResult(record: PermissionApprovalResultRecord): void;
}

export const IAgentPermissionRulesService =
  createDecorator<IAgentPermissionRulesService>('agentPermissionRulesService');
