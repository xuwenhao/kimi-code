// apps/kimi-web/src/lib/toolStatus.ts
// Pure mapping from a ToolStatusBadge (derived in messagesToTurns from the tool
// result's outcome × approval-record join) to its i18n key + Badge variant.
// Shared by GenericTool and ExitPlanModeTool so every tool row renders the
// same control-flow badges; i18n strings live in `tools.status.*`.

import type { ToolStatusBadge } from '../types';

export interface StatusBadgeView {
  /** Key into the `tools.status` i18n namespace. */
  key: string;
  variant: 'neutral' | 'warning' | 'danger';
}

export const STATUS_BADGE_VIEW: Record<ToolStatusBadge, StatusBadgeView> = {
  rejected: { key: 'tools.status.rejected', variant: 'danger' },
  revise: { key: 'tools.status.revise', variant: 'warning' },
  cancelled: { key: 'tools.status.cancelled', variant: 'neutral' },
  denied: { key: 'tools.status.denied', variant: 'danger' },
  notRun: { key: 'tools.status.notRun', variant: 'neutral' },
  invalid: { key: 'tools.status.invalid', variant: 'danger' },
  skipped: { key: 'tools.status.skipped', variant: 'neutral' },
  interrupted: { key: 'tools.status.interrupted', variant: 'neutral' },
};

export function statusBadgeView(badge: ToolStatusBadge | undefined): StatusBadgeView | null {
  return badge === undefined ? null : STATUS_BADGE_VIEW[badge];
}
