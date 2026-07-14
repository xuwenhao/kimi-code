// apps/kimi-web/test/daemon-client.test.ts
// DaemonKimiWebApi.getSessionGoal — wire → app mapping of GET /sessions/{id}/goal:
// a present snapshot, explicit null (no active goal), and the request URL.

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { DaemonKimiWebApi } from '../src/api/daemon/client';
import { toAppApprovalRequest, toAppEvent } from '../src/api/daemon/mappers';
import type { WireEvent } from '../src/api/daemon/wire';

function envelope(data: unknown): Response {
  return new Response(JSON.stringify({ code: 0, msg: '', data }), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

const WIRE_GOAL = {
  goalId: 'goal_1',
  objective: 'fix all lint warnings',
  status: 'active',
  turnsUsed: 1,
  tokensUsed: 0,
  wallClockMs: 0,
  budget: {
    tokenBudget: null,
    turnBudget: null,
    wallClockBudgetMs: null,
    remainingTokens: null,
    remainingTurns: null,
    remainingWallClockMs: null,
    tokenBudgetReached: false,
    turnBudgetReached: false,
    wallClockBudgetReached: false,
    overBudget: false,
  },
};

function createApi(): DaemonKimiWebApi {
  return new DaemonKimiWebApi({
    serverHttpUrl: 'http://daemon.test',
    clientId: 'web_test',
    clientName: 'test',
    clientVersion: '0.0.0',
    clientUiMode: 'test',
  });
}

describe('DaemonKimiWebApi.getSessionGoal', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps a present goal snapshot', async () => {
    vi.mocked(fetch).mockResolvedValue(envelope(WIRE_GOAL));
    const goal = await createApi().getSessionGoal('sess_1');
    expect(goal?.objective).toBe('fix all lint warnings');
    expect(goal?.status).toBe('active');
    expect(goal?.turnsUsed).toBe(1);
  });

  it('maps null to null (no active goal)', async () => {
    vi.mocked(fetch).mockResolvedValue(envelope(null));
    const goal = await createApi().getSessionGoal('sess_1');
    expect(goal).toBeNull();
  });

  it('requests the session goal endpoint', async () => {
    vi.mocked(fetch).mockResolvedValue(envelope(null));
    await createApi().getSessionGoal('sess_42');
    expect(vi.mocked(fetch).mock.calls[0]?.[0]).toBe(
      'http://daemon.test/api/v1/sessions/sess_42/goal',
    );
  });
});

// ---------------------------------------------------------------------------
// approval_results side maps + approval_data
// ---------------------------------------------------------------------------

const WIRE_SESSION = {
  id: 'sess_1',
  title: 's',
  created_at: '2026-01-01T00:00:00.000Z',
  updated_at: '2026-01-01T00:00:00.000Z',
  status: 'idle',
  archived: false,
  metadata: { cwd: '/workspace' },
  agent_config: { model: 'kimi-code' },
  usage: {
    input_tokens: 0,
    output_tokens: 0,
    cache_read_tokens: 0,
    cache_creation_tokens: 0,
    total_cost_usd: 0,
    context_tokens: 0,
    context_limit: 0,
    turn_count: 0,
  },
  permission_rules: [],
  message_count: 0,
  last_seq: 0,
};

function wireSnapshot(extra: Record<string, unknown> = {}) {
  return {
    as_of_seq: 5,
    epoch: 'ep_1',
    session: WIRE_SESSION,
    messages: { items: [], has_more: false },
    in_flight_turn: null,
    pending_approvals: [],
    pending_questions: [],
    ...extra,
  };
}

describe('DaemonKimiWebApi.listMessages approval_results', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('maps the approval_results side map', async () => {
    vi.mocked(fetch).mockResolvedValue(
      envelope({
        items: [],
        has_more: false,
        approval_results: {
          call_1: {
            decision: 'rejected',
            source: 'user',
            feedback: 'Add verification steps.',
            selected_label: 'Revise',
          },
        },
      }),
    );
    const page = await createApi().listMessages('sess_1');
    expect(page.approvalResults?.['call_1']).toEqual({
      decision: 'rejected',
      source: 'user',
      scope: undefined,
      feedback: 'Add verification steps.',
      selectedLabel: 'Revise',
    });
  });

  it('omits approvalResults when the server sends none (legacy)', async () => {
    vi.mocked(fetch).mockResolvedValue(envelope({ items: [], has_more: false }));
    const page = await createApi().listMessages('sess_1');
    expect(page.approvalResults).toBeUndefined();
  });
});

describe('DaemonKimiWebApi.getSessionSnapshot approval_results', () => {
  beforeEach(() => {
    vi.stubGlobal('fetch', vi.fn());
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('hydrates approvalResults from the snapshot side map', async () => {
    vi.mocked(fetch).mockResolvedValue(
      envelope(
        wireSnapshot({
          approval_results: {
            call_9: { decision: 'approved', source: 'auto' },
          },
        }),
      ),
    );
    const snap = await createApi().getSessionSnapshot('sess_1');
    expect(snap.approvalResults['call_9']).toMatchObject({
      decision: 'approved',
      source: 'auto',
    });
  });

  it('defaults to an empty map on a legacy snapshot', async () => {
    vi.mocked(fetch).mockResolvedValue(envelope(wireSnapshot()));
    const snap = await createApi().getSessionSnapshot('sess_1');
    expect(snap.approvalResults).toEqual({});
  });
});

describe('toAppApprovalRequest approval_data', () => {
  const base = {
    approval_id: 'ap_1',
    session_id: 'sess_1',
    tool_call_id: 'tc_1',
    tool_name: 'ExitPlanMode',
    action: 'plan_review',
    expires_at: '2099-01-01T00:00:00.000Z',
    created_at: '2026-01-01T00:00:00.000Z',
  };

  it('reads the display payload from approval_data', () => {
    const req = toAppApprovalRequest({
      ...base,
      approval_data: { kind: 'plan_review', plan: '# New' },
    });
    expect(req.display).toMatchObject({ plan: '# New' });
  });
});

describe('toAppEvent event.approval.resolved', () => {
  it('carries tool_call_id and the full decision', () => {
    const event = toAppEvent({
      type: 'event.approval.resolved',
      seq: 3,
      session_id: 'sess_1',
      timestamp: '2026-01-01T00:00:00.000Z',
      payload: {
        approval_id: 'ap_1',
        tool_call_id: 'tc_1',
        decision: 'rejected',
        source: 'user',
        feedback: 'Add verification steps.',
        selected_label: 'Revise',
        resolved_by: 'user',
        resolved_at: '2026-01-01T00:00:01.000Z',
      },
    } as unknown as WireEvent);

    expect(event).toMatchObject({
      type: 'approvalResolved',
      sessionId: 'sess_1',
      approvalId: 'ap_1',
      toolCallId: 'tc_1',
      decision: 'rejected',
      source: 'user',
      feedback: 'Add verification steps.',
      selectedLabel: 'Revise',
    });
  });

  it('omits the new fields for a legacy broadcast', () => {
    const event = toAppEvent({
      type: 'event.approval.resolved',
      seq: 4,
      session_id: 'sess_1',
      timestamp: '2026-01-01T00:00:00.000Z',
      payload: {
        approval_id: 'ap_2',
        decision: 'approved',
        resolved_by: 'user',
        resolved_at: '2026-01-01T00:00:01.000Z',
      },
    } as unknown as WireEvent);

    expect(event).toMatchObject({ type: 'approvalResolved', decision: 'approved' });
    expect(event.type === 'approvalResolved' ? event.toolCallId : 'n/a').toBeUndefined();
  });
});
