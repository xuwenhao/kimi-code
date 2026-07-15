/**
 * `PromptService` unit tests.
 *
 * Hermetic: a fake `ICoreProcessService` returns canned session list + records
 * the `prompt` / `cancel` payloads. A stub `IEventService` collects published
 * events into an array we can inspect and drives synthesis via
 * `bus.publish(turn.*)` → PromptService's private subscriber. A stub
 * `ISessionService` exposes `onDidClose` so cleanup tests can trigger it.
 *
 * Coverage:
 *   - submit returns PromptItem with status='running' or status='queued'
 *   - submit registers an active prompt → second submit enters daemon queue
 *   - concurrent submit startup preserves one active prompt and FIFO queueing
 *   - submit translates protocol content → kosong content (text + image_url)
 *   - submit on unknown sid → SessionNotFoundError
 *   - submit on a session with an active completed/aborted prompt succeeds
 *   - bus.publish of `turn.started` captures turnId (via PromptService subscriber)
 *   - bus.publish of `turn.ended` (top-level, completed) synthesizes prompt.completed
 *   - bus.publish of `turn.ended` with reason=cancelled synthesizes prompt.aborted
 *   - goal continuation turns retain one prompt until a verified terminal boundary
 *   - bus.publish of nested turn.ended ignored (non-top-level)
 *   - bus.publish on events for an unknown session is a no-op
 *   - abort() rejects PromptNotFoundError when no active prompt
 *   - abort() returns {aborted: true} + publishes prompt.aborted
 *   - abort owns queue draining until cancellation settles, including failure recovery
 *   - second abort() → PromptAlreadyCompletedError (40903)
 *   - busy submit queues instead of throwing; list returns active + queued
 *   - steer removes queued prompts and dispatches core.rpc.steer
 *   - failed steer restores queued prompts
 *   - per-request stateless controls (model / thinking / permission_mode /
 *     plan_mode) bootstrap once, diff-dispatch on change, no-op on match,
 *     reseed after session close, agent.status.updated mirrors into shadow.
 */

import { createControlledPromise } from '@antfu/utils';
import { describe, expect, it, vi } from 'vitest';

import { Emitter } from '../../src';

import type {
  CoreRPC,
  Event,
  GoalToolResult,
  SessionSummary,
} from '../../src';
import type { PromptSubmission, Session } from '@moonshot-ai/protocol';

import {
  type IAuthSummaryService,
  type IEventService,
  type ICoreProcessService,
  type ILogService,
  type ISessionService,
  PromptAlreadyCompletedError,
  PromptNotFoundError,
  PromptService,
  SessionBusyError,
  SessionNotFoundError,
} from '../../src/services';

const SID = 'sess_01PT';
const SESSION_CREATED_AT = 1_700_000_000_000;

function goalResult(status: 'active' | 'blocked' | 'paused'): GoalToolResult {
  return {
    goal: {
      goalId: 'goal_1',
      objective: 'Finish the current task',
      status,
      turnsUsed: 1,
      tokensUsed: 100,
      wallClockMs: 1_000,
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
    },
  };
}

function mkSummary(id = SID): SessionSummary {
  return {
    id,
    workDir: '/tmp/ws',
    sessionDir: `/tmp/sessions/${id}`,
    createdAt: SESSION_CREATED_AT,
    updatedAt: SESSION_CREATED_AT,
  };
}

/**
 * Default body for a submit() that exercises the per-turn override path —
 * all four runtime controls are populated, so bootstrap + diff-dispatch
 * fire. Spread overrides on top per-test as needed. Tests that want the
 * content-only path (zero bootstrap, zero setters) use `mkBodyMinimal`.
 */
function mkBody(over: Partial<PromptSubmission> = {}): PromptSubmission {
  return {
    content: [{ type: 'text', text: 'hi' }],
    model: 'kimi-code/k2',
    thinking: 'off',
    permission_mode: 'manual',
    plan_mode: false,
    ...over,
  };
}

/**
 * Minimal submit body — content only, no per-turn overrides. Triggers the
 * stateful-session path: no bootstrap RPCs, no setter dispatch, no
 * dispatch-log entries. Mirrors what the canonical web client sends after
 * setting state via `POST /sessions/{sid}/profile`.
 */
function mkBodyMinimal(over: Partial<PromptSubmission> = {}): PromptSubmission {
  return {
    content: [{ type: 'text', text: 'hi' }],
    ...over,
  };
}

interface RpcRecord {
  promptCalls: unknown[];
  steerCalls: unknown[];
  cancelCalls: unknown[];
  setModelCalls: unknown[];
  setThinkingCalls: unknown[];
  setPermissionCalls: unknown[];
  enterPlanCalls: unknown[];
  cancelPlanCalls: unknown[];
  getSwarmModeCalls: number;
  startBtwCalls: unknown[];
  enterSwarmCalls: unknown[];
  exitSwarmCalls: unknown[];
  createGoalCalls: unknown[];
  pauseGoalCalls: unknown[];
  resumeGoalCalls: unknown[];
  cancelGoalCalls: unknown[];
  getGoalCalls: unknown[];
  getConfigCalls: number;
  getPermissionCalls: number;
  getPlanCalls: number;
}

interface BridgeStubOptions {
  /** Initial bootstrap values returned by getConfig/getPermission/getPlan. */
  config?: { modelAlias?: string; thinkingEffort?: string };
  permission?: { mode: 'manual' | 'yolo' | 'auto' };
  plan?: null | { id: string; content: string; path: string };
  sessions?: SessionSummary[];
  onPrompt?: (payload: unknown) => void | Promise<void>;
  onCancel?: (payload: unknown) => void | Promise<void>;
  onGetGoal?: (payload: unknown) => GoalToolResult | Promise<GoalToolResult>;
}

function makeBridge(
  opts: BridgeStubOptions = {},
): { bridge: ICoreProcessService; record: RpcRecord } {
  const record: RpcRecord = {
    promptCalls: [],
    steerCalls: [],
    cancelCalls: [],
    setModelCalls: [],
    setThinkingCalls: [],
    setPermissionCalls: [],
    enterPlanCalls: [],
    cancelPlanCalls: [],
    getSwarmModeCalls: 0,
    startBtwCalls: [],
    enterSwarmCalls: [],
    exitSwarmCalls: [],
    createGoalCalls: [],
    pauseGoalCalls: [],
    resumeGoalCalls: [],
    cancelGoalCalls: [],
    getGoalCalls: [],
    getConfigCalls: 0,
    getPermissionCalls: 0,
    getPlanCalls: 0,
  };
  const config = {
    cwd: '/tmp/ws',
    modelCapabilities: {} as unknown,
    thinkingEffort: opts.config?.thinkingEffort ?? 'off',
    systemPrompt: '',
    modelAlias: opts.config?.modelAlias ?? 'kimi-code/k2',
  };
  const permission = { mode: opts.permission?.mode ?? 'manual', rules: [] };
  const plan = opts.plan === undefined ? null : opts.plan;
  const sessions = opts.sessions ?? [mkSummary()];

  const rpc: Partial<CoreRPC> = {
    listSessions: vi.fn().mockImplementation(async (payload) => {
      // Mirror the store's `sessionId` filter: a filtered lookup returns
      // only the matching summary (or [] when unknown), which is what
      // `_requireSession` now relies on for its existence check.
      const id = (payload as { sessionId?: string } | undefined)?.sessionId;
      return id === undefined ? sessions : sessions.filter((s) => s.id === id);
    }),
    resumeSession: vi.fn().mockResolvedValue(undefined as unknown as never),
    prompt: vi.fn().mockImplementation(async (payload) => {
      record.promptCalls.push(payload);
      await opts.onPrompt?.(payload);
    }),
    steer: vi.fn().mockImplementation(async (payload) => {
      record.steerCalls.push(payload);
    }),
    cancel: vi.fn().mockImplementation(async (payload) => {
      record.cancelCalls.push(payload);
      await opts.onCancel?.(payload);
    }),
    getGoal: vi.fn().mockImplementation(async (payload) => {
      record.getGoalCalls.push(payload);
      return opts.onGetGoal?.(payload) ?? { goal: null };
    }),
    getConfig: vi.fn().mockImplementation(async () => {
      record.getConfigCalls += 1;
      return config;
    }),
    getPermission: vi.fn().mockImplementation(async () => {
      record.getPermissionCalls += 1;
      return permission;
    }),
    getPlan: vi.fn().mockImplementation(async () => {
      record.getPlanCalls += 1;
      return plan;
    }),
    setModel: vi.fn().mockImplementation(async (payload) => {
      record.setModelCalls.push(payload);
      return { model: (payload as { model: string }).model };
    }),
    setThinking: vi.fn().mockImplementation(async (payload) => {
      record.setThinkingCalls.push(payload);
    }),
    setPermission: vi.fn().mockImplementation(async (payload) => {
      record.setPermissionCalls.push(payload);
    }),
    enterPlan: vi.fn().mockImplementation(async (payload) => {
      record.enterPlanCalls.push(payload);
    }),
    cancelPlan: vi.fn().mockImplementation(async (payload) => {
      record.cancelPlanCalls.push(payload);
    }),
    getSwarmMode: vi.fn().mockImplementation(async () => {
      record.getSwarmModeCalls += 1;
      return false;
    }),
    startBtw: vi.fn().mockImplementation(async (payload) => {
      record.startBtwCalls.push(payload);
      return 'agent_btw';
    }),
    enterSwarm: vi.fn().mockImplementation(async (payload) => {
      record.enterSwarmCalls.push(payload);
    }),
    exitSwarm: vi.fn().mockImplementation(async (payload) => {
      record.exitSwarmCalls.push(payload);
    }),
    createGoal: vi.fn().mockImplementation(async (payload) => {
      record.createGoalCalls.push(payload);
      return {
        goalId: 'goal_1',
        objective: (payload as { objective: string }).objective,
        status: 'active',
      };
    }),
    pauseGoal: vi.fn().mockImplementation(async (payload) => {
      record.pauseGoalCalls.push(payload);
      return { goalId: 'goal_1', status: 'paused' };
    }),
    resumeGoal: vi.fn().mockImplementation(async (payload) => {
      record.resumeGoalCalls.push(payload);
      return { goalId: 'goal_1', status: 'active' };
    }),
    cancelGoal: vi.fn().mockImplementation(async (payload) => {
      record.cancelGoalCalls.push(payload);
      return { goalId: 'goal_1', status: 'cancelled' };
    }),
  };
  const bridge: ICoreProcessService = {
    rpc: rpc as CoreRPC,
    ready: vi.fn().mockResolvedValue(undefined),
    dispose: vi.fn(),
    _serviceBrand: undefined,
  };
  return { bridge, record };
}

function makeBus(): {
  bus: IEventService;
  events: Event[];
  triggerSubscribers: (e: Event) => void;
} {
  const events: Event[] = [];
  const emitter = new Emitter<Event>();
  const bus: IEventService = {
    publish: (e: Event) => {
      events.push(e);
      // Drive any subscribers (mirrors EventService publish → onDidPublish fire).
      emitter.fire(e);
    },
    onDidPublish: emitter.event,
    _serviceBrand: undefined,
  };
  function triggerSubscribers(e: Event): void {
    emitter.fire(e);
  }
  return { bus, events, triggerSubscribers };
}

/**
 * Stub `IAuthSummaryService` for hermetic prompt-service tests. Default
 * `ensureReady()` resolves; tests that need to exercise the readiness gate
 * can pass `{ ensureReadyError }` and assert the error surfaces.
 */
function makeAuth(opts: { ensureReadyError?: Error } = {}): IAuthSummaryService {
  return {
    get: vi.fn().mockResolvedValue({
      ready: true,
      providers_count: 1,
      default_model: 'kimi-k2',
      managed_provider: null,
    }),
    ensureReady: vi.fn().mockImplementation(async () => {
      if (opts.ensureReadyError) throw opts.ensureReadyError;
    }),
    _serviceBrand: undefined,
  };
}

/**
 * Stub `ISessionService` for hermetic prompt-service tests. Only the
 * `onDidClose` event accessor is consumed by PromptService; `triggerClose`
 * fires the close event to exercise shadow cleanup.
 */
function makeSessionService(): {
  sessionService: ISessionService;
  triggerClose: (sid: string) => void;
} {
  const closeEmitter = new Emitter<{ sessionId: string }>();
  const createEmitter = new Emitter<{ session: Session }>();
  const sessionService: ISessionService = {
    _serviceBrand: undefined,
    create: vi.fn() as unknown as ISessionService['create'],
    list: vi.fn() as unknown as ISessionService['list'],
    get: vi.fn() as unknown as ISessionService['get'],
    update: vi.fn() as unknown as ISessionService['update'],
    fork: vi.fn() as unknown as ISessionService['fork'],
    listChildren: vi.fn() as unknown as ISessionService['listChildren'],
    createChild: vi.fn() as unknown as ISessionService['createChild'],
    getStatus: vi.fn() as unknown as ISessionService['getStatus'],
    getSessionWarnings: vi.fn() as unknown as ISessionService['getSessionWarnings'],
    compact: vi.fn() as unknown as ISessionService['compact'],
    undo: vi.fn() as unknown as ISessionService['undo'],
    archive: vi.fn() as unknown as ISessionService['archive'],
    onDidCreate: createEmitter.event,
    onDidClose: closeEmitter.event,
  };
  return {
    sessionService,
    triggerClose: (sid: string) => closeEmitter.fire({ sessionId: sid }),
  };
}

class NoopLogService implements ILogService {
  readonly _serviceBrand: undefined;

  debug(): void {}
  info(): void {}
  warn(): void {}
  error(): void {}
  child(): ILogService {
    return this;
  }
}

class ThrowingWarnLogService extends NoopLogService {
  override warn(): void {
    throw new Error('warning sink failed');
  }
}

function newSvc(
  bridge: ICoreProcessService,
  bus: IEventService,
  auth: IAuthSummaryService = makeAuth(),
  sessionService: ISessionService = makeSessionService().sessionService,
  logService: ILogService = new NoopLogService(),
): PromptService {
  return new PromptService(bridge, bus, auth, sessionService, logService);
}

describe('PromptService.submit', () => {
  it('returns ULID-shaped prompt_id + user_message_id derived from it', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    const result = await impl.submit(SID, mkBody());
    expect(result.prompt_id).toMatch(/^prompt_[0-9A-HJKMNP-TV-Z]{26}$/);
    expect(result.user_message_id).toMatch(/^msg_sess_01PT_pending_prompt_/);
  });

  it('translates text + image content to kosong ContentParts', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(
      SID,
      mkBody({
        content: [
          { type: 'text', text: 'hello' },
          { type: 'image', source: { kind: 'url', url: 'https://a.png' } },
        ],
      }),
    );
    expect(record.promptCalls).toHaveLength(1);
    const payload = record.promptCalls[0] as {
      sessionId: string;
      agentId: string;
      input: Array<Record<string, unknown>>;
    };
    expect(payload.sessionId).toBe(SID);
    expect(payload.agentId).toBe('main');
    expect(payload.input).toEqual([
      { type: 'text', text: 'hello' },
      { type: 'image_url', imageUrl: { url: 'https://a.png' } },
    ]);
  });

  it('translates base64 image content to a data URL image part', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(
      SID,
      mkBody({
        content: [
          { type: 'text', text: 'describe this' },
          {
            type: 'image',
            source: {
              kind: 'base64',
              media_type: 'image/png',
              data: 'aGVsbG8=',
            },
          },
        ],
      }),
    );
    const payload = record.promptCalls[0] as {
      input: Array<Record<string, unknown>>;
    };
    expect(payload.input).toEqual([
      { type: 'text', text: 'describe this' },
      {
        type: 'image_url',
        imageUrl: { url: 'data:image/png;base64,aGVsbG8=' },
      },
    ]);
  });

  it('publishes prompt.submitted when a prompt starts running', async () => {
    const { bridge } = makeBridge();
    const { bus, events } = makeBus();
    const impl = newSvc(bridge, bus);
    const body = mkBody({ content: [{ type: 'text', text: 'hello from client a' }] });

    const result = await impl.submit(SID, body);

    const submitted = events.find((event) => event.type === 'prompt.submitted') as
      | {
          type: 'prompt.submitted';
          sessionId: string;
          agentId: string;
          promptId: string;
          userMessageId: string;
          status: string;
          content: readonly PromptSubmission['content'][number][];
        }
      | undefined;
    expect(submitted).toBeDefined();
    expect(submitted?.sessionId).toBe(SID);
    expect(submitted?.agentId).toBe('main');
    expect(submitted).toMatchObject({
      promptId: result.prompt_id,
      userMessageId: result.user_message_id,
      status: 'running',
      content: body.content,
    });
  });

  it('publishes prompt.submitted before core prompt events can start the turn', async () => {
    const { bus, events } = makeBus();
    const { bridge } = makeBridge({
      onPrompt: () => {
        bus.publish({
          type: 'turn.started',
          turnId: 7,
          origin: { kind: 'user' },
          sessionId: SID,
          agentId: 'main',
        } as unknown as Event);
      },
    });
    const impl = newSvc(bridge, bus);

    const result = await impl.submit(SID, mkBodyMinimal());

    expect(events.map((event) => event.type).slice(0, 2)).toEqual([
      'prompt.submitted',
      'turn.started',
    ]);
    expect(events[0]).toMatchObject({
      type: 'prompt.submitted',
      promptId: result.prompt_id,
      status: 'running',
    });
  });

  it('queues a second prompt when a non-terminal prompt is already active', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    const first = await impl.submit(SID, mkBody({ content: [{ type: 'text', text: 'one' }] }));
    const second = await impl.submit(SID, mkBody({ content: [{ type: 'text', text: 'two' }] }));
    const listed = await impl.list(SID);

    expect(first.status).toBe('running');
    expect(second.status).toBe('queued');
    expect(listed.active?.prompt_id).toBe(first.prompt_id);
    expect(listed.queued.map((p) => p.prompt_id)).toEqual([second.prompt_id]);
  });

  it('serializes concurrent submits while the first prompt is still starting', async () => {
    const goalReadStarted = createControlledPromise<void>();
    const releaseGoalRead = createControlledPromise<GoalToolResult>();
    const { bridge, record } = makeBridge({
      onGetGoal: () => {
        goalReadStarted.resolve();
        return releaseGoalRead;
      },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const firstResult = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    await goalReadStarted;
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );

    expect(second.status).toBe('queued');
    expect(record.promptCalls).toHaveLength(0);
    releaseGoalRead.resolve({ goal: null });
    await expect(firstResult).resolves.toMatchObject({ status: 'running' });
    expect(record.promptCalls).toHaveLength(1);
  });

  it('keeps the earlier follower ahead when its preflight finishes after a later follower', async () => {
    const earlierResumeStarted = createControlledPromise<void>();
    const releaseEarlierResume = createControlledPromise<void>();
    const earlierPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'earlier follower') earlierPromptStarted.resolve();
      },
    });
    let resumeCalls = 0;
    vi.mocked(bridge.rpc.resumeSession).mockImplementation(async () => {
      resumeCalls += 1;
      if (resumeCalls === 2) {
        earlierResumeStarted.resolve();
        await releaseEarlierResume;
      }
      return undefined as never;
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );

    const earlierResult = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'earlier follower' }] }),
    );
    await earlierResumeStarted;
    const later = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'later follower' }] }),
    );
    releaseEarlierResume.resolve();
    const earlier = await earlierResult;

    expect((await impl.list(SID)).queued.map((item) => item.prompt_id)).toEqual([
      earlier.prompt_id,
      later.prompt_id,
    ]);

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await earlierPromptStarted;

    expect(record.promptCalls[1]).toEqual({
      sessionId: SID,
      agentId: 'main',
      input: [{ type: 'text', text: 'earlier follower' }],
      promptId: earlier.prompt_id,
    });
  });

  it('starts the queued prompt when active-prompt setup fails', async () => {
    const goalReadStarted = createControlledPromise<void>();
    const releaseGoalRead = createControlledPromise<GoalToolResult>();
    const nextPromptStarted = createControlledPromise<void>();
    let goalRead = 0;
    const { bridge } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 1) {
          goalReadStarted.resolve();
          return releaseGoalRead;
        }
        return { goal: null };
      },
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const firstResult = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    await goalReadStarted;
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );
    const firstRejected = expect(firstResult).rejects.toThrow('goal lookup failed');
    releaseGoalRead.reject(new Error('goal lookup failed'));

    await firstRejected;
    await nextPromptStarted;

    expect(second.status).toBe('queued');
    expect((await impl.list(SID)).active?.prompt_id).toBe(second.prompt_id);
  });

  it('advances after emitting exactly one failed completion for a queued startup failure', async () => {
    const thirdPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'fails during startup') {
          throw new Error('queued startup failed');
        }
        if (text === 'third') thirdPromptStarted.resolve();
      },
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    const failed = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'fails during startup' }] }),
    );
    const third = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'third' }] }),
    );

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await thirdPromptStarted;

    const failedCompletions = events.filter((event) => {
      const candidate = event as unknown as {
        type?: string;
        promptId?: string;
        reason?: string;
      };
      return candidate.type === 'prompt.completed' && candidate.promptId === failed.prompt_id;
    });
    expect(failedCompletions).toHaveLength(1);
    expect(failedCompletions[0]).toMatchObject({ reason: 'failed' });
    expect((await impl.list(SID)).active?.prompt_id).toBe(third.prompt_id);
    expect(
      record.promptCalls.map(
        (payload) => (payload as { input: Array<{ text?: string }> }).input[0]?.text,
      ),
    ).toEqual(['active', 'fails during startup', 'third']);
  });

  it('starts an agent-scoped prompt without querying main-agent goal state', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const result = await impl.submit(
      SID,
      mkBodyMinimal({
        agent_id: 'agent_btw',
        content: [{ type: 'text', text: 'side question' }],
      }),
    );

    expect(result.status).toBe('running');
    expect(record.getGoalCalls).toHaveLength(0);
    expect(record.promptCalls).toHaveLength(1);
  });

  it('runs an agent-scoped prompt without queueing behind the main prompt', async () => {
    const { bridge, record } = makeBridge();
    const { bus, events } = makeBus();
    const impl = newSvc(bridge, bus);

    const main = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'main' }] }));
    const btw = await impl.submit(
      SID,
      mkBodyMinimal({
        agent_id: 'agent_btw',
        content: [{ type: 'text', text: 'side question' }],
      }),
    );
    const listed = await impl.list(SID);

    expect(main.status).toBe('running');
    expect(btw.status).toBe('running');
    expect(listed.active?.prompt_id).toBe(main.prompt_id);
    expect(listed.queued).toHaveLength(0);
    expect(record.promptCalls).toEqual([
      {
        sessionId: SID,
        agentId: 'main',
        input: [{ type: 'text', text: 'main' }],
        promptId: main.prompt_id,
      },
      {
        sessionId: SID,
        agentId: 'agent_btw',
        input: [{ type: 'text', text: 'side question' }],
        promptId: btw.prompt_id,
      },
    ]);
    const submitted = events.filter((event) => event.type === 'prompt.submitted') as Array<{
      type: 'prompt.submitted';
      agentId: string;
      promptId: string;
    }>;
    expect(submitted.map((event) => [event.agentId, event.promptId])).toEqual([
      ['main', main.prompt_id],
      ['agent_btw', btw.prompt_id],
    ]);
  });

  it('publishes prompt.submitted when a prompt is queued', async () => {
    const { bridge } = makeBridge();
    const { bus, events } = makeBus();
    const impl = newSvc(bridge, bus);

    await impl.submit(SID, mkBody({ content: [{ type: 'text', text: 'one' }] }));
    const second = await impl.submit(SID, mkBody({ content: [{ type: 'text', text: 'two' }] }));

    const submitted = events.filter((event) => event.type === 'prompt.submitted') as Array<{
      type: 'prompt.submitted';
      promptId: string;
      status: string;
      content: readonly PromptSubmission['content'][number][];
    }>;
    expect(submitted.map((event) => event.promptId)).toContain(second.prompt_id);
    expect(submitted.at(-1)).toMatchObject({
      promptId: second.prompt_id,
      status: 'queued',
      content: [{ type: 'text', text: 'two' }],
    });
  });

  it('starts the next queued prompt after the active prompt completes', async () => {
    const nextPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const first = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }));
    const second = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }));

    triggerSubscribers({
      type: 'turn.started',
      turnId: 7,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 7,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await nextPromptStarted;

    expect(record.promptCalls).toHaveLength(2);
    expect(record.promptCalls[1]).toEqual({
      sessionId: SID,
      agentId: 'main',
      input: [{ type: 'text', text: 'two' }],
      promptId: second.prompt_id,
    });
    const listed = await impl.list(SID);
    expect(listed.active?.prompt_id).toBe(second.prompt_id);
    expect(listed.queued).toHaveLength(0);
    expect(first.status).toBe('running');
  });

  it('throws SessionNotFoundError on unknown session id', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await expect(impl.submit('sess_missing', mkBody())).rejects.toBeInstanceOf(
      SessionNotFoundError,
    );
  });

  it('clears active state if bridge.prompt() rejects', async () => {
    const { bridge } = makeBridge();
    (bridge.rpc.prompt as unknown as ReturnType<typeof vi.fn>)
      .mockRejectedValueOnce(new Error('boom'))
      .mockResolvedValue(undefined);
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await expect(impl.submit(SID, mkBody())).rejects.toThrowError(/boom/);
    // A second submit must succeed (state was cleared).
    await impl.submit(SID, mkBody());
  });

  it('calls resumeSession before prompt so cross-restart sessions resolve', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody());
    const resumeMock = bridge.rpc.resumeSession as ReturnType<typeof vi.fn>;
    const promptMock = bridge.rpc.prompt as ReturnType<typeof vi.fn>;
    expect(resumeMock).toHaveBeenCalledWith(
      { sessionId: SID },
      { signal: expect.any(AbortSignal) },
    );
    const resumeOrder = resumeMock.mock.invocationCallOrder[0];
    const promptOrder = promptMock.mock.invocationCallOrder[0];
    expect(resumeOrder).toBeDefined();
    expect(promptOrder).toBeDefined();
    expect(resumeOrder!).toBeLessThan(promptOrder!);
  });
});

describe('PromptService.startBtw', () => {
  it('starts a side-channel agent through core RPC', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    await expect(impl.startBtw(SID)).resolves.toBe('agent_btw');

    expect(record.startBtwCalls).toEqual([{ sessionId: SID, agentId: 'main' }]);
  });

  it('rejects a late side-channel result from a closed session lifecycle', async () => {
    const retiredStartBtwReachedCore = createControlledPromise<void>();
    const releaseRetiredStartBtw = createControlledPromise<string>();
    const { bridge, record } = makeBridge();
    let startCalls = 0;
    vi.mocked(bridge.rpc.startBtw).mockImplementation(async (payload) => {
      startCalls += 1;
      record.startBtwCalls.push(payload);
      if (startCalls === 1) {
        retiredStartBtwReachedCore.resolve();
        return releaseRetiredStartBtw;
      }
      return 'agent_replacement';
    });
    const { bus } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());

    const retiredResult = impl.startBtw(SID);
    const retiredRejected = expect(retiredResult).rejects.toThrow('Session closed');
    await retiredStartBtwReachedCore;
    triggerClose(SID);
    await expect(impl.startBtw(SID)).resolves.toBe('agent_replacement');
    releaseRetiredStartBtw.resolve('agent_retired');
    await retiredRejected;

    expect(record.startBtwCalls).toEqual([
      { sessionId: SID, agentId: 'main' },
      { sessionId: SID, agentId: 'main' },
    ]);
  });
});

describe('PromptService lifecycle synthesis (via IEventService.onDidPublish)', () => {
  it('captures turnId on the first turn.started after submit', async () => {
    const { bridge } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 42,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(impl._activeForTest(SID)?.turnId).toBe(42);
  });

  it('ignores unrelated turn events while prompt admission is still pending', async () => {
    const promptReachedCore = createControlledPromise<void>();
    const admitPrompt = createControlledPromise<{
      readonly kind: 'started';
      readonly turnId: number;
    }>();
    const { bridge, record } = makeBridge();
    vi.mocked(bridge.rpc.prompt).mockImplementation(async (payload) => {
      record.promptCalls.push(payload);
      promptReachedCore.resolve();
      return admitPrompt;
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);

    const submitting = impl.submit(SID, mkBodyMinimal());
    await promptReachedCore;
    const promptId = (record.promptCalls[0] as { promptId: string }).promptId;

    triggerSubscribers({
      type: 'turn.started',
      turnId: 700,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 700,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.started',
      turnId: 701,
      origin: { kind: 'user' },
      promptId: 'prompt_unrelated',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    expect(impl._activeForTest(SID)?.turnId).toBeNull();
    expect(events.some((event) => event.type === 'prompt.completed')).toBe(false);

    triggerSubscribers({
      type: 'turn.started',
      turnId: 42,
      origin: { kind: 'user' },
      promptId,
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(impl._activeForTest(SID)?.turnId).toBe(42);

    admitPrompt.resolve({ kind: 'started', turnId: 42 });
    await submitting;
  });

  it('binds a deferred prompt only to its acknowledged deferred id', async () => {
    const { bridge } = makeBridge();
    vi.mocked(bridge.rpc.prompt).mockResolvedValue({
      kind: 'deferred',
      deferredPromptId: 'deferred_owned',
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const submitted = await impl.submit(SID, mkBodyMinimal());

    triggerSubscribers({
      type: 'turn.started',
      turnId: 70,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.started',
      turnId: 71,
      origin: { kind: 'user' },
      promptId: submitted.prompt_id,
      deferredPromptId: 'deferred_other',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(impl._activeForTest(SID)?.turnId).toBeNull();

    triggerSubscribers({
      type: 'turn.started',
      turnId: 72,
      origin: { kind: 'user' },
      promptId: submitted.prompt_id,
      deferredPromptId: 'deferred_owned',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(impl._activeForTest(SID)?.turnId).toBe(72);
  });

  it('aborts the current continuation turn after its turn id changes', async () => {
    const { bridge, record } = makeBridge({
      onGetGoal: () => goalResult('active'),
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const submitted = await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 42,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 42,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.started',
      turnId: 99,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await impl.abort(SID, submitted.prompt_id);

    expect(record.cancelCalls).toEqual([
      {
        sessionId: SID,
        agentId: 'main',
        turnId: 99,
        expectedPromptId: submitted.prompt_id,
        requireActive: true,
      },
    ]);
  });

  it('keeps the prompt active while a goal is between continuation turns', async () => {
    const terminalReadStarted = createControlledPromise<void>();
    const releaseTerminalRead = createControlledPromise<GoalToolResult>();
    let goalRead = 0;
    const { bridge, record } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 1) return goalResult('active');
        terminalReadStarted.resolve();
        return releaseTerminalRead;
      },
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const first = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );
    events.length = 0;

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await terminalReadStarted;

    const listed = await impl.list(SID);
    expect(listed.active?.prompt_id).toBe(first.prompt_id);
    expect(listed.queued.map((prompt) => prompt.prompt_id)).toEqual([second.prompt_id]);
    expect(record.promptCalls).toHaveLength(1);
    expect(events.some((event) => event.type === 'prompt.completed')).toBe(false);
    releaseTerminalRead.resolve(goalResult('active'));
    await releaseTerminalRead;
  });

  it('settles a terminal goal update without waiting for stale verification', async () => {
    const terminalReadStarted = createControlledPromise<void>();
    const releaseTerminalRead = createControlledPromise<GoalToolResult>();
    const completed = createControlledPromise<{ reason: string }>();
    const nextPromptStarted = createControlledPromise<void>();
    let goalRead = 0;
    const { bridge, record } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 1) return goalResult('active');
        if (goalRead === 2) {
          terminalReadStarted.resolve();
          return releaseTerminalRead;
        }
        return { goal: null };
      },
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    impl.onDidComplete((event) => {
      completed.resolve(event);
    });
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );
    events.length = 0;

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await terminalReadStarted;
    triggerSubscribers({
      type: 'goal.updated',
      snapshot: goalResult('blocked').goal,
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await Promise.all([completed, nextPromptStarted]);

    expect((await completed).reason).toBe('blocked');
    expect((await impl.list(SID)).active?.prompt_id).toBe(second.prompt_id);
    expect(record.promptCalls).toHaveLength(2);
    expect(events.filter((event) => event.type === 'prompt.completed')).toHaveLength(1);

    const staleVerification = (
      bridge.rpc.getGoal as unknown as ReturnType<typeof vi.fn>
    ).mock.results[1]?.value as Promise<GoalToolResult>;
    releaseTerminalRead.resolve(goalResult('active'));
    await staleVerification;
    await Promise.resolve();

    expect((await impl.list(SID)).active?.prompt_id).toBe(second.prompt_id);
    expect(events.filter((event) => event.type === 'prompt.completed')).toHaveLength(1);
  });

  it('settles the latest continuation when another turn ends during verification', async () => {
    const firstTerminalReadStarted = createControlledPromise<void>();
    const releaseFirstTerminalRead = createControlledPromise<GoalToolResult>();
    const completed = createControlledPromise<void>();
    const nextPromptStarted = createControlledPromise<void>();
    let goalRead = 0;
    const { bridge } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 1) return goalResult('active');
        if (goalRead === 2) {
          firstTerminalReadStarted.resolve();
          return releaseFirstTerminalRead;
        }
        return { goal: null };
      },
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    impl.onDidComplete(() => {
      completed.resolve();
    });
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );
    events.length = 0;

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await firstTerminalReadStarted;
    triggerSubscribers({
      type: 'turn.started',
      turnId: 2,
      origin: { kind: 'goal_continuation' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'goal.updated',
      snapshot: null,
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 2,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    releaseFirstTerminalRead.resolve({ goal: null });

    await Promise.all([completed, nextPromptStarted]);

    expect(events.filter((event) => event.type === 'prompt.completed')).toHaveLength(1);
  });

  it('completes a terminal prompt when the goal verification RPC fails', async () => {
    let goalRead = 0;
    const { bridge } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 1) return { goal: null };
        throw new Error('goal lookup failed');
      },
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(
      bridge,
      bus,
      makeAuth(),
      makeSessionService().sessionService,
      new ThrowingWarnLogService(),
    );
    const completed = createControlledPromise<void>();
    impl.onDidComplete(() => {
      completed.resolve();
    });
    await impl.submit(SID, mkBodyMinimal());

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await completed;

    expect((await impl.list(SID)).active).toBeNull();
  });

  it('synthesizes prompt.completed on top-level turn.ended (reason=completed)', async () => {
    const { bridge } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const completed = createControlledPromise<void>();
    impl.onDidComplete(() => {
      completed.resolve();
    });
    const submit = await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 7,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 7,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await completed;
    expect(events).toHaveLength(1);
    const synth = events[0] as unknown as {
      type: string;
      promptId: string;
      reason: string;
    };
    expect(synth.type).toBe('prompt.completed');
    expect(synth.promptId).toBe(submit.prompt_id);
    expect(synth.reason).toBe('completed');
    expect(impl._activeForTest(SID)).toBeUndefined();
  });

  it('preserves blocked reason when synthesizing prompt.completed', async () => {
    const { bridge } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const completed = createControlledPromise<void>();
    impl.onDidComplete(() => {
      completed.resolve();
    });
    const submit = await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 7,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 7,
      reason: 'blocked',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await completed;
    expect(events).toHaveLength(1);
    const synth = events[0] as unknown as {
      type: string;
      promptId: string;
      reason: string;
    };
    expect(synth.type).toBe('prompt.completed');
    expect(synth.promptId).toBe(submit.prompt_id);
    expect(synth.reason).toBe('blocked');
    expect(impl._activeForTest(SID)).toBeUndefined();
  });

  it('fires onDidComplete listener before bus.publish', async () => {
    const { bridge } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const submit = await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 7,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    const handlerArgs: unknown[] = [];
    const handlerCalledBeforePublish: boolean[] = [];
    impl.onDidComplete((e) => {
      handlerArgs.push(e);
      handlerCalledBeforePublish.push(
        events.filter(
          (ev) => (ev as unknown as { type?: string }).type === 'prompt.completed',
        ).length === 0,
      );
    });
    const completed = createControlledPromise<void>();
    impl.onDidComplete(() => {
      completed.resolve();
    });
    events.length = 0;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 7,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await completed;
    expect(handlerArgs).toHaveLength(1);
    expect((handlerArgs[0] as { promptId: string }).promptId).toBe(submit.prompt_id);
    expect(handlerCalledBeforePublish[0]).toBe(true);
  });

  it('synthesizes prompt.aborted on top-level turn.ended (reason=cancelled)', async () => {
    const { bridge } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const aborted = createControlledPromise<void>();
    impl.onDidAbort(() => {
      aborted.resolve();
    });
    await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 8,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 8,
      reason: 'cancelled',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await aborted;
    expect(events).toHaveLength(1);
    expect((events[0] as unknown as { type: string }).type).toBe('prompt.aborted');
  });

  it('ignores nested turn.ended (different turnId) so prompt stays active', async () => {
    const { bridge } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 99,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(events).toEqual([]);
    expect(impl._activeForTest(SID)?.completed).toBe(false);
  });

  it('removes every active prompt and ignores pending verification after session close', async () => {
    const verificationStarted = createControlledPromise<void>();
    const releaseVerification = createControlledPromise<GoalToolResult>();
    let goalRead = 0;
    const { bridge } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 1) return goalResult('active');
        verificationStarted.resolve();
        return releaseVerification;
      },
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());
    await impl.submit(SID, mkBodyMinimal());
    await impl.submit(
      SID,
      mkBodyMinimal({
        agent_id: 'agent_btw',
        content: [{ type: 'text', text: 'side channel' }],
      }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 7,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 7,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await verificationStarted;
    const verificationRpc = (
      bridge.rpc.getGoal as unknown as ReturnType<typeof vi.fn>
    ).mock.results[1]?.value as Promise<GoalToolResult>;
    events.length = 0;

    triggerClose(SID);

    expect(impl._activeForTest(SID)).toBeUndefined();
    expect(impl._activeForTest(SID, 'agent_btw')).toBeUndefined();
    releaseVerification.resolve({ goal: null });
    await verificationRpc;
    expect(events).toEqual([]);
  });

  it('is a no-op for events on a session with no active prompt', async () => {
    const { bridge } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    newSvc(bridge, bus);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(events).toEqual([]);
  });
});

describe('PromptService.abort', () => {
  it('throws PromptNotFoundError when no active prompt for the session', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await expect(impl.abort(SID, 'prompt_xyz')).rejects.toBeInstanceOf(
      PromptNotFoundError,
    );
  });

  it('returns {aborted: true} and publishes prompt.aborted', async () => {
    const { bridge, record } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const submit = await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 5,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;
    const result = await impl.abort(SID, submit.prompt_id);
    expect(result.aborted).toBe(true);
    expect(record.cancelCalls).toHaveLength(1);
    expect(record.cancelCalls[0]).toEqual({
      sessionId: SID,
      agentId: 'main',
      turnId: 5,
      expectedPromptId: submit.prompt_id,
      requireActive: true,
    });
    expect(events).toHaveLength(1);
    expect((events[0] as unknown as { type: string }).type).toBe('prompt.aborted');
  });

  it('throws PromptAlreadyCompletedError on the second abort', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    const submit = await impl.submit(SID, mkBody());
    await impl.abort(SID, submit.prompt_id);
    await expect(impl.abort(SID, submit.prompt_id)).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
  });

  it('remembers more than one completed prompt for idempotent aborts', async () => {
    const firstCompleted = createControlledPromise<void>();
    const secondCompleted = createControlledPromise<void>();
    const { bridge } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    let completionCount = 0;
    impl.onDidComplete(() => {
      completionCount += 1;
      if (completionCount === 1) firstCompleted.resolve();
      if (completionCount === 2) secondCompleted.resolve();
    });

    const first = await impl.submit(SID, mkBodyMinimal());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await firstCompleted;

    const second = await impl.submit(SID, mkBodyMinimal());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 2,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 2,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await secondCompleted;

    await expect(impl.abort(SID, first.prompt_id)).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    await expect(impl.abort(SID, second.prompt_id)).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
  });

  it('does not publish a retired abort after the session closes', async () => {
    const cancelReachedCore = createControlledPromise<void>();
    const releaseCancel = createControlledPromise<void>();
    const { bridge } = makeBridge({
      onCancel: async () => {
        cancelReachedCore.resolve();
        await releaseCancel;
      },
    });
    const { bus, events } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());
    const retired = await impl.submit(SID, mkBodyMinimal());

    const abortResult = impl.abort(SID, retired.prompt_id);
    const abortRejected = expect(abortResult).rejects.toThrow('Session closed');
    await cancelReachedCore;
    triggerClose(SID);
    const replacement = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'replacement' }] }),
    );
    await abortRejected;
    releaseCancel.resolve();
    const coreCancel = vi.mocked(bridge.rpc.cancel).mock.results[0]?.value as Promise<void>;
    await coreCancel;

    expect(
      events.some((event) => {
        const payload = event as unknown as { type?: string; promptId?: string };
        return payload.type === 'prompt.aborted' && payload.promptId === retired.prompt_id;
      }),
    ).toBe(false);
    expect((await impl.list(SID)).active?.prompt_id).toBe(replacement.prompt_id);
  });

  it('does not dispatch a delayed abort into a replacement session', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());
    const retired = await impl.submit(SID, mkBodyMinimal());

    const abortResult = impl.abort(SID, retired.prompt_id);
    const abortRejected = expect(abortResult).rejects.toThrow('Session closed');
    triggerClose(SID);
    const replacement = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'replacement' }] }),
    );
    await abortRejected;
    await Promise.resolve();

    expect(record.cancelCalls).toHaveLength(0);
    expect((await impl.list(SID)).active?.prompt_id).toBe(replacement.prompt_id);
  });

  it('keeps new submissions queued until an in-flight abort finishes', async () => {
    const cancelStarted = createControlledPromise<void>();
    const releaseCancel = createControlledPromise<void>();
    const nextPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onCancel: async () => {
        cancelStarted.resolve();
        await releaseCancel;
      },
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const first = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 5,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;

    const abortResult = impl.abort(SID, first.prompt_id);
    await cancelStarted;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 5,
      reason: 'cancelled',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    const third = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'three' }] }),
    );

    const whileCancelling = await impl.list(SID);
    expect(third.status).toBe('queued');
    expect(whileCancelling.active?.prompt_id).toBe(first.prompt_id);
    expect(whileCancelling.queued.map((prompt) => prompt.prompt_id)).toEqual([
      second.prompt_id,
      third.prompt_id,
    ]);
    expect(record.promptCalls).toHaveLength(1);
    expect(events.filter((event) => event.type === 'prompt.aborted')).toHaveLength(0);

    releaseCancel.resolve();
    await abortResult;
    await nextPromptStarted;

    const afterCancel = await impl.list(SID);
    expect(afterCancel.active?.prompt_id).toBe(second.prompt_id);
    expect(afterCancel.queued.map((prompt) => prompt.prompt_id)).toEqual([
      third.prompt_id,
    ]);
    expect(events.filter((event) => event.type === 'prompt.aborted')).toHaveLength(1);
  });

  it('settles an observed terminal turn when the cancel RPC rejects', async () => {
    const cancelStarted = createControlledPromise<void>();
    const releaseCancel = createControlledPromise<void>();
    const nextGoalReadStarted = createControlledPromise<void>();
    const releaseNextGoalRead = createControlledPromise<GoalToolResult>();
    const aborted = createControlledPromise<void>();
    const nextPromptStarted = createControlledPromise<void>();
    let goalRead = 0;
    const { bridge } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 1) return goalResult('active');
        if (goalRead === 2) {
          nextGoalReadStarted.resolve();
          return releaseNextGoalRead;
        }
        return { goal: null };
      },
      onCancel: async () => {
        cancelStarted.resolve();
        await releaseCancel;
        throw new Error('cancel failed');
      },
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    impl.onDidAbort(() => {
      aborted.resolve();
    });
    const first = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 5,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    const abortResult = expect(impl.abort(SID, first.prompt_id)).rejects.toThrow(
      'cancel failed',
    );
    await cancelStarted;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 5,
      reason: 'cancelled',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'goal.updated',
      snapshot: goalResult('paused').goal,
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    releaseCancel.resolve();

    await abortResult;
    await nextGoalReadStarted;

    const listed = await impl.list(SID);
    expect(listed.active?.prompt_id).toBe(second.prompt_id);
    expect(listed.queued).toHaveLength(0);

    await aborted;
    releaseNextGoalRead.resolve({ goal: null });
    await nextPromptStarted;
  });

  it('aborts a queued prompt without calling core.rpc.cancel', async () => {
    const { bridge, record } = makeBridge();
    const { bus, events } = makeBus();
    const impl = newSvc(bridge, bus);

    const first = await impl.submit(SID, mkBody());
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'queued' }] }),
    );
    expect(second.status).toBe('queued');

    events.length = 0;
    const result = await impl.abort(SID, second.prompt_id);

    expect(result.aborted).toBe(true);
    expect(record.cancelCalls).toHaveLength(0);
    const listed = await impl.list(SID);
    expect(listed.active?.prompt_id).toBe(first.prompt_id);
    expect(listed.queued).toHaveLength(0);
    expect(events).toHaveLength(1);
    const ev = events[0] as unknown as { type: string; promptId?: string };
    expect(ev.type).toBe('prompt.aborted');
    expect(ev.promptId).toBe(second.prompt_id);
  });
});

describe('PromptService.getCurrentPromptId', () => {
  it('returns the active prompt id while running', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    const submit = await impl.submit(SID, mkBody());
    expect(impl.getCurrentPromptId(SID)).toBe(submit.prompt_id);
  });

  it('returns undefined when idle', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    expect(impl.getCurrentPromptId(SID)).toBeUndefined();
  });

  it('returns undefined after the prompt completes', async () => {
    const { bridge } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const completed = createControlledPromise<void>();
    impl.onDidComplete(() => {
      completed.resolve();
    });
    const submit = await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await completed;
    expect(impl.getCurrentPromptId(SID)).toBeUndefined();
  });
});

describe('PromptService.abortBySession', () => {
  it('prevents dispatch when abort wins during prompt startup', async () => {
    const goalReadStarted = createControlledPromise<void>();
    const releaseGoalRead = createControlledPromise<GoalToolResult>();
    const { bridge, record } = makeBridge({
      onGetGoal: () => {
        goalReadStarted.resolve();
        return releaseGoalRead;
      },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const submitResult = impl.submit(SID, mkBodyMinimal());
    await goalReadStarted;
    const submitRejected = expect(submitResult).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );

    await expect(impl.abortBySession(SID)).resolves.toEqual({ aborted: true });
    releaseGoalRead.resolve({ goal: null });
    await submitRejected;

    expect(record.promptCalls).toHaveLength(0);
    expect((await impl.list(SID)).active).toBeNull();
  });

  it('reserves authority before preflight so abort cannot be lost', async () => {
    const resumeStarted = createControlledPromise<void>();
    const releaseResume = createControlledPromise<void>();
    const nextPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    let resumeCalls = 0;
    vi.mocked(bridge.rpc.resumeSession).mockImplementation(async () => {
      resumeCalls += 1;
      if (resumeCalls === 1) {
        resumeStarted.resolve();
        await releaseResume;
      }
      return undefined as never;
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const firstSubmit = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'one' }] }),
    );
    const firstRejected = expect(firstSubmit).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    await resumeStarted;

    await expect(impl.abortBySession(SID)).resolves.toEqual({ aborted: true });
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'two' }] }),
    );

    expect(second.status).toBe('queued');
    expect((await impl.list(SID)).active).toBeNull();
    expect((await impl.list(SID)).queued.map((item) => item.prompt_id)).toEqual([
      second.prompt_id,
    ]);
    expect(record.promptCalls).toHaveLength(0);

    releaseResume.resolve();
    await firstRejected;
    await nextPromptStarted;

    expect((await impl.list(SID)).active?.prompt_id).toBe(second.prompt_id);
    expect(record.promptCalls).toHaveLength(1);
  });

  it('prevents a promoted follower from dispatching when stop arrives during its preflight', async () => {
    const followerResumeStarted = createControlledPromise<void>();
    const releaseFollowerResume = createControlledPromise<void>();
    const activeCompleted = createControlledPromise<void>();
    const { bridge, record } = makeBridge();
    let resumeCalls = 0;
    vi.mocked(bridge.rpc.resumeSession).mockImplementation(async () => {
      resumeCalls += 1;
      if (resumeCalls === 2) {
        followerResumeStarted.resolve();
        await releaseFollowerResume;
      }
      return undefined as never;
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const active = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    impl.onDidComplete((event) => {
      if (event.promptId === active.prompt_id) activeCompleted.resolve();
    });
    const followerResult = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'pending follower' }] }),
    );
    const followerRejected = expect(followerResult).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    await followerResumeStarted;

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await activeCompleted;

    await expect(impl.abortBySession(SID)).resolves.toEqual({ aborted: true });
    await followerRejected;
    releaseFollowerResume.resolve();
    const rawResume = vi.mocked(bridge.rpc.resumeSession).mock.results[1]?.value as Promise<void>;
    await rawResume;

    expect(record.promptCalls).toHaveLength(1);
    expect(await impl.list(SID)).toMatchObject({ active: null, queued: [] });
  });

  it('records a cancelled startup mutation before compensating for the next prompt', async () => {
    const setterStarted = createControlledPromise<void>();
    const releaseSetter = createControlledPromise<void>();
    const nextPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'two') nextPromptStarted.resolve();
      },
    });
    vi.mocked(bridge.rpc.setModel).mockImplementation(async (payload) => {
      record.setModelCalls.push(payload);
      setterStarted.resolve();
      await releaseSetter;
      return { model: (payload as { model: string }).model };
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const firstSubmit = impl.submit(
      SID,
      mkBody({
        content: [{ type: 'text', text: 'one' }],
        model: 'kimi-code/k1',
        thinking: 'high',
        permission_mode: 'yolo',
        goal_objective: 'Do not create this after cancellation',
      }),
    );
    const firstRejected = expect(firstSubmit).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    await setterStarted;

    await expect(impl.abortBySession(SID)).resolves.toEqual({ aborted: true });
    const second = await impl.submit(
      SID,
      mkBody({
        content: [{ type: 'text', text: 'two' }],
        model: 'kimi-code/k2',
      }),
    );
    expect(second.status).toBe('queued');
    expect(record.promptCalls).toHaveLength(0);

    releaseSetter.resolve();
    await firstRejected;
    await nextPromptStarted;

    expect(record.setModelCalls).toEqual([
      { sessionId: SID, agentId: 'main', model: 'kimi-code/k1' },
      { sessionId: SID, agentId: 'main', model: 'kimi-code/k2' },
    ]);
    expect(record.setThinkingCalls).toHaveLength(0);
    expect(record.setPermissionCalls).toHaveLength(0);
    expect(record.createGoalCalls).toHaveLength(0);
    expect(impl._dispatchLogForTest(SID)?.map((entry) => entry.kind)).toEqual([
      'setModel',
      'setModel',
    ]);
    expect(impl.getAgentStateSnapshot(SID)?.model).toBe('kimi-code/k2');
    expect((await impl.list(SID)).active?.prompt_id).toBe(second.prompt_id);
  });

  it('joins concurrent stops onto one rejected cancel operation', async () => {
    const cancelStarted = createControlledPromise<void>();
    const releaseCancel = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onCancel: async () => {
        cancelStarted.resolve();
        await releaseCancel;
      },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBodyMinimal());

    const firstStop = impl.abortBySession(SID);
    await cancelStarted;
    const secondStop = impl.abortBySession(SID);
    releaseCancel.reject(new Error('shared cancel failed'));
    const outcomes = await Promise.allSettled([firstStop, secondStop]);

    expect(outcomes).toEqual([
      { status: 'rejected', reason: expect.objectContaining({ message: 'shared cancel failed' }) },
      { status: 'rejected', reason: expect.objectContaining({ message: 'shared cancel failed' }) },
    ]);
    expect(record.cancelCalls).toHaveLength(1);
  });

  it('retires a failed active prompt when prompt rejection precedes cancel rejection', async () => {
    const activePromptStarted = createControlledPromise<void>();
    const rejectActivePrompt = createControlledPromise<void>();
    const cancelStarted = createControlledPromise<void>();
    const rejectCancel = createControlledPromise<void>();
    const followerPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: async (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'active') {
          activePromptStarted.resolve();
          await rejectActivePrompt;
        }
        if (text === 'follower') followerPromptStarted.resolve();
      },
      onCancel: async () => {
        cancelStarted.resolve();
        await rejectCancel;
      },
    });
    const { bus, events } = makeBus();
    const impl = newSvc(bridge, bus);
    const activeResult = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    await activePromptStarted;
    const activeId = events.find((event) => event.type === 'prompt.submitted') as
      | { promptId: string }
      | undefined;
    const follower = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'follower' }] }),
    );

    const stopResult = impl.abortBySession(SID);
    const activeRejected = expect(activeResult).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    const stopRejected = expect(stopResult).rejects.toThrow('cancel failed after prompt failed');
    await cancelStarted;
    const corePromptRequest = vi.mocked(bridge.rpc.prompt).mock.results[0]?.value as Promise<void>;
    const corePromptRejected = expect(corePromptRequest).rejects.toThrow('prompt failed first');
    rejectActivePrompt.reject(new Error('prompt failed first'));
    await corePromptRejected;
    rejectCancel.reject(new Error('cancel failed after prompt failed'));

    await Promise.all([activeRejected, stopRejected, followerPromptStarted]);

    expect((await impl.list(SID)).active?.prompt_id).toBe(follower.prompt_id);
    expect(
      events.filter((event) => {
        const candidate = event as unknown as {
          type?: string;
          promptId?: string;
          reason?: string;
        };
        return (
          candidate.type === 'prompt.completed' &&
          candidate.promptId === activeId?.promptId &&
          candidate.reason === 'failed'
        );
      }),
    ).toHaveLength(1);
    expect(record.promptCalls).toHaveLength(2);
  });

  it('promotes the queue when cancel succeeds before any turn event or prompt acknowledgement', async () => {
    const activePromptStarted = createControlledPromise<void>();
    const neverAcknowledgeActive = createControlledPromise<void>();
    const followerPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: async (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'active') {
          activePromptStarted.resolve();
          await neverAcknowledgeActive;
        }
        if (text === 'follower') followerPromptStarted.resolve();
      },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    const activeResult = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    await activePromptStarted;
    const follower = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'follower' }] }),
    );
    const activeRejected = expect(activeResult).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    const activePromptId = impl.getCurrentPromptId(SID);

    await expect(impl.abortBySession(SID)).resolves.toEqual({ aborted: true });
    await Promise.all([activeRejected, followerPromptStarted]);

    expect((await impl.list(SID)).active?.prompt_id).toBe(follower.prompt_id);
    expect(record.cancelCalls).toEqual([
      {
        sessionId: SID,
        agentId: 'main',
        expectedPromptId: activePromptId,
        requireActive: true,
      },
    ]);
    expect(record.promptCalls).toHaveLength(2);
  });

  it('delegates to abort when a daemon prompt is active', async () => {
    const { bridge, record } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const submit = await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'turn.started',
      turnId: 7,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;

    const result = await impl.abortBySession(SID);

    expect(result.aborted).toBe(true);
    expect(record.cancelCalls).toHaveLength(1);
    expect(record.cancelCalls[0]).toEqual({
      sessionId: SID,
      agentId: 'main',
      turnId: 7,
      expectedPromptId: submit.prompt_id,
      requireActive: true,
    });
    expect(events).toHaveLength(1);
    expect((events[0] as unknown as { type: string }).type).toBe('prompt.aborted');
  });

  it('calls core.rpc.cancel without turnId when no daemon prompt is active', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const result = await impl.abortBySession(SID);

    expect(result.aborted).toBe(true);
    expect(record.cancelCalls).toHaveLength(1);
    expect(record.cancelCalls[0]).toEqual({
      sessionId: SID,
      agentId: 'main',
    });
  });

  it('does not let an idle stop cross into a replacement session lifecycle', async () => {
    const firstListReachedCore = createControlledPromise<void>();
    const releaseFirstList = createControlledPromise<SessionSummary[]>();
    const { bridge, record } = makeBridge();
    let listCalls = 0;
    vi.mocked(bridge.rpc.listSessions).mockImplementation(async () => {
      listCalls += 1;
      if (listCalls === 1) {
        firstListReachedCore.resolve();
        return releaseFirstList;
      }
      return [mkSummary()];
    });
    const { bus } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());

    const stopping = impl.abortBySession(SID);
    const stopRejected = expect(stopping).rejects.toThrow('Session closed');
    await firstListReachedCore;
    triggerClose(SID);
    const replacement = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'replacement' }] }),
    );
    await stopRejected;
    releaseFirstList.resolve([mkSummary()]);
    const retiredList = vi.mocked(bridge.rpc.listSessions).mock.results[0]
      ?.value as Promise<SessionSummary[]>;
    await retiredList;

    expect(record.cancelCalls).toHaveLength(0);
    expect((await impl.list(SID)).active?.prompt_id).toBe(replacement.prompt_id);
  });
});

describe('PromptService queue steer', () => {
  it('does not steer a queued prompt before the active turn is admitted', async () => {
    const goalReadStarted = createControlledPromise<void>();
    const releaseGoalRead = createControlledPromise<GoalToolResult>();
    const { bridge, record } = makeBridge({
      onGetGoal: () => {
        goalReadStarted.resolve();
        return releaseGoalRead;
      },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const activeResult = impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    await goalReadStarted;
    const queued = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'queued' }] }),
    );

    await expect(impl.steer(SID, [queued.prompt_id])).rejects.toBeInstanceOf(
      SessionBusyError,
    );
    expect(record.steerCalls).toHaveLength(0);
    expect((await impl.list(SID)).queued.map((prompt) => prompt.prompt_id)).toEqual([
      queued.prompt_id,
    ]);

    releaseGoalRead.resolve({ goal: null });
    await activeResult;
  });

  it('steers a queued prompt into the active turn without starting a new prompt', async () => {
    const { bridge, record } = makeBridge();
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const active = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }));
    const queued = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'queued' }] }));
    let reentrantAbort: Promise<unknown> | undefined;
    bus.onDidPublish((event) => {
      if (event.type !== 'prompt.steered' || reentrantAbort !== undefined) return;
      reentrantAbort = impl.abort(SID, queued.prompt_id);
      void reentrantAbort.catch(() => undefined);
    });
    triggerSubscribers({
      type: 'turn.started',
      turnId: 11,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    const result = await impl.steer(SID, [queued.prompt_id]);

    expect(result).toEqual({ steered: true, prompt_ids: [queued.prompt_id] });
    await expect(reentrantAbort).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    expect(record.promptCalls).toHaveLength(1);
    expect(record.cancelCalls).toHaveLength(0);
    expect(record.steerCalls).toEqual([
      {
        sessionId: SID,
        agentId: 'main',
        input: [{ type: 'text', text: 'queued' }],
        expectedPromptId: active.prompt_id,
        requireActive: true,
      },
    ]);
    expect((await impl.list(SID)).queued).toHaveLength(0);
    expect(
      events.some((event) => {
        const payload = event as unknown as {
          type?: string;
          activePromptId?: string;
          promptIds?: readonly string[];
        };
        return (
          payload.type === 'prompt.steered' &&
          payload.activePromptId === active.prompt_id &&
          payload.promptIds?.[0] === queued.prompt_id
        );
      }),
    ).toBe(true);
  });

  it('joins multiple queued prompts in admission order', async () => {
    const { bridge, record } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const active = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }));
    const first = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'first' }] }));
    const second = await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'second' }] }));
    triggerSubscribers({
      type: 'turn.started',
      turnId: 12,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    const result = await impl.steer(SID, [second.prompt_id, first.prompt_id]);

    expect(result.prompt_ids).toEqual([first.prompt_id, second.prompt_id]);
    expect(record.steerCalls).toEqual([
      {
        sessionId: SID,
        agentId: 'main',
        input: [{ type: 'text', text: 'first\n\nsecond' }],
        expectedPromptId: active.prompt_id,
        requireActive: true,
      },
    ]);
    expect((await impl.list(SID)).queued).toHaveLength(0);
  });

  it('keeps queued prompts when core steer fails', async () => {
    const { bridge } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    vi.mocked(bridge.rpc.steer).mockRejectedValueOnce(new Error('steer failed'));
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }));
    const queued = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'queued' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 13,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await expect(impl.steer(SID, [queued.prompt_id])).rejects.toThrow('steer failed');

    expect((await impl.list(SID)).queued.map((prompt) => prompt.prompt_id)).toEqual([
      queued.prompt_id,
    ]);
  });

  it('retains a failed deduplicated steer in admission order', async () => {
    const steerReachedCore = createControlledPromise<void>();
    const rejectSteer = createControlledPromise<void>();
    const { bridge, record } = makeBridge();
    vi.mocked(bridge.rpc.steer).mockImplementation(async (payload) => {
      record.steerCalls.push(payload);
      steerReachedCore.resolve();
      await rejectSteer;
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const active = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    const first = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'first' }] }),
    );
    const selected = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'selected' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 14,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    const steering = impl.steer(SID, [selected.prompt_id, selected.prompt_id]);
    const steerRejected = expect(steering).rejects.toThrow('steer failed');
    await steerReachedCore;
    const later = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'later' }] }),
    );
    rejectSteer.reject(new Error('steer failed'));
    await steerRejected;

    expect(record.steerCalls).toEqual([
      {
        sessionId: SID,
        agentId: 'main',
        input: [{ type: 'text', text: 'selected' }],
        expectedPromptId: active.prompt_id,
        requireActive: true,
      },
    ]);
    expect((await impl.list(SID)).queued.map((prompt) => prompt.prompt_id)).toEqual([
      first.prompt_id,
      selected.prompt_id,
      later.prompt_id,
    ]);
  });

  it('does not promote a later prompt past a pending failed steer', async () => {
    const steerReachedCore = createControlledPromise<void>();
    const rejectSteer = createControlledPromise<void>();
    const activeCompleted = createControlledPromise<void>();
    const selectedPromptStarted = createControlledPromise<void>();
    let completedWhileSteering = false;
    const { bridge, record } = makeBridge({
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'selected') selectedPromptStarted.resolve();
      },
    });
    vi.mocked(bridge.rpc.steer).mockImplementation(async (payload) => {
      record.steerCalls.push(payload);
      steerReachedCore.resolve();
      await rejectSteer;
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    impl.onDidComplete(() => {
      completedWhileSteering = true;
      activeCompleted.resolve();
    });
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    const selected = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'selected' }] }),
    );
    const later = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'later' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 16,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    const steering = impl.steer(SID, [selected.prompt_id]);
    const steerRejected = expect(steering).rejects.toThrow('steer failed');
    await steerReachedCore;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 16,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await Promise.resolve();
    await Promise.resolve();

    expect(record.promptCalls).toHaveLength(1);
    expect(completedWhileSteering).toBe(false);
    rejectSteer.reject(new Error('steer failed'));
    await Promise.all([steerRejected, activeCompleted, selectedPromptStarted]);

    expect((await impl.list(SID)).active?.prompt_id).toBe(selected.prompt_id);
    expect((await impl.list(SID)).queued.map((prompt) => prompt.prompt_id)).toEqual([
      later.prompt_id,
    ]);
    expect(
      record.promptCalls.map(
        (payload) => (payload as { input: Array<{ text?: string }> }).input[0]?.text,
      ),
    ).toEqual(['active', 'selected']);
  });

  it('stops the active transaction when abort targets a pending steer selection', async () => {
    const steerReachedCore = createControlledPromise<void>();
    const releaseSteer = createControlledPromise<void>();
    const followerPromptStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'follower') followerPromptStarted.resolve();
      },
    });
    vi.mocked(bridge.rpc.steer).mockImplementation(async () => {
      steerReachedCore.resolve();
      await releaseSteer;
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    let reentrantSelectedAbort: Promise<unknown> | undefined;
    const active = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    const selected = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'selected' }] }),
    );
    impl.onDidAbort((event) => {
      if (event.promptId !== selected.prompt_id || reentrantSelectedAbort !== undefined) {
        return;
      }
      reentrantSelectedAbort = impl.abort(SID, selected.prompt_id);
      void reentrantSelectedAbort.catch(() => undefined);
    });
    const follower = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'follower' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 17,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;

    const steering = impl.steer(SID, [selected.prompt_id]);
    const steerRejected = expect(steering).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    await steerReachedCore;
    await expect(impl.abort(SID, selected.prompt_id)).resolves.toEqual({ aborted: true });
    await Promise.all([steerRejected, followerPromptStarted]);
    await expect(reentrantSelectedAbort).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );

    releaseSteer.resolve();
    await Promise.resolve();
    expect(events.some((event) => event.type === 'prompt.steered')).toBe(false);
    expect(
      events
        .filter((event) => event.type === 'prompt.aborted')
        .map((event) => (event as unknown as { promptId: string }).promptId),
    ).toEqual([selected.prompt_id, active.prompt_id]);
    expect((await impl.list(SID)).active?.prompt_id).toBe(follower.prompt_id);
    expect(
      record.promptCalls.map(
        (payload) =>
          (payload as { input: Array<{ text?: string }> }).input[0]?.text,
      ),
    ).toEqual(['active', 'follower']);
  });

  it('reports a second steer as busy while the first steer is pending', async () => {
    const steerReachedCore = createControlledPromise<void>();
    const releaseSteer = createControlledPromise<void>();
    const { bridge } = makeBridge();
    vi.mocked(bridge.rpc.steer).mockImplementation(async () => {
      steerReachedCore.resolve();
      await releaseSteer;
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBodyMinimal());
    const first = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'first' }] }),
    );
    const second = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'second' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 18,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    const steering = impl.steer(SID, [first.prompt_id]);
    await steerReachedCore;
    await expect(impl.steer(SID, [second.prompt_id])).rejects.toBeInstanceOf(
      SessionBusyError,
    );

    releaseSteer.resolve();
    await steering;
  });

  it('cancels an active prompt without waiting for a hung steer acknowledgement', async () => {
    const steerReachedCore = createControlledPromise<void>();
    const releaseSteer = createControlledPromise<void>();
    const cancelReachedCore = createControlledPromise<void>();
    const followerPromptStarted = createControlledPromise<void>();
    let cancelCalled = false;
    const { bridge, record } = makeBridge({
      onCancel: () => {
        cancelCalled = true;
        cancelReachedCore.resolve();
      },
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'follower') followerPromptStarted.resolve();
      },
    });
    vi.mocked(bridge.rpc.steer).mockImplementation(async () => {
      steerReachedCore.resolve();
      await releaseSteer;
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    const active = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    const selected = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'selected' }] }),
    );
    const follower = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'follower' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 19,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;

    const steering = impl.steer(SID, [selected.prompt_id]);
    const steerRejected = expect(steering).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    await steerReachedCore;
    const aborting = impl.abort(SID, active.prompt_id);
    await cancelReachedCore;
    expect(cancelCalled).toBe(true);

    await Promise.all([steerRejected, aborting, followerPromptStarted]);

    // The transport mock deliberately ignores its signal. Resolving it later
    // must not publish a late steer or consume the selected queue item.
    releaseSteer.resolve();
    await Promise.resolve();

    expect(events.some((event) => event.type === 'prompt.steered')).toBe(false);
    expect(
      events
        .filter((event) => event.type === 'prompt.aborted')
        .map((event) => (event as unknown as { promptId: string }).promptId),
    ).toEqual([selected.prompt_id, active.prompt_id]);
    expect((await impl.list(SID)).active?.prompt_id).toBe(follower.prompt_id);
    expect((await impl.list(SID)).queued).toHaveLength(0);
    expect(
      record.promptCalls.map(
        (payload) =>
          (payload as { input: Array<{ text?: string }> }).input[0]?.text,
      ),
    ).toEqual(['active', 'follower']);
  });

  it('publishes a successful steer before terminal completion without re-reading goal state', async () => {
    const steerReachedCore = createControlledPromise<void>();
    const releaseSteer = createControlledPromise<void>();
    const terminalGoalRead = createControlledPromise<void>();
    const completed = createControlledPromise<void>();
    const followerPromptStarted = createControlledPromise<void>();
    let goalRead = 0;
    const { bridge, record } = makeBridge({
      onGetGoal: () => {
        goalRead += 1;
        if (goalRead === 2) terminalGoalRead.resolve();
        // A redundant third read would observe an unrelated newly-created
        // goal and incorrectly pin the already-terminal active prompt.
        return goalRead >= 3 ? goalResult('active') : { goal: null };
      },
      onPrompt: (payload) => {
        const text = (payload as { input: Array<{ text?: string }> }).input[0]?.text;
        if (text === 'follower') followerPromptStarted.resolve();
      },
    });
    vi.mocked(bridge.rpc.steer).mockImplementation(async (payload) => {
      record.steerCalls.push(payload);
      steerReachedCore.resolve();
      await releaseSteer;
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    let getGoalCallsAtCompletion = -1;
    impl.onDidComplete(() => {
      getGoalCallsAtCompletion = record.getGoalCalls.length;
      completed.resolve();
    });
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    const selected = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'selected' }] }),
    );
    const follower = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'follower' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 20,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;

    const steering = impl.steer(SID, [selected.prompt_id]);
    await steerReachedCore;
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 20,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    await terminalGoalRead;
    await Promise.resolve();
    expect(events.some((event) => event.type === 'prompt.completed')).toBe(false);

    releaseSteer.resolve();
    await Promise.all([steering, completed, followerPromptStarted]);

    expect(
      events
        .map((event) => event.type)
        .filter((type) => type === 'prompt.steered' || type === 'prompt.completed'),
    ).toEqual(['prompt.steered', 'prompt.completed']);
    expect(getGoalCallsAtCompletion).toBe(2);
    expect((await impl.list(SID)).active?.prompt_id).toBe(follower.prompt_id);
  });

  it('does not resurrect a pending steer after service disposal', async () => {
    const steerReachedCore = createControlledPromise<void>();
    const releaseSteer = createControlledPromise<void>();
    const { bridge, record } = makeBridge();
    vi.mocked(bridge.rpc.steer).mockImplementation(async (payload) => {
      record.steerCalls.push(payload);
      steerReachedCore.resolve();
      await releaseSteer;
    });
    const { bus, events, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }),
    );
    const selected = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'selected' }] }),
    );
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'follower' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 21,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    events.length = 0;

    const steering = impl.steer(SID, [selected.prompt_id]);
    const steerRejected = expect(steering).rejects.toMatchObject({
      name: 'AbortError',
    });
    await steerReachedCore;
    impl.dispose();
    await steerRejected;

    releaseSteer.resolve();
    await Promise.resolve();
    await Promise.resolve();
    expect(record.promptCalls).toHaveLength(1);
    expect(events.some((event) => event.type === 'prompt.steered')).toBe(false);
  });

  it('does not restore a failed steer into a replacement session lifecycle', async () => {
    const steerStarted = createControlledPromise<void>();
    const rejectSteer = createControlledPromise<void>();
    const { bridge, record } = makeBridge();
    vi.mocked(bridge.rpc.steer).mockImplementation(async (payload) => {
      record.steerCalls.push(payload);
      steerStarted.resolve();
      await rejectSteer;
    });
    const { bus, triggerSubscribers } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());
    await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'retired active' }] }),
    );
    const retiredQueued = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'retired queued' }] }),
    );
    triggerSubscribers({
      type: 'turn.started',
      turnId: 15,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    const steerResult = impl.steer(SID, [retiredQueued.prompt_id]);
    const steerRejected = expect(steerResult).rejects.toThrow('Session closed');
    await steerStarted;
    triggerClose(SID);
    const replacement = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'replacement' }] }),
    );
    rejectSteer.reject(new Error('retired steer failed'));
    await steerRejected;

    expect(await impl.list(SID)).toMatchObject({
      active: { prompt_id: replacement.prompt_id },
      queued: [],
    });
    expect(
      record.promptCalls.map(
        (payload) => (payload as { input: Array<{ text?: string }> }).input[0]?.text,
      ),
    ).toEqual(['retired active', 'replacement']);
  });

  it('throws PromptNotFoundError when steering a prompt that is not queued', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'active' }] }));

    await expect(impl.steer(SID, ['prompt_missing'])).rejects.toBeInstanceOf(
      PromptNotFoundError,
    );
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Stateless per-request session controls (model / thinking / permission_mode /
// plan_mode)
// ─────────────────────────────────────────────────────────────────────────────

describe('PromptService stateless controls — bootstrap + shadow', () => {
  it('bootstraps shadow from getConfig/getPermission/getPlan on first submit', async () => {
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2', thinkingEffort: 'medium' },
      permission: { mode: 'yolo' },
      plan: { id: 'plan_abc', content: '', path: '/tmp/p' },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ thinking: 'medium', permission_mode: 'yolo', plan_mode: true }));
    const snap = impl._agentStateForTest(SID);
    expect(snap).toEqual({
      model: 'kimi-code/k2',
      thinking: 'medium',
      permissionMode: 'yolo',
      planMode: true,
      swarmMode: false,
    });
    // Getters fired exactly once each.
    expect(record.getConfigCalls).toBe(1);
    expect(record.getPermissionCalls).toBe(1);
    expect(record.getPlanCalls).toBe(1);
    expect(record.getSwarmModeCalls).toBe(1);
    // No setters fired because body matched the bootstrap snapshot.
    expect(record.setModelCalls).toEqual([]);
    expect(record.setThinkingCalls).toEqual([]);
    expect(record.setPermissionCalls).toEqual([]);
    expect(record.enterPlanCalls).toEqual([]);
    expect(record.cancelPlanCalls).toEqual([]);
  });

  it('does not re-bootstrap on subsequent submits in the same session', async () => {
    const { bridge, record } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody());
    // Complete the first prompt so the second can start.
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await impl.submit(SID, mkBody({ content: [{ type: 'text', text: 'again' }] }));
    expect(record.getConfigCalls).toBe(1);
    expect(record.getPermissionCalls).toBe(1);
    expect(record.getPlanCalls).toBe(1);
  });

  it('re-bootstraps after the session closes', async () => {
    const { bridge, record } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());
    await impl.submit(SID, mkBody());
    expect(record.getConfigCalls).toBe(1);
    // First prompt cleared on completion so the second submit isn't busy.
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    triggerClose(SID);
    expect(impl._agentStateForTest(SID)).toBeUndefined();

    await impl.submit(SID, mkBody({ content: [{ type: 'text', text: 'after-close' }] }));
    expect(record.getConfigCalls).toBe(2);
    expect(record.getPermissionCalls).toBe(2);
    expect(record.getPlanCalls).toBe(2);
  });

  it('does not revive a closed session from a late bootstrap result', async () => {
    const bootstrapStarted = createControlledPromise<void>();
    const releaseConfig = createControlledPromise<
      Awaited<ReturnType<CoreRPC['getConfig']>>
    >();
    const { bridge, record } = makeBridge();
    vi.mocked(bridge.rpc.getConfig).mockImplementation(async () => {
      record.getConfigCalls += 1;
      bootstrapStarted.resolve();
      return releaseConfig;
    });
    const { bus } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());

    const retiredSubmit = impl.submit(SID, mkBody({ model: 'kimi-code/k1' }));
    const retiredRejection = expect(retiredSubmit).rejects.toBeInstanceOf(
      PromptAlreadyCompletedError,
    );
    await bootstrapStarted;
    triggerClose(SID);

    const replacement = await impl.submit(
      SID,
      mkBodyMinimal({ content: [{ type: 'text', text: 'replacement' }] }),
    );
    releaseConfig.resolve({
      cwd: '/tmp/ws',
      modelCapabilities: {} as never,
      thinkingEffort: 'off',
      systemPrompt: '',
      modelAlias: 'kimi-code/k2',
    });
    await retiredRejection;

    expect(impl._agentStateForTest(SID)).toBeUndefined();
    expect(impl._dispatchLogForTest(SID)).toBeUndefined();
    expect(record.setModelCalls).toHaveLength(0);
    expect(record.promptCalls).toHaveLength(1);
    expect((await impl.list(SID)).active?.prompt_id).toBe(replacement.prompt_id);
  });
});

describe('PromptService stateless controls — diff dispatch', () => {
  it('issues setModel only when the body model differs from the shadow', async () => {
    const { bridge, record } = makeBridge({ config: { modelAlias: 'kimi-code/k2' } });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ model: 'kimi-code/k2' }));
    expect(record.setModelCalls).toEqual([]);

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await impl.submit(SID, mkBody({ model: 'kimi-code/k1' }));
    expect(record.setModelCalls).toEqual([
      { sessionId: SID, agentId: 'main', model: 'kimi-code/k1' },
    ]);
    expect(impl._agentStateForTest(SID)?.model).toBe('kimi-code/k1');
  });

  it('issues setThinking only when the body effort differs from the shadow', async () => {
    const { bridge, record } = makeBridge({ config: { thinkingEffort: 'off' } });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ thinking: 'off' }));
    expect(record.setThinkingCalls).toEqual([]);

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await impl.submit(SID, mkBody({ thinking: 'high' }));
    expect(record.setThinkingCalls).toEqual([
      { sessionId: SID, agentId: 'main', effort: 'high' },
    ]);
    expect(impl._agentStateForTest(SID)?.thinking).toBe('high');
  });

  it('issues setPermission only when the mode differs from the shadow', async () => {
    const { bridge, record } = makeBridge({ permission: { mode: 'manual' } });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ permission_mode: 'manual' }));
    expect(record.setPermissionCalls).toEqual([]);

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await impl.submit(SID, mkBody({ permission_mode: 'yolo' }));
    expect(record.setPermissionCalls).toEqual([
      { sessionId: SID, agentId: 'main', mode: 'yolo' },
    ]);
  });

  it('enters plan mode when plan_mode goes false→true', async () => {
    const { bridge, record } = makeBridge({ plan: null });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ plan_mode: true }));
    expect(record.enterPlanCalls).toEqual([
      { sessionId: SID, agentId: 'main' },
    ]);
    expect(record.cancelPlanCalls).toEqual([]);
    expect(impl._agentStateForTest(SID)?.planMode).toBe(true);
  });

  it('cancels plan mode when plan_mode goes true→false', async () => {
    const { bridge, record } = makeBridge({
      plan: { id: 'plan_xyz', content: '', path: '/tmp/p' },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ plan_mode: false }));
    expect(record.cancelPlanCalls).toEqual([
      { sessionId: SID, agentId: 'main' },
    ]);
    expect(record.enterPlanCalls).toEqual([]);
    expect(impl._agentStateForTest(SID)?.planMode).toBe(false);
  });

  it('no-ops on repeated identical submissions (no extra setter RPCs)', async () => {
    const { bridge, record } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);

    for (let i = 0; i < 3; i++) {
      await impl.submit(
        SID,
        mkBody({ content: [{ type: 'text', text: `t${i}` }] }),
      );
      // Run the turn to completion so the next iteration's submit isn't busy.
      triggerSubscribers({
        type: 'turn.started',
        turnId: i + 1,
        origin: { kind: 'user' },
        sessionId: SID,
        agentId: 'main',
      } as unknown as Event);
      triggerSubscribers({
        type: 'turn.ended',
        turnId: i + 1,
        reason: 'completed',
        sessionId: SID,
        agentId: 'main',
      } as unknown as Event);
    }
    expect(record.setModelCalls).toEqual([]);
    expect(record.setThinkingCalls).toEqual([]);
    expect(record.setPermissionCalls).toEqual([]);
    expect(record.enterPlanCalls).toEqual([]);
    expect(record.cancelPlanCalls).toEqual([]);
  });
});

describe('PromptService stateless controls — live shadow updates', () => {
  it('mirrors agent.status.updated into the shadow', async () => {
    const { bridge } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'agent.status.updated',
      model: 'kimi-code/k1',
      thinkingEffort: 'high',
      permission: 'yolo',
      planMode: true,
      swarmMode: true,
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(impl.getAgentStateSnapshot(SID)).toMatchObject({
      model: 'kimi-code/k1',
      thinking: 'high',
      permissionMode: 'yolo',
      planMode: true,
      swarmMode: true,
    });
  });

  it('ignores side-channel status updates when maintaining the main-agent shadow', async () => {
    const { bridge } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody());
    triggerSubscribers({
      type: 'agent.status.updated',
      model: 'side-channel/model',
      thinkingEffort: 'max',
      permission: 'yolo',
      planMode: true,
      swarmMode: true,
      sessionId: SID,
      agentId: 'agent_btw',
    } as unknown as Event);

    expect(impl.getAgentStateSnapshot(SID)).toEqual({
      model: 'kimi-code/k2',
      thinking: 'off',
      permissionMode: 'manual',
      planMode: false,
      swarmMode: false,
    });
  });

  it('shadow update suppresses diff dispatch when body matches the new state', async () => {
    const { bridge, record } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody());
    // Out-of-band mutation lands on the bus.
    triggerSubscribers({
      type: 'agent.status.updated',
      permission: 'yolo',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    record.setPermissionCalls.length = 0;
    await impl.submit(SID, mkBody({ permission_mode: 'yolo' }));
    expect(record.setPermissionCalls).toEqual([]);
  });

  it('is a no-op on agent.status.updated for sessions without a shadow yet', async () => {
    const { bridge } = makeBridge();
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    triggerSubscribers({
      type: 'agent.status.updated',
      model: 'kimi-code/k1',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(impl._agentStateForTest(SID)).toBeUndefined();
  });

  it('does not let a late setter acknowledgement overwrite a newer status event', async () => {
    const setterReachedCore = createControlledPromise<void>();
    const releaseSetter = createControlledPromise<{
      readonly model: string;
      readonly providerName?: string;
    }>();
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2' },
    });
    vi.mocked(bridge.rpc.setModel).mockImplementation(async (payload) => {
      record.setModelCalls.push(payload);
      setterReachedCore.resolve();
      return releaseSetter;
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);

    const applying = impl.applyAgentState(
      SID,
      { model: 'kimi-code/k1' },
      'meta',
    );
    await setterReachedCore;
    triggerSubscribers({
      type: 'agent.status.updated',
      model: 'kimi-code/k3',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    releaseSetter.resolve({ model: 'kimi-code/k1' });
    await applying;

    expect(impl.getAgentStateSnapshot(SID)?.model).toBe('kimi-code/k3');
    expect(record.setModelCalls).toEqual([
      { sessionId: SID, agentId: 'main', model: 'kimi-code/k1' },
    ]);
  });

  it('overlays status events that arrive during initial shadow bootstrap', async () => {
    const configReadStarted = createControlledPromise<void>();
    const releaseConfig = createControlledPromise<
      Awaited<ReturnType<CoreRPC['getConfig']>>
    >();
    const { bridge, record } = makeBridge();
    vi.mocked(bridge.rpc.getConfig).mockImplementation(async () => {
      record.getConfigCalls += 1;
      configReadStarted.resolve();
      return releaseConfig;
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);

    const applying = impl.applyAgentState(
      SID,
      { model: 'kimi-code/live' },
      'meta',
    );
    await configReadStarted;
    triggerSubscribers({
      type: 'agent.status.updated',
      model: 'kimi-code/live',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    releaseConfig.resolve({
      cwd: '/tmp/ws',
      modelCapabilities: {} as never,
      thinkingEffort: 'off',
      systemPrompt: '',
      modelAlias: 'kimi-code/k2',
    });
    await applying;

    expect(record.setModelCalls).toHaveLength(0);
    expect(impl.getAgentStateSnapshot(SID)?.model).toBe('kimi-code/live');
  });
});

describe('PromptService stateless controls — dispatch log', () => {
  it('is undefined before any submit and stays empty when the body matches bootstrap', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    expect(impl._dispatchLogForTest(SID)).toBeUndefined();
    // Default body matches the default bridge bootstrap (model=k2,
    // thinking=off, permission=manual, plan=null).
    await impl.submit(SID, mkBody());
    // No setter fired -> buffer never allocated -> still undefined.
    expect(impl._dispatchLogForTest(SID)).toBeUndefined();
  });

  it('appends one entry per setter dispatched, in the order setModel/setThinking/setPermission/(enter|cancel)Plan', async () => {
    const { bridge } = makeBridge({
      config: { modelAlias: 'kimi-code/k2', thinkingEffort: 'off' },
      permission: { mode: 'manual' },
      plan: null,
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(
      SID,
      mkBody({
        model: 'kimi-code/k1',
        thinking: 'high',
        permission_mode: 'yolo',
        plan_mode: true,
      }),
    );
    const log = impl._dispatchLogForTest(SID);
    expect(log).toBeDefined();
    const kinds = (log ?? []).map((e) => e.kind);
    expect(kinds).toEqual(['setModel', 'setThinking', 'setPermission', 'enterPlan']);
    expect(log?.[0]?.payload).toEqual({
      sessionId: SID,
      agentId: 'main',
      model: 'kimi-code/k1',
    });
    expect(log?.[3]?.payload).toEqual({ sessionId: SID, agentId: 'main' });
    // Every entry from a prompt-body override path is tagged source='prompt'.
    expect((log ?? []).every((e) => e.source === 'prompt')).toBe(true);
    // Each entry should be attributed to the prompt id returned by submit;
    // they all share the same id within a single submit.
    expect(new Set((log ?? []).map((e) => e.promptId)).size).toBe(1);
  });

  it('does NOT append entries when a repeat submit matches the shadow', async () => {
    const { bridge } = makeBridge({ plan: null });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    // First submit toggles plan_mode on -> 1 entry.
    await impl.submit(SID, mkBody({ plan_mode: true }));
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    expect(impl._dispatchLogForTest(SID)?.length).toBe(1);
    expect(impl._dispatchLogForTest(SID)?.[0]?.kind).toBe('enterPlan');

    // Second submit with the same plan_mode -> shadow suppresses dispatch.
    // This is the property scenario 04 cannot observe over WS frames alone.
    await impl.submit(SID, mkBody({ plan_mode: true }));
    expect(impl._dispatchLogForTest(SID)?.length).toBe(1);
  });

  it('clears the buffer when the session closes (re-bootstrap on next submit)', async () => {
    const { bridge } = makeBridge({ plan: null });
    const { bus } = makeBus();
    const { sessionService, triggerClose } = makeSessionService();
    const impl = new PromptService(bridge, bus, makeAuth(), sessionService, new NoopLogService());
    await impl.submit(SID, mkBody({ plan_mode: true }));
    expect(impl._dispatchLogForTest(SID)?.length).toBe(1);
    triggerClose(SID);
    expect(impl._dispatchLogForTest(SID)).toBeUndefined();
  });

  it('bootstraps swarmMode from getSwarmMode', async () => {
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2', thinkingEffort: 'off' },
      permission: { mode: 'manual' },
      plan: null,
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ model: 'kimi-code/k1' }));
    expect(record.getSwarmModeCalls).toBe(1);
  });

  it('dispatches enterSwarm/exitSwarm and records them in the log', async () => {
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2', thinkingEffort: 'off' },
      permission: { mode: 'manual' },
      plan: null,
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);

    await impl.submit(SID, mkBody({ swarm_mode: true }));
    expect(record.enterSwarmCalls.length).toBe(1);
    expect(record.enterSwarmCalls[0]).toEqual({
      sessionId: SID,
      agentId: 'main',
      trigger: 'manual',
    });
    let log = impl._dispatchLogForTest(SID);
    expect(log?.some((e) => e.kind === 'enterSwarm')).toBe(true);

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await impl.submit(SID, mkBody({ swarm_mode: false }));
    expect(record.exitSwarmCalls.length).toBe(1);
    expect(record.exitSwarmCalls[0]).toEqual({ sessionId: SID, agentId: 'main' });
    log = impl._dispatchLogForTest(SID);
    expect(log?.some((e) => e.kind === 'exitSwarm')).toBe(true);
  });

  it('does not re-dispatch swarm_mode when it matches the shadow', async () => {
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2', thinkingEffort: 'off' },
      permission: { mode: 'manual' },
      plan: null,
    });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ swarm_mode: true }));
    expect(record.enterSwarmCalls.length).toBe(1);

    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);

    await impl.submit(SID, mkBody({ swarm_mode: true }));
    expect(record.enterSwarmCalls.length).toBe(1);
  });

  it('dispatches createGoal and records it in the log', async () => {
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2', thinkingEffort: 'off' },
      permission: { mode: 'manual' },
      plan: null,
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBody({ goal_objective: 'Refactor the auth module' }));
    expect(record.createGoalCalls.length).toBe(1);
    expect(record.createGoalCalls[0]).toEqual({
      sessionId: SID,
      agentId: 'main',
      objective: 'Refactor the auth module',
      replace: false,
    });
    const log = impl._dispatchLogForTest(SID);
    expect(log?.some((e) => e.kind === 'createGoal')).toBe(true);
  });

  it('dispatches goal control actions and records them in the log', async () => {
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2', thinkingEffort: 'off' },
      permission: { mode: 'manual' },
      plan: null,
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.applyAgentState(SID, { goal_control: 'pause' }, 'meta');
    expect(record.pauseGoalCalls.length).toBe(1);
    await impl.applyAgentState(SID, { goal_control: 'resume' }, 'meta');
    expect(record.resumeGoalCalls.length).toBe(1);
    await impl.applyAgentState(SID, { goal_control: 'cancel' }, 'meta');
    expect(record.cancelGoalCalls.length).toBe(1);
    const log = impl._dispatchLogForTest(SID);
    expect(log?.filter((e) => e.kind === 'pauseGoal').length).toBe(1);
    expect(log?.filter((e) => e.kind === 'resumeGoal').length).toBe(1);
    expect(log?.filter((e) => e.kind === 'cancelGoal').length).toBe(1);
  });
});

describe('PromptService stateful session — content-only path', () => {
  it('issues zero bootstrap RPCs and zero setters when the body carries no controls', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.submit(SID, mkBodyMinimal());
    // Bootstrap getters never ran (no body control to diff against).
    expect(record.getConfigCalls).toBe(0);
    expect(record.getPermissionCalls).toBe(0);
    expect(record.getPlanCalls).toBe(0);
    expect(record.getSwarmModeCalls).toBe(0);
    // No setters fired either.
    expect(record.setModelCalls).toEqual([]);
    expect(record.setThinkingCalls).toEqual([]);
    expect(record.setPermissionCalls).toEqual([]);
    expect(record.enterPlanCalls).toEqual([]);
    expect(record.cancelPlanCalls).toEqual([]);
    // Shadow stays absent — there's nothing to remember.
    expect(impl._agentStateForTest(SID)).toBeUndefined();
    // Dispatch log untouched.
    expect(impl._dispatchLogForTest(SID)).toBeUndefined();
    // The prompt itself fires through to bridge.prompt.
    expect(record.promptCalls).toHaveLength(1);
  });

  it('reuses the shadow established by a prior submit for subsequent content-only submits', async () => {
    const { bridge, record } = makeBridge({ config: { modelAlias: 'kimi-code/k2' } });
    const { bus, triggerSubscribers } = makeBus();
    const impl = newSvc(bridge, bus);
    // First submit carries an override → bootstrap + dispatch.
    await impl.submit(SID, mkBody({ model: 'kimi-code/k9' }));
    expect(record.setModelCalls).toHaveLength(1);
    triggerSubscribers({
      type: 'turn.started',
      turnId: 1,
      origin: { kind: 'user' },
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    triggerSubscribers({
      type: 'turn.ended',
      turnId: 1,
      reason: 'completed',
      sessionId: SID,
      agentId: 'main',
    } as unknown as Event);
    record.setModelCalls.length = 0;
    // Second submit content-only — uses the shadow, no setter re-fires.
    await impl.submit(SID, mkBodyMinimal({ content: [{ type: 'text', text: 'follow-up' }] }));
    expect(record.setModelCalls).toEqual([]);
    expect(impl._agentStateForTest(SID)?.model).toBe('kimi-code/k9');
  });
});

describe('PromptService.applyAgentState (POST /sessions/{sid}/profile path)', () => {
  it('throws SessionNotFoundError on unknown sid', async () => {
    const { bridge } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await expect(
      impl.applyAgentState('sess_missing', { model: 'kimi-code/k1' }, 'meta'),
    ).rejects.toBeInstanceOf(SessionNotFoundError);
  });

  it('is a no-op when the patch carries no fields (no bootstrap, no setter)', async () => {
    const { bridge, record } = makeBridge();
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.applyAgentState(SID, {}, 'meta');
    expect(record.getConfigCalls).toBe(0);
    expect(record.setModelCalls).toEqual([]);
    expect(impl._agentStateForTest(SID)).toBeUndefined();
  });

  it('dispatches setThinking and records source="meta" when patch differs from shadow', async () => {
    const { bridge, record } = makeBridge({ config: { thinkingEffort: 'off' } });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.applyAgentState(SID, { thinking: 'high' }, 'meta');
    expect(record.setThinkingCalls).toEqual([
      { sessionId: SID, agentId: 'main', effort: 'high' },
    ]);
    expect(impl._agentStateForTest(SID)?.thinking).toBe('high');
    const log = impl._dispatchLogForTest(SID);
    expect(log).toHaveLength(1);
    expect(log?.[0]?.source).toBe('meta');
    // No prompt minted → entry's promptId is the empty string.
    expect(log?.[0]?.promptId).toBe('');
  });

  it('subsequent content-only submit observes the shadow set via /profile and dispatches nothing', async () => {
    const { bridge, record } = makeBridge({
      config: { thinkingEffort: 'off' },
      permission: { mode: 'manual' },
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);
    await impl.applyAgentState(
      SID,
      { thinking: 'high', permission_mode: 'yolo' },
      'meta',
    );
    record.setThinkingCalls.length = 0;
    record.setPermissionCalls.length = 0;
    await impl.submit(SID, mkBodyMinimal());
    expect(record.setThinkingCalls).toEqual([]);
    expect(record.setPermissionCalls).toEqual([]);
    expect(impl._agentStateForTest(SID)).toMatchObject({
      thinking: 'high',
      permissionMode: 'yolo',
    });
  });

  it('serializes concurrent setters so the later profile patch owns the final shadow', async () => {
    const firstSetterStarted = createControlledPromise<void>();
    const releaseFirstSetter = createControlledPromise<void>();
    const secondSetterStarted = createControlledPromise<void>();
    const { bridge, record } = makeBridge({
      config: { modelAlias: 'kimi-code/k2' },
    });
    let setterCalls = 0;
    vi.mocked(bridge.rpc.setModel).mockImplementation(async (payload) => {
      setterCalls += 1;
      record.setModelCalls.push(payload);
      if (setterCalls === 1) {
        firstSetterStarted.resolve();
        await releaseFirstSetter;
      } else {
        secondSetterStarted.resolve();
      }
      return { model: (payload as { model: string }).model };
    });
    const { bus } = makeBus();
    const impl = newSvc(bridge, bus);

    const earlier = impl.applyAgentState(
      SID,
      { model: 'kimi-code/k1' },
      'meta',
    );
    await firstSetterStarted;
    const later = impl.applyAgentState(
      SID,
      { model: 'kimi-code/k3' },
      'meta',
    );

    expect(record.setModelCalls).toEqual([
      { sessionId: SID, agentId: 'main', model: 'kimi-code/k1' },
    ]);
    releaseFirstSetter.resolve();
    await Promise.all([earlier, later, secondSetterStarted]);

    expect(record.setModelCalls).toEqual([
      { sessionId: SID, agentId: 'main', model: 'kimi-code/k1' },
      { sessionId: SID, agentId: 'main', model: 'kimi-code/k3' },
    ]);
    expect(impl.getAgentStateSnapshot(SID)?.model).toBe('kimi-code/k3');
  });
});
