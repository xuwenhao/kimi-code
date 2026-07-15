/**
 * `PromptService` — implementation of `IPromptService`.
 */

import { Disposable, InstantiationType, registerSingleton } from '../../di';
import { Emitter } from '../../base/common/event';
import type {
  Event,
  PromptItem,
  PromptListResponse,
  PromptSubmission,
  PromptSteerResult,
  PromptSubmitResult,
  PromptThinking,
} from '@moonshot-ai/protocol';
import type { PermissionMode } from '../../agent/permission';
import { ulid } from 'ulid';

import { ICoreProcessService } from '../coreProcess/coreProcess';
import { IAuthSummaryService } from '../authSummary/authSummary';
import { IEventService } from '../event/event';
import { ILogService } from '../logger/logger';
import { ISessionService, SessionNotFoundError } from '../session/session';
import { abortable, abortError, userCancellationReason } from '../../utils/abort';
import { isKimiError } from '../../errors';
import {
  IPromptService,
  PromptNotFoundError,
  PromptAlreadyCompletedError,
  SessionBusyError,
  type AgentStatePatch,
  type AgentStateSnapshot,
  type AgentStateSource,
  type PromptAbortResult,
  type PromptDispatchLogEntry,
  type SyntheticPromptCompletedEvent,
  type SyntheticPromptAbortedEvent,
  type SyntheticPromptSteeredEvent,
  type SyntheticPromptSubmittedEvent,
} from './prompt';

const MAIN_AGENT_ID = 'main';

function promptKey(sessionId: string, agentId: string): string {
  return `${sessionId}\u0000${agentId}`;
}

/** Cap per-session dispatch-log entries; ring-buffer drops oldest on overflow. */
const DISPATCH_LOG_CAP = 100;
const TERMINAL_PROMPT_CAP = 100;

/**
 * `true` iff any of the runtime-control fields is defined on the patch.
 * Used to short-circuit `applyAgentState` / the prompt-body override path
 * when the caller carries nothing actionable.
 */
function hasAnyAgentStateField(patch: AgentStatePatch): boolean {
  return (
    patch.model !== undefined ||
    patch.thinking !== undefined ||
    patch.permission_mode !== undefined ||
    patch.plan_mode !== undefined ||
    patch.swarm_mode !== undefined ||
    patch.goal_objective !== undefined ||
    patch.goal_control !== undefined
  );
}

/**
 * Extract the runtime-control fields from a `PromptSubmission` body into a
 * shadow-shaped patch. Returns `undefined` when the body carries none of the
 * fields — the submit path skips both shadow bootstrap and diff-dispatch in
 * that case, saving RPCs on hot content-only prompts.
 */
function pickAgentStatePatch(body: PromptSubmission): AgentStatePatch | undefined {
  const patch: AgentStatePatch = {};
  if (body.model !== undefined) patch.model = body.model;
  if (body.thinking !== undefined) patch.thinking = body.thinking;
  if (body.permission_mode !== undefined) patch.permission_mode = body.permission_mode;
  if (body.plan_mode !== undefined) patch.plan_mode = body.plan_mode;
  if (body.swarm_mode !== undefined) patch.swarm_mode = body.swarm_mode;
  if (body.goal_objective !== undefined) patch.goal_objective = body.goal_objective;
  if (body.goal_control !== undefined) patch.goal_control = body.goal_control;
  return hasAnyAgentStateField(patch) ? patch : undefined;
}

/**
 * Per-session "active prompt" state. Cleared on completion/abort.
 *
 * `turnId === null` when the prompt has been submitted but the first
 * `turn.started` hasn't arrived yet (the RPC pair queues calls before
 * `ready()` so the gap is small but non-zero in practice).
 *
 * Terminal states are removed from the active map; the service separately
 * remembers the most recent terminal prompt id for idempotent aborts.
 */
interface PromptState {
  agentId: string;
  promptId: string;
  userMessageId: string;
  body: PromptSubmission;
  createdAt: string;
  sessionRevision: number;
  preflightPending: boolean;
  readyToStart: boolean;
  submitted: boolean;
  startupPending: boolean;
  startupFailed: boolean;
  promptDispatched: boolean;
  dispatchAcknowledged: boolean;
  acceptedTurnId: number | null;
  acceptedDeferredPromptId: string | null;
  dispatchController: AbortController;
  workerTerminal: boolean;
  workerTerminalBarrier: Promise<void>;
  resolveWorkerTerminal: () => void;
  turnId: number | null;
  turnObserved: boolean;
  turnRevision: number;
  goalStatus: string | null;
  goalActive: boolean;
  goalDriven: boolean;
  goalRevision: number;
  deferredOutcome: 'completed' | 'cancelled' | 'failed' | 'blocked' | null;
  terminalCheckPending: boolean;
  aborting: boolean;
  abortPromise: Promise<PromptAbortResult> | undefined;
  completed: boolean;
  aborted: boolean;
}

interface SteerReservation {
  readonly sessionRevision: number;
  readonly target: PromptState;
  readonly selected: readonly PromptState[];
  readonly controller: AbortController;
}

type CorePromptPart =
  | { readonly type: 'text'; readonly text: string }
  | { readonly type: 'image_url'; readonly imageUrl: { readonly url: string } }
  | { readonly type: 'video_url'; readonly videoUrl: { readonly url: string } };

function createWorkerTerminalBarrier(): {
  promise: Promise<void>;
  resolve: () => void;
} {
  let resolve!: () => void;
  const promise = new Promise<void>((settle) => {
    resolve = settle;
  });
  return { promise, resolve };
}

function toPromptItem(state: PromptState, status: 'running' | 'queued'): PromptItem {
  return {
    prompt_id: state.promptId,
    user_message_id: state.userMessageId,
    status,
    content: state.body.content,
    created_at: state.createdAt,
  };
}

function contentToCoreParts(content: PromptSubmission['content']): CorePromptPart[] {
  const input: CorePromptPart[] = [];
  for (const part of content) {
    switch (part.type) {
      case 'text':
        input.push({ type: 'text', text: part.text });
        break;
      case 'image':
        if (part.source.kind === 'url') {
          input.push({
            type: 'image_url',
            imageUrl: { url: part.source.url },
          });
        } else if (part.source.kind === 'base64') {
          input.push({
            type: 'image_url',
            imageUrl: {
              url: `data:${part.source.media_type};base64,${part.source.data}`,
            },
          });
        }
        break;
      case 'video':
        if (part.source.kind === 'url') {
          input.push({
            type: 'video_url',
            videoUrl: { url: part.source.url },
          });
        } else if (part.source.kind === 'base64') {
          input.push({
            type: 'video_url',
            videoUrl: {
              url: `data:${part.source.media_type};base64,${part.source.data}`,
            },
          });
        }
        break;
      case 'file':
      case 'thinking':
      case 'tool_result':
      case 'tool_use':
        break;
    }
  }
  return input;
}

function steerContentToCoreParts(states: readonly PromptState[]): CorePromptPart[] {
  const textBodies: string[] = [];
  let allText = true;
  for (const state of states) {
    const texts: string[] = [];
    for (const part of state.body.content) {
      if (part.type !== 'text') {
        allText = false;
        break;
      }
      texts.push(part.text);
    }
    if (!allText) break;
    textBodies.push(texts.join('\n'));
  }
  if (allText) {
    return [{ type: 'text', text: textBodies.join('\n\n') }];
  }

  const input: CorePromptPart[] = [];
  states.forEach((state, index) => {
    if (index > 0) input.push({ type: 'text', text: '\n\n' });
    input.push(...contentToCoreParts(state.body.content));
  });
  return input;
}

/**
 * Type guard for `turn.started` agent-core events.
 */
function isTurnStarted(e: Event): e is Event & {
  type: 'turn.started';
  turnId: number;
  promptId?: string;
  deferredPromptId?: string;
} {
  return (e as { type?: string }).type === 'turn.started';
}

/**
 * Type guard for `turn.ended` agent-core events.
 */
function isTurnEnded(e: Event): e is Event & {
  type: 'turn.ended';
  turnId: number;
  reason: 'completed' | 'cancelled' | 'failed' | 'blocked';
} {
  return (e as { type?: string }).type === 'turn.ended';
}

/**
 * Type guard for `agent.status.updated` agent-core events. Carries the
 * subset of fields we mirror into the per-session shadow on every live
 * change.
 */
function isAgentStatusUpdated(e: Event): e is Event & {
  type: 'agent.status.updated';
  model?: string;
  thinkingEffort?: string;
  permission?: PermissionMode;
  planMode?: boolean;
  swarmMode?: boolean;
} {
  return (e as { type?: string }).type === 'agent.status.updated';
}

function isGoalUpdated(e: Event): e is Event & {
  type: 'goal.updated';
  snapshot: null | { status: string };
} {
  return (e as { type?: string }).type === 'goal.updated';
}

/**
 * Per-session shadow of `model` / `thinking` / `permissionMode` /
 * `planMode`. Type re-exported from `./prompt` so the daemon debug route
 * can consume it without reaching into `PromptService` internals.
 * Absent until first `submit` bootstraps. See `_bootstrapAgentState` +
 * `_applyAgentState`.
 */

export class PromptService
  extends Disposable
  implements IPromptService
{
  readonly _serviceBrand: undefined;

  /** Active prompt per session. Cleared on completion / abort emission. */
  private readonly _active = new Map<string, PromptState>();

  private readonly _queued = new Map<string, PromptState[]>();

  private readonly _steerReservations = new Map<string, SteerReservation>();

  private readonly _terminalPromptIds = new Map<string, Set<string>>();

  /**
   * Monotonic lifecycle revision per session id. A close increments it before
   * clearing state, so async work from the retired lifecycle cannot repopulate
   * shadows or dispatch controls into a later session reusing the same id.
   */
  private readonly _sessionRevision = new Map<string, number>();

  /**
   * Session-scoped single-writer tails for runtime controls. The lifecycle
   * revision is part of the key so work from a closed session can never block
   * or overwrite a replacement session that reuses the same id.
   */
  private readonly _controlTails = new Map<string, Promise<void>>();

  private readonly _sessionLifecycleControllers = new Map<
    string,
    { revision: number; controller: AbortController }
  >();

  private _disposed = false;

  /**
   * Per-session shadow of `model` / `thinking` / `permissionMode` /
   * `planMode`. Absent until first `submit` bootstraps. See
   * `_bootstrapAgentState` + `_applyAgentState`.
   */
  private readonly _agentState = new Map<string, AgentStateSnapshot>();

  private readonly _agentStatusRevision = new Map<string, number>();

  private readonly _agentStatusFieldRevision = new Map<
    string,
    Partial<Record<keyof AgentStateSnapshot, number>>
  >();

  private readonly _latestAgentStatus = new Map<
    string,
    { revision: number; snapshot: AgentStateSnapshot }
  >();

  /**
   * Per-session ring buffer of stateless-control setter dispatches.
   * Each entry records `{ts, kind, payload, promptId}` immediately after
   * the underlying `core.rpc.*` setter resolves inside `_applyAgentState`.
   * The buffer is capped at `DISPATCH_LOG_CAP`; on overflow the oldest
   * entry is dropped. Cleared on `ISessionService.onDidClose` together
   * with the shadow. Exposed via `_dispatchLogForTest` for the daemon's
   * `/debug/prompts/{sid}/dispatch-log` route + unit tests — never read
   * on the hot path.
   */
  private readonly _dispatchLog = new Map<string, PromptDispatchLogEntry[]>();

  /**
   * VSCode-style Emitter for `prompt.completed` synthetic events. Listener
   * exceptions route to `onUnexpectedError` inside `Emitter.fire()`. Owned
   * via `_register(...)` so it disposes when PromptService is torn down.
   */
  private readonly _onDidComplete = this._register(
    new Emitter<SyntheticPromptCompletedEvent>(),
  );
  readonly onDidComplete = this._onDidComplete.event;
  /**
   * VSCode-style Emitter for `prompt.aborted` synthetic events. Same
   * ownership + exception-routing semantics as `_onDidComplete`.
   */
  private readonly _onDidAbort = this._register(
    new Emitter<SyntheticPromptAbortedEvent>(),
  );
  readonly onDidAbort = this._onDidAbort.event;

  constructor(
    @ICoreProcessService private readonly core: ICoreProcessService,
    @IEventService private readonly eventService: IEventService,
    @IAuthSummaryService private readonly auth: IAuthSummaryService,
    @ISessionService private readonly sessionService: ISessionService,
    @ILogService private readonly _logger: ILogService,
  ) {
    super();
    // Self-subscribe to the event stream for lifecycle synthesis.
    // `onDidPublish` is the VSCode-style accessor — calling it registers
    // `_handleBusEvent` and returns an `IDisposable` that detaches when
    // disposed. We register it through `this._register(...)` so the
    // listener tears down when PromptService disposes (which happens BEFORE
    // the event service disposes per start.ts wiring order). Re-entrance
    // is safe: synthesised `prompt.*` events don't match the `turn.*`
    // predicates below.
    this._register(
      this.eventService.onDidPublish(this._handleBusEvent.bind(this)),
    );
    // Drop the per-session shadow when a session closes so the next
    // submit for a freshly-recreated session re-bootstraps cleanly.
    this._register(
      this.sessionService.onDidClose(({ sessionId }) => {
        this._sessionLifecycleControllers
          .get(sessionId)
          ?.controller.abort(abortError('Session closed'));
        this._sessionLifecycleControllers.delete(sessionId);
        this._sessionRevision.set(sessionId, this._getSessionRevision(sessionId) + 1);
        this._agentState.delete(sessionId);
        this._agentStatusRevision.delete(sessionId);
        this._agentStatusFieldRevision.delete(sessionId);
        this._latestAgentStatus.delete(sessionId);
        this._dispatchLog.delete(sessionId);
        for (const key of this._terminalPromptIds.keys()) {
          if (key.startsWith(`${sessionId}\u0000`)) {
            this._terminalPromptIds.delete(key);
          }
        }
        for (const key of this._queued.keys()) {
          if (!key.startsWith(`${sessionId}\u0000`)) continue;
          for (const state of this._queued.get(key) ?? []) {
            state.dispatchController.abort(abortError('Session closed'));
            this._signalWorkerTerminal(state);
          }
          this._queued.delete(key);
        }
        for (const key of this._steerReservations.keys()) {
          if (!key.startsWith(`${sessionId}\u0000`)) continue;
          this._steerReservations
            .get(key)
            ?.controller.abort(abortError('Session closed'));
          this._steerReservations.delete(key);
        }
        // Active prompt state belongs to the session lifecycle too. Removing
        // it makes any in-flight terminal verification harmless: its identity
        // guard observes that the state is no longer authoritative and exits
        // without publishing a synthetic completion for the closed session.
        for (const [key, state] of this._active) {
          if (!key.startsWith(`${sessionId}\u0000`)) continue;
          state.dispatchController.abort(abortError('Session closed'));
          this._signalWorkerTerminal(state);
          this._active.delete(key);
        }
      }),
    );
  }

  // --- IPromptService --------------------------------------------------------

  async list(sid: string): Promise<PromptListResponse> {
    await this._requireSession(sid);
    const key = promptKey(sid, MAIN_AGENT_ID);
    const active = this._active.get(key);
    return {
      active:
        active !== undefined && active.submitted && !active.completed && !active.aborted
          ? toPromptItem(active, 'running')
          : null,
      queued: (this._queued.get(key) ?? [])
        .filter((state) => state.submitted)
        .map((state) => toPromptItem(state, 'queued')),
    };
  }

  async submit(sid: string, body: PromptSubmission): Promise<PromptSubmitResult> {
    // Register every submission synchronously, before any preflight await.
    // This fixes both admission order and stop-vs-preflight races: a follower
    // keeps its invocation-order queue position even if a later request finishes
    // auth first, and promotion makes an unfinished preflight visible to the
    // session abort path instead of letting it revive after stop returns.
    const sessionRevision = this._getSessionRevision(sid);
    const promptId = `prompt_${ulid()}`;
    const state = this._createPromptState(sid, promptId, body, sessionRevision);
    const key = promptKey(sid, state.agentId);
    if (
      this._active.get(key) === undefined &&
      this._steerReservations.get(key) === undefined
    ) {
      this._active.set(key, state);
    } else {
      this._enqueue(sid, state);
    }
    const signal = state.dispatchController.signal;

    try {
      await this._requireSession(sid, signal);
      this._assertSubmissionAuthority(sid, key, state);
      const resumeRequest = this._trackStartupFence(
        sid,
        key,
        state,
        this.core.rpc.resumeSession({ sessionId: sid }, { signal }),
      );
      await abortable(
        resumeRequest,
        signal,
      );
      this._assertSubmissionAuthority(sid, key, state);

      // Readiness gate. Throws AuthProvisioningRequired / AuthTokenMissing /
      // AuthModelNotResolved before the prompt is handed off to agent-core.
      // Daemon route layer maps these to 40110/40111/40113.
      await abortable(this.auth.ensureReady(), signal);
      this._assertSubmissionAuthority(sid, key, state);
      state.preflightPending = false;
      state.readyToStart = true;

      if (this._active.get(key) !== state) {
        const item = toPromptItem(state, 'queued');
        state.submitted = true;
        this._publishSubmitted(sid, state, item);
        return item;
      }

      const item = toPromptItem(state, 'running');
      const startup = this._startPrompt(sid, state, () => {
        state.submitted = true;
        this._publishSubmitted(sid, state, item);
      });
      // A stop must settle the HTTP submit even when a mocked/uncooperative
      // control RPC ignores its transport signal. The internal startup promise
      // remains observed and retains the scheduler fence until that mutation
      // itself settles, so a queued successor cannot overlap it.
      await abortable(startup, signal);
      return item;
    } catch (error) {
      state.preflightPending = false;
      if (!state.readyToStart) {
        this._discardRegisteredSubmission(sid, key, state);
      }
      if (signal.aborted) {
        throw new PromptAlreadyCompletedError(sid, state.promptId);
      }
      throw error;
    }
  }

  async startBtw(sid: string): Promise<string> {
    const sessionRevision = this._getSessionRevision(sid);
    const signal = this._getSessionLifecycleSignal(sid, sessionRevision);
    await this._requireSession(sid, signal);
    this._assertSessionRevision(sid, sessionRevision);
    await abortable(
      this.core.rpc.resumeSession({ sessionId: sid }, { signal }),
      signal,
    );
    this._assertSessionRevision(sid, sessionRevision);
    await abortable(this.auth.ensureReady(), signal);
    this._assertSessionRevision(sid, sessionRevision);
    const agentId = await abortable(
      this.core.rpc.startBtw(
        {
          sessionId: sid,
          agentId: MAIN_AGENT_ID,
        },
        { signal },
      ),
      signal,
    );
    this._assertSessionRevision(sid, sessionRevision);
    return agentId;
  }

  async steer(sid: string, promptIds: readonly string[]): Promise<PromptSteerResult> {
    const sessionRevision = this._getSessionRevision(sid);
    const lifecycleSignal = this._getSessionLifecycleSignal(sid, sessionRevision);
    await this._requireSession(sid, lifecycleSignal);
    this._assertSessionRevision(sid, sessionRevision);
    if (promptIds.length === 0) {
      throw new PromptNotFoundError(sid, '');
    }
    const key = promptKey(sid, MAIN_AGENT_ID);
    if (this._steerReservations.has(key)) {
      const activePromptId = this._steerReservations.get(key)?.target.promptId ?? '';
      throw new SessionBusyError(sid, activePromptId);
    }
    const active = this._active.get(key);
    if (active === undefined || active.completed || active.aborted || active.aborting) {
      throw new PromptNotFoundError(sid, promptIds[0]!);
    }
    const queue = this._queued.get(key) ?? [];
    const requestedPromptIds = new Set(promptIds);
    for (const promptId of requestedPromptIds) {
      if (!queue.some((item) => item.submitted && item.promptId === promptId)) {
        throw new PromptNotFoundError(sid, promptId);
      }
    }
    const selected = queue.filter(
      (state) => state.submitted && requestedPromptIds.has(state.promptId),
    );
    const selectedPromptIds = selected.map((state) => state.promptId);

    if (!active.promptDispatched) {
      throw new SessionBusyError(sid, active.promptId);
    }

    const reservation: SteerReservation = {
      sessionRevision,
      target: active,
      selected,
      controller: new AbortController(),
    };
    this._steerReservations.set(key, reservation);
    // Keep reserved prompts in the authoritative queue until core admission
    // succeeds. The reservation itself blocks scheduling and mutation, while
    // list() continues to expose the pending work. Failure and cancellation
    // therefore need no error-prone remove/restore dance.
    const signal = AbortSignal.any([
      lifecycleSignal,
      reservation.controller.signal,
    ]);

    try {
      await abortable(
        this.core.rpc.steer(
          {
            sessionId: sid,
            agentId: MAIN_AGENT_ID,
            input: steerContentToCoreParts(selected),
            expectedPromptId: active.promptId,
            requireActive: true,
          },
          { signal },
        ),
        signal,
      );
      this._assertSessionRevision(sid, sessionRevision);
      signal.throwIfAborted();

      const selectedSet = new Set(selected);
      const currentQueue = this._queued.get(key) ?? [];
      this._replaceQueue(
        sid,
        MAIN_AGENT_ID,
        currentQueue.filter((state) => !selectedSet.has(state)),
      );
      for (const state of selected) {
        this._recordTerminalPromptId(key, state.promptId);
      }
      // Admission is committed before observers see prompt.steered. A
      // synchronous listener that acts on a selected id must observe it as a
      // terminal child, never as a still-pending reservation that can cancel
      // the active owner.
      if (this._steerReservations.get(key) === reservation) {
        this._steerReservations.delete(key);
      }

      const event: SyntheticPromptSteeredEvent = {
        type: 'prompt.steered',
        agentId: MAIN_AGENT_ID,
        sessionId: sid,
        activePromptId: active.promptId,
        promptIds: selectedPromptIds,
        content: selected.flatMap((state) => state.body.content),
        steeredAt: new Date().toISOString(),
      };
      this.eventService.publish(event as unknown as Event);
      return { steered: true, prompt_ids: selectedPromptIds };
    } catch (error) {
      if (reservation.selected.every((state) => state.aborted)) {
        // A concurrent stop linearized this steer selection into the active
        // cancellation transaction. Surface the declared terminal-prompt
        // sentinel instead of leaking an internal AbortError as a generic 500.
        throw new PromptAlreadyCompletedError(sid, selectedPromptIds[0]!);
      }
      if (isKimiError(error) && error.code === 'turn.agent_busy') {
        throw new SessionBusyError(sid, active.promptId);
      }
      throw error;
    } finally {
      if (this._steerReservations.get(key) === reservation) {
        this._steerReservations.delete(key);
      }
      if (!this._disposed && this._getSessionRevision(sid) === sessionRevision) {
        this._resumeTerminalAfterSteer(sid, key, active);
        this._startNextQueued(sid, MAIN_AGENT_ID);
      }
    }
  }

  private async _startPrompt(
    sid: string,
    state: PromptState,
    onStarted?: () => void,
  ): Promise<void> {
    const key = promptKey(sid, state.agentId);
    if (this._active.get(key) !== state) {
      if (this._active.get(key) !== undefined) {
        throw new PromptAlreadyCompletedError(sid, state.promptId);
      }
      this._assertSessionRevision(sid, state.sessionRevision);
      this._active.set(key, state);
    }
    try {
      const assertAuthority = (): void => {
        this._assertPromptAuthority(sid, key, state);
      };
      const signal = state.dispatchController.signal;

      await this._withSessionControlLock(sid, state.sessionRevision, async () => {
        assertAuthority();
        state.startupPending = true;
        const overridePatch =
          state.agentId === MAIN_AGENT_ID ? pickAgentStatePatch(state.body) : undefined;
        if (overridePatch !== undefined) {
          const bootstrapped = await this._ensureAgentStateBootstrapped(
            sid,
            state.sessionRevision,
            assertAuthority,
            signal,
          );
          assertAuthority();
          if (!bootstrapped) {
            throw new PromptAlreadyCompletedError(sid, state.promptId);
          }
          await this._applyAgentStateInternal(
            sid,
            overridePatch,
            'prompt',
            state.promptId,
            assertAuthority,
          );
        }
        assertAuthority();

        if (state.agentId === MAIN_AGENT_ID) {
          const goalRevision = state.goalRevision;
          const goal = await this.core.rpc.getGoal(
            {
              sessionId: sid,
              agentId: state.agentId,
            },
            { signal },
          );
          assertAuthority();
          if (state.goalRevision === goalRevision) {
            this._recordGoalSnapshot(state, goal.goal);
          }
        }

        const input = contentToCoreParts(state.body.content);
        onStarted?.();
        assertAuthority();
        this._logger.debug(
          { sid, promptId: state.promptId, agentId: state.agentId, partCount: input.length },
          '[DBG prompt-service.submit] -> core.rpc.prompt(...)',
        );
        state.promptDispatched = true;
        const promptRequest = this.core.rpc
          .prompt(
            {
              sessionId: sid,
              agentId: state.agentId,
              input,
              promptId: state.promptId,
            },
            { signal },
          )
          .then((result) => {
            state.dispatchAcknowledged = true;
            if (result?.kind === 'started') {
              state.acceptedTurnId = result.turnId;
              if (!state.turnObserved) state.turnId = result.turnId;
            } else if (result?.kind === 'deferred') {
              state.acceptedDeferredPromptId = result.deferredPromptId;
            }
            return result;
          });
        // The RPC proxy invokes the actual Agent handler asynchronously. Keep
        // the single-writer lock through its acknowledgement so a later profile
        // setter cannot overtake this prompt's overrides. A verified terminal
        // worker event is an equivalent barrier and releases the lock even if
        // the transport response itself is uncooperative.
        await Promise.race([promptRequest, state.workerTerminalBarrier]);
      });

      this._logger.debug(
        { sid, promptId: state.promptId },
        '[DBG prompt-service.submit] core.rpc.prompt(...) acknowledged',
      );
    } catch (error) {
      state.startupFailed = true;
      if (!state.aborting) this._settleStartupFailure(sid, key, state);
      this._logger.debug(
        { sid, promptId: state.promptId, err: (error as Error)?.message ?? error },
        '[DBG prompt-service.submit] failed to start prompt',
      );
      if (isKimiError(error) && error.code === 'turn.agent_busy') {
        throw new SessionBusyError(sid, state.promptId);
      }
      throw error;
    } finally {
      state.startupPending = false;
      if ((state.aborted || state.completed) && !state.aborting) {
        this._retireStartupState(sid, key, state);
      }
    }
  }

  private _publishSubmitted(sid: string, state: PromptState, item: PromptSubmitResult): void {
    const event: SyntheticPromptSubmittedEvent = {
      type: 'prompt.submitted',
      agentId: state.agentId,
      sessionId: sid,
      promptId: item.prompt_id,
      userMessageId: item.user_message_id,
      status: item.status,
      content: item.content,
      createdAt: item.created_at,
    };
    this.eventService.publish(event);
  }

  private _publishAborted(sid: string, agentId: string, pid: string): void {
    const ev: SyntheticPromptAbortedEvent = {
      type: 'prompt.aborted',
      agentId,
      sessionId: sid,
      promptId: pid,
      abortedAt: new Date().toISOString(),
    };
    // Fire typed listeners BEFORE publishing the synth event: PromptService
    // must still trigger the typed event THEN call publish() for the synthetic
    // event.
    this._onDidAbort.fire(ev);
    this.eventService.publish(ev as unknown as Event);
  }

  async abort(sid: string, pid: string): Promise<PromptAbortResult> {
    // Claim active authority before the first await. A stop request must beat
    // an older submit whose list/auth preflight is still pending.
    const active = this._findActivePrompt(sid, pid);
    if (active !== undefined) {
      const [key, state] = active;
      if (state.completed || state.aborted || state.aborting) {
        throw new PromptAlreadyCompletedError(sid, pid);
      }
      return this._beginActiveAbort(sid, key, state);
    }

    const reserved = this._findSteerReservation(sid, pid);
    if (reserved !== undefined) {
      const [key, reservation] = reserved;
      if (reservation.target.aborting && reservation.target.abortPromise !== undefined) {
        return reservation.target.abortPromise;
      }
      if (
        reservation.target.completed ||
        reservation.target.aborted ||
        reservation.target.aborting
      ) {
        throw new PromptAlreadyCompletedError(sid, pid);
      }
      // A selected queued prompt and its active target form one transaction
      // once steer admission is reserved. Stopping either id cancels both,
      // which gives clients holding the newer queued id a race-free stop path.
      return this._beginActiveAbort(sid, key, reservation.target);
    }

    // Queued prompt: remove it synchronously. The queue is the session proof;
    // waiting for listSessions first would reopen the same preflight race.
    const queued = this._findQueuedPrompt(sid, pid);
    if (queued !== undefined) {
      const [key, queue, index, state] = queued;
      state.dispatchController.abort(userCancellationReason());
      state.aborted = true;
      state.preflightPending = false;
      queue.splice(index, 1);
      if (queue.length === 0) this._queued.delete(key);
      this._recordTerminalPromptId(key, pid);
      if (state.submitted) this._publishAborted(sid, state.agentId, pid);
      return { aborted: true };
    }

    await this._requireSession(sid);
    if (this._hasTerminalPrompt(sid, pid)) {
      throw new PromptAlreadyCompletedError(sid, pid);
    }
    throw new PromptNotFoundError(sid, pid);
  }

  private _beginActiveAbort(
    sid: string,
    key: string,
    state: PromptState,
  ): Promise<PromptAbortResult> {
    const signal = this._getSessionLifecycleSignal(sid, state.sessionRevision);
    const turnId = state.turnId;
    state.aborting = true;
    state.dispatchController.abort(userCancellationReason());
    const reservation = this._steerReservations.get(key);
    if (reservation?.target === state) {
      // Do not let an uncooperative/hung steer acknowledgement pin stop and
      // the scheduler forever. The correlated core cancel below removes any
      // owner-targeted steer that was admitted before this signal won the race.
      reservation.controller.abort(userCancellationReason());
      this._cancelReservedPrompts(sid, key, reservation.selected);
    }
    const operation = Promise.resolve().then(() =>
      this._performActiveAbort(sid, key, state, turnId, signal),
    );
    state.abortPromise = operation;
    void operation.then(
      () => {
        if (state.abortPromise === operation) state.abortPromise = undefined;
      },
      () => {
        if (state.abortPromise === operation) state.abortPromise = undefined;
      },
    );
    return operation;
  }

  private async _performActiveAbort(
    sid: string,
    key: string,
    state: PromptState,
    turnId: number | null,
    signal: AbortSignal,
  ): Promise<PromptAbortResult> {
    try {
      signal.throwIfAborted();
      if (
        this._getSessionRevision(sid) !== state.sessionRevision ||
        this._active.get(key) !== state
      ) {
        return { aborted: true };
      }
      if (!state.promptDispatched) {
        state.aborting = false;
        this._markAbortedPrompt(sid, key, state);
        if (!state.startupPending) this._retireStartupState(sid, key, state);
        return { aborted: true };
      }
      const cancelArgs: {
        sessionId: string;
        agentId: string;
        turnId?: number;
        expectedPromptId: string;
        requireActive: true;
      } = {
        sessionId: sid,
        agentId: state.agentId,
        expectedPromptId: state.promptId,
        requireActive: true,
      };
      if (turnId !== null) cancelArgs.turnId = turnId;
      await abortable(this.core.rpc.cancel(cancelArgs, { signal }), signal);
    } catch (error) {
      state.aborting = false;
      if (isKimiError(error) && error.code === 'turn.agent_busy') {
        // The owner-qualified cancellation is allowed to lose a race with the
        // target's natural terminal boundary, but it can never affect a newer
        // unrelated worker. Treat that safe miss as an idempotent stop.
        if (this._active.get(key) === state) {
          this._markAbortedPrompt(sid, key, state);
          this._signalWorkerTerminal(state);
          this._retireStartupState(sid, key, state);
        }
        return { aborted: true };
      }
      if (this._active.get(key) === state) {
        if (!state.promptDispatched) {
          this._markAbortedPrompt(sid, key, state);
          if (!state.startupPending) this._retireStartupState(sid, key, state);
        } else if (state.startupFailed) {
          this._settleStartupFailure(sid, key, state);
        } else if (state.deferredOutcome !== null) {
          if (state.goalActive) {
            void this._verifyTerminalPrompt(sid, key, state);
          } else {
            this._settleTerminalPrompt(
              sid,
              key,
              state,
              state.deferredOutcome,
              state.turnRevision,
            );
          }
        }
      }
      throw error;
    }

    state.aborting = false;
    if (
      this._getSessionRevision(sid) !== state.sessionRevision ||
      this._active.get(key) !== state
    ) {
      return { aborted: true };
    }
    this._markAbortedPrompt(sid, key, state);
    if (state.promptDispatched) this._signalWorkerTerminal(state);
    if (state.promptDispatched || !state.startupPending) {
      // Once prompt dispatch is issued, the same lifecycle signal prevents a
      // not-yet-invoked RPC handler from launching late; if the handler already
      // ran, successful cancel is its terminal barrier. Before dispatch, retain
      // only an uncooperative raw control mutation until it truly settles.
      this._retireStartupState(sid, key, state);
    }
    return { aborted: true };
  }

  async abortBySession(sid: string): Promise<PromptAbortResult> {
    const sessionRevision = this._getSessionRevision(sid);
    const signal = this._getSessionLifecycleSignal(sid, sessionRevision);
    const key = promptKey(sid, MAIN_AGENT_ID);
    const state = this._active.get(key);
    if (state !== undefined && !state.completed && !state.aborted && !state.aborting) {
      return this._beginActiveAbort(sid, key, state);
    }
    if (state?.aborting === true && state.abortPromise !== undefined) {
      return state.abortPromise;
    }
    if (state?.aborted === true) return { aborted: true };

    await this._requireSession(sid, signal);
    this._assertSessionRevision(sid, sessionRevision);
    // A submit can reserve the scheduler slot while validation is in flight.
    const current = this._active.get(key);
    if (current !== undefined && !current.completed && !current.aborted && !current.aborting) {
      return this._beginActiveAbort(sid, key, current);
    }
    if (current?.aborting === true && current.abortPromise !== undefined) {
      return current.abortPromise;
    }
    if (current?.aborted === true) return { aborted: true };

    // No daemon-managed active prompt. Cancel any direct agent-core turn such
    // as a skill activation without requiring a turn id.
    this._assertSessionRevision(sid, sessionRevision);
    await abortable(
      this.core.rpc.cancel(
        { sessionId: sid, agentId: MAIN_AGENT_ID },
        { signal },
      ),
      signal,
    );
    return { aborted: true };
  }

  getCurrentPromptId(sid: string): string | undefined {
    const state = this._active.get(promptKey(sid, MAIN_AGENT_ID));
    if (state === undefined || !state.submitted || state.completed || state.aborted) {
      return undefined;
    }
    return state.promptId;
  }

  /**
   * `IPromptService.applyAgentState` — entry point shared by
   * `submit` (per-turn override) and `SessionService.update`
   * (`POST /sessions/{sid}/profile`). Validates the session exists,
   * bootstraps the shadow lazily, then diff-dispatches each non-shadow
   * field through the matching `core.rpc.*` setter. Dispatch-log
   * entries are tagged with the `source` so downstream observers can
   * tell prompt-driven and profile-driven setters apart.
   *
   * No-op when every field matches the shadow; throws on setter failure
   * (the caller / route layer surfaces the error). Empty `patch` is
   * accepted and bootstraps nothing — useful for SessionService.update
   * paths that need to no-op cleanly when the body carries no runtime
   * controls.
   */
  async applyAgentState(
    sid: string,
    patch: AgentStatePatch,
    source: AgentStateSource,
    promptId?: string,
  ): Promise<void> {
    if (!hasAnyAgentStateField(patch)) return;
    const sessionRevision = this._getSessionRevision(sid);
    const signal = this._getSessionLifecycleSignal(sid, sessionRevision);
    await this._requireSession(sid, signal);
    const assertAuthority = (): void => {
      this._assertSessionRevision(sid, sessionRevision);
    };
    assertAuthority();
    await this._withSessionControlLock(sid, sessionRevision, async () => {
      assertAuthority();
      const bootstrapped = await this._ensureAgentStateBootstrapped(
        sid,
        sessionRevision,
        assertAuthority,
        signal,
      );
      assertAuthority();
      if (!bootstrapped) throw new SessionNotFoundError(sid);
      await this._applyAgentStateInternal(
        sid,
        patch,
        source,
        promptId ?? '',
        assertAuthority,
        signal,
      );
    });
  }

  // --- IPromptService typed event accessors ---------------------------------
  //
  // `onDidComplete` / `onDidAbort` are declared above as `Emitter<T>.event`
  // getters; consumers subscribe via `svc.onDidComplete(handler)` (returns
  // IDisposable) and own the detach lifetime through
  // `Disposable._register(...)`.

  // --- Stateless session controls (per-request diff dispatch) ---------------

  /**
   * Seed the per-session shadow from `getConfig` / `getPermission` /
   * `getPlan` if not yet bootstrapped. Idempotent across submits within a
   * session lifetime; cleared on `ISessionService.onDidClose`.
   *
   * The three RPCs run in parallel — they share no preconditions.
   */
  private async _ensureAgentStateBootstrapped(
    sid: string,
    sessionRevision: number,
    assertAuthority: () => void = () => {},
    signal?: AbortSignal,
  ): Promise<boolean> {
    if (this._getSessionRevision(sid) !== sessionRevision) return false;
    if (this._agentState.has(sid)) return true;
    const statusRevision = this._agentStatusRevision.get(sid) ?? 0;
    const [config, permission, plan, swarmMode] = await Promise.all([
      this.core.rpc.getConfig({ sessionId: sid, agentId: MAIN_AGENT_ID }, { signal }),
      this.core.rpc.getPermission({ sessionId: sid, agentId: MAIN_AGENT_ID }, { signal }),
      this.core.rpc.getPlan({ sessionId: sid, agentId: MAIN_AGENT_ID }, { signal }),
      this.core.rpc.getSwarmMode({ sessionId: sid, agentId: MAIN_AGENT_ID }, { signal }),
    ]);
    assertAuthority();
    if (this._getSessionRevision(sid) !== sessionRevision) return false;
    const snapshot: AgentStateSnapshot = {};
    if (config.modelAlias !== undefined) snapshot.model = config.modelAlias;
    // `AgentConfigData.thinkingEffort` is typed `string` but in practice
    // takes one of the `PromptThinking` literals (`off|low|...|max`); the
    // narrow cast lets diff comparisons stay typed without forcing
    // protocol to import from agent-core.
    snapshot.thinking = config.thinkingEffort as PromptThinking;
    snapshot.permissionMode = permission.mode;
    snapshot.planMode = plan !== null;
    snapshot.swarmMode = swarmMode;
    const liveStatus = this._latestAgentStatus.get(sid);
    if (liveStatus !== undefined && liveStatus.revision > statusRevision) {
      Object.assign(snapshot, liveStatus.snapshot);
    }
    this._agentState.set(sid, snapshot);
    return true;
  }

  /**
   * Diff-dispatch: for each of the four controls present on `patch`,
   * call the matching `core.rpc.*` setter ONLY when the value differs
   * from the shadow. Each setter runs serially so any failure surfaces
   * to the caller. Each successful setter also appends to the per-session
   * dispatch-log ring buffer; absence of an entry between two prompts is
   * the proof that the shadow suppressed a redundant dispatch.
   *
   * Pre-condition: `_ensureAgentStateBootstrapped(sid)` already ran (the
   * shadow Map carries `sid`). Callers must guard.
   */
  private async _applyAgentStateInternal(
    sid: string,
    patch: AgentStatePatch,
    source: AgentStateSource,
    promptId: string,
    assertAuthority: () => void = () => {},
    signal?: AbortSignal,
  ): Promise<void> {
    assertAuthority();
    const shadow = this._agentState.get(sid);
    if (shadow === undefined) {
      // Bootstrap is a precondition; a missing shadow here is a bug,
      // not a recoverable state.
      throw new Error(
        `PromptService._applyAgentStateInternal: shadow not bootstrapped for sid=${sid}`,
      );
    }
    const agentId = MAIN_AGENT_ID;
    type ShadowField = keyof AgentStateSnapshot;
    interface DispatchToken {
      field: ShadowField | undefined;
      revision: number;
    }
    const dispatch = async <T>(
      operation: () => T | PromiseLike<T>,
      field?: ShadowField,
    ): Promise<DispatchToken> => {
      assertAuthority();
      const revision =
        field === undefined
          ? 0
          : (this._agentStatusFieldRevision.get(sid)?.[field] ?? 0);
      await operation();
      return { field, revision };
    };
    const commitDispatch = (
      token: DispatchToken,
      commit: (writeShadow: boolean) => void,
    ): void => {
      // The mutation may already have taken effect before an abort/close wins
      // the response race. Commit it to the same-lifecycle shadow first, then
      // re-check prompt authority to prevent any following setter or dispatch.
      if (this._agentState.get(sid) === shadow) {
        const writeShadow =
          token.field === undefined ||
          (this._agentStatusFieldRevision.get(sid)?.[token.field] ?? 0) ===
            token.revision;
        commit(writeShadow);
      }
      assertAuthority();
    };

    if (patch.model !== undefined && patch.model !== shadow.model) {
      const payload = { sessionId: sid, agentId, model: patch.model };
      const token = await dispatch(
        () => this.core.rpc.setModel(payload, { signal }),
        'model',
      );
      commitDispatch(token, (writeShadow) => {
        if (writeShadow) shadow.model = patch.model;
        this._recordDispatch(sid, 'setModel', payload, promptId, source);
      });
    }
    if (patch.thinking !== undefined && patch.thinking !== shadow.thinking) {
      const payload = { sessionId: sid, agentId, effort: patch.thinking as PromptThinking };
      const token = await dispatch(
        () => this.core.rpc.setThinking(payload, { signal }),
        'thinking',
      );
      commitDispatch(token, (writeShadow) => {
        if (writeShadow) shadow.thinking = patch.thinking;
        this._recordDispatch(sid, 'setThinking', payload, promptId, source);
      });
    }
    if (
      patch.permission_mode !== undefined &&
      patch.permission_mode !== shadow.permissionMode
    ) {
      const payload = {
        sessionId: sid,
        agentId,
        mode: patch.permission_mode as PermissionMode,
      };
      const token = await dispatch(
        () => this.core.rpc.setPermission(payload, { signal }),
        'permissionMode',
      );
      commitDispatch(token, (writeShadow) => {
        if (writeShadow) shadow.permissionMode = patch.permission_mode as PermissionMode;
        this._recordDispatch(sid, 'setPermission', payload, promptId, source);
      });
    }
    if (patch.plan_mode !== undefined && patch.plan_mode !== shadow.planMode) {
      const payload = { sessionId: sid, agentId };
      if (patch.plan_mode) {
        const token = await dispatch(
          () => this.core.rpc.enterPlan(payload, { signal }),
          'planMode',
        );
        commitDispatch(token, (writeShadow) => {
          if (writeShadow) shadow.planMode = true;
          this._recordDispatch(sid, 'enterPlan', payload, promptId, source);
        });
      } else {
        // `cancelPlan({id?})` accepts an omitted id — `PlanMode.cancel`
        // clears whatever id is currently active. Shadow doesn't track
        // ids, so we always omit.
        const token = await dispatch(
          () => this.core.rpc.cancelPlan(payload, { signal }),
          'planMode',
        );
        commitDispatch(token, (writeShadow) => {
          if (writeShadow) shadow.planMode = false;
          this._recordDispatch(sid, 'cancelPlan', payload, promptId, source);
        });
      }
    }

    // Swarm mode toggle. enterSwarm/exitSwarm are idempotent no-throw on
    // the agent side; we still guard with the shadow to avoid redundant
    // dispatch-log entries.
    if (patch.swarm_mode !== undefined && patch.swarm_mode !== shadow.swarmMode) {
      const payload = { sessionId: sid, agentId };
      if (patch.swarm_mode) {
        const enterPayload = { ...payload, trigger: 'manual' as const };
        const token = await dispatch(
          () => this.core.rpc.enterSwarm(enterPayload, { signal }),
          'swarmMode',
        );
        commitDispatch(token, (writeShadow) => {
          if (writeShadow) shadow.swarmMode = true;
          this._recordDispatch(sid, 'enterSwarm', enterPayload, promptId, source);
        });
      } else {
        const token = await dispatch(
          () => this.core.rpc.exitSwarm(payload, { signal }),
          'swarmMode',
        );
        commitDispatch(token, (writeShadow) => {
          if (writeShadow) shadow.swarmMode = false;
          this._recordDispatch(sid, 'exitSwarm', payload, promptId, source);
        });
      }
    }

    // Goal creation. createGoal throws KimiError on invalid input
    // (GOAL_OBJECTIVE_EMPTY, GOAL_OBJECTIVE_TOO_LONG) or when a goal is
    // already active without replace=true (GOAL_ALREADY_EXISTS). Let these
    // propagate so the REST route layer can map them to the right code.
    if (patch.goal_objective !== undefined) {
      const payload = {
        sessionId: sid,
        agentId,
        objective: patch.goal_objective,
        replace: false,
      };
      const token = await dispatch(() => this.core.rpc.createGoal(payload, { signal }));
      commitDispatch(token, () => {
        this._recordDispatch(sid, 'createGoal', payload, promptId, source);
      });
      // `goal_objective` is a one-shot creation trigger; do not keep it on
      // the shadow.
    }

    // Goal lifecycle control. Each action maps to its own RPC; errors
    // (GOAL_NOT_FOUND, GOAL_STATUS_INVALID, GOAL_NOT_RESUMABLE) propagate.
    if (patch.goal_control !== undefined) {
      const payload = { sessionId: sid, agentId };
      switch (patch.goal_control) {
        case 'pause':
          {
            const token = await dispatch(() =>
              this.core.rpc.pauseGoal(payload, { signal }),
            );
            commitDispatch(token, () => {
              this._recordDispatch(sid, 'pauseGoal', payload, promptId, source);
            });
          }
          break;
        case 'resume':
          {
            const token = await dispatch(() =>
              this.core.rpc.resumeGoal(payload, { signal }),
            );
            commitDispatch(token, () => {
              this._recordDispatch(sid, 'resumeGoal', payload, promptId, source);
            });
          }
          break;
        case 'cancel':
          {
            const token = await dispatch(() =>
              this.core.rpc.cancelGoal(payload, { signal }),
            );
            commitDispatch(token, () => {
              this._recordDispatch(sid, 'cancelGoal', payload, promptId, source);
            });
          }
          break;
      }
      // `goal_control` is a one-shot action trigger; do not keep it on the
      // shadow.
    }
  }

  /**
   * Append a dispatch entry to the per-session ring buffer, evicting the
   * oldest entry when the cap is hit. Called only from
   * `_applyAgentStateInternal` after the underlying setter resolves
   * successfully.
   */
  private _recordDispatch(
    sid: string,
    kind: PromptDispatchLogEntry['kind'],
    payload: Record<string, unknown>,
    promptId: string,
    source: AgentStateSource,
  ): void {
    let buf = this._dispatchLog.get(sid);
    if (buf === undefined) {
      buf = [];
      this._dispatchLog.set(sid, buf);
    }
    buf.push({
      ts: new Date().toISOString(),
      kind,
      // Shallow copy so future shadow mutations / callers can't mutate
      // the recorded payload retroactively.
      payload: { ...payload },
      promptId,
      source,
    });
    if (buf.length > DISPATCH_LOG_CAP) {
      buf.splice(0, buf.length - DISPATCH_LOG_CAP);
    }
  }

  // --- Private event handler (replaces IPromptLifecycleObserver) ----------

  private _handleBusEvent(event: Event): void {
    const sid = (event as { sessionId?: string }).sessionId;
    if (sid === undefined || sid === '') return;

    // Mirror live main-agent status into the per-session shadow. A BTW or
    // subagent has independent controls and must never poison main's diff
    // suppression state.
    if (isAgentStatusUpdated(event)) {
      const statusAgentId = (event as { agentId?: string }).agentId ?? MAIN_AGENT_ID;
      if (statusAgentId !== MAIN_AGENT_ID) return;
      const revision = (this._agentStatusRevision.get(sid) ?? 0) + 1;
      this._agentStatusRevision.set(sid, revision);
      const fieldRevisions = this._agentStatusFieldRevision.get(sid) ?? {};
      const liveSnapshot: AgentStateSnapshot = {};
      if (event.model !== undefined) {
        liveSnapshot.model = event.model;
        fieldRevisions.model = revision;
      }
      if (event.thinkingEffort !== undefined) {
        liveSnapshot.thinking = event.thinkingEffort as PromptThinking;
        fieldRevisions.thinking = revision;
      }
      if (event.permission !== undefined) {
        liveSnapshot.permissionMode = event.permission;
        fieldRevisions.permissionMode = revision;
      }
      if (event.planMode !== undefined) {
        liveSnapshot.planMode = event.planMode;
        fieldRevisions.planMode = revision;
      }
      if (event.swarmMode !== undefined) {
        liveSnapshot.swarmMode = event.swarmMode;
        fieldRevisions.swarmMode = revision;
      }
      this._agentStatusFieldRevision.set(sid, fieldRevisions);
      this._latestAgentStatus.set(sid, {
        revision,
        snapshot: {
          ...this._latestAgentStatus.get(sid)?.snapshot,
          ...liveSnapshot,
        },
      });
      const shadow = this._agentState.get(sid);
      if (shadow !== undefined) {
        Object.assign(shadow, liveSnapshot);
      }
      // status events are also published normally; fall through to allow
      // other event-type handlers below — but there's no overlap today.
      return;
    }

    const agentId = (event as { agentId?: string }).agentId ?? MAIN_AGENT_ID;
    const key = promptKey(sid, agentId);
    const state = this._active.get(key);
    if (state === undefined) return;

    if (isGoalUpdated(event)) {
      if (state.agentId !== MAIN_AGENT_ID) return;
      this._recordGoalSnapshot(state, event.snapshot);
      if (!state.goalActive && !state.aborting && state.deferredOutcome !== null) {
        // A live goal.updated event is newer and more authoritative than any
        // getGoal verification already in flight. Settle directly so an
        // uncooperative/stale verification RPC cannot pin the prompt and its
        // queue forever; the normal identity/revision guards make a late RPC
        // result harmless.
        this._settleTerminalPrompt(
          sid,
          key,
          state,
          state.deferredOutcome,
          state.turnRevision,
        );
      }
      return;
    }

    if (isTurnStarted(event)) {
      if (!state.promptDispatched) return;
      if (!this._ownsTurnStartedEvent(state, event)) return;
      state.turnId = event.turnId;
      state.turnObserved = true;
      state.turnRevision += 1;
      state.deferredOutcome = null;
      return;
    }

    if (isTurnEnded(event)) {
      if (!state.promptDispatched) return;
      // Only fire on the top-level turn end. Nested turn.ended events fly
      // through without prompt-level synthesis.
      if (state.turnId === null || event.turnId !== state.turnId) return;

      state.turnId = null;
      state.turnRevision += 1;
      state.deferredOutcome = event.reason;

      if (state.aborting) {
        return;
      }

      if (state.agentId === MAIN_AGENT_ID) {
        void this._verifyTerminalPrompt(sid, key, state);
        return;
      }

      this._settleTerminalPrompt(sid, key, state, event.reason, state.turnRevision);
    }
  }

  private _ownsTurnStartedEvent(
    state: PromptState,
    event: Event & {
      type: 'turn.started';
      turnId: number;
      promptId?: string;
      deferredPromptId?: string;
    },
  ): boolean {
    if (!state.turnObserved) {
      if (event.promptId !== undefined) {
        if (event.promptId !== state.promptId) return false;
        if (state.acceptedTurnId !== null && event.turnId !== state.acceptedTurnId) {
          return false;
        }
        if (
          state.acceptedDeferredPromptId !== null &&
          event.deferredPromptId !== state.acceptedDeferredPromptId
        ) {
          return false;
        }
        return true;
      }

      // Current workers always echo `promptId`. The acknowledgement checks are
      // retained for in-process/older bridges and make the fallback precise
      // whenever the admission result carries a turn/deferred id.
      if (!state.dispatchAcknowledged) return false;
      if (state.acceptedTurnId !== null) return event.turnId === state.acceptedTurnId;
      if (state.acceptedDeferredPromptId !== null) {
        return event.deferredPromptId === state.acceptedDeferredPromptId;
      }
      return true;
    }

    // A goal worker owns continuation turns after the initially correlated
    // turn. They intentionally carry no caller prompt id; accept them only
    // after the previous owned turn ended while the same goal remains active.
    return state.turnId === null && state.goalActive && event.promptId === undefined;
  }

  private _recordGoalSnapshot(
    state: PromptState,
    snapshot: null | { status: string },
  ): void {
    state.goalRevision += 1;
    state.goalStatus = snapshot?.status ?? null;
    state.goalActive = state.goalStatus === 'active';
    if (state.goalActive) state.goalDriven = true;
  }

  private async _verifyTerminalPrompt(
    sid: string,
    key: string,
    state: PromptState,
  ): Promise<void> {
    if (state.terminalCheckPending || state.deferredOutcome === null) return;

    state.terminalCheckPending = true;
    const outcome = state.deferredOutcome;
    const goalRevision = state.goalRevision;
    const turnRevision = state.turnRevision;
    try {
      try {
        const current = await this.core.rpc.getGoal({
          sessionId: sid,
          agentId: state.agentId,
        });
        if (this._active.get(key) !== state) return;
        if (state.goalRevision === goalRevision) {
          this._recordGoalSnapshot(state, current.goal);
        }
      } catch (error) {
        this._warnBestEffort(
          { sid, promptId: state.promptId, error },
          'failed to verify prompt termination goal state',
        );
      }

      this._settleTerminalPrompt(sid, key, state, outcome, turnRevision);
    } finally {
      state.terminalCheckPending = false;
      if (
        this._active.get(key) === state &&
        !state.aborting &&
        state.turnId === null &&
        !state.goalActive &&
        state.deferredOutcome !== null &&
        state.turnRevision !== turnRevision
      ) {
        void this._verifyTerminalPrompt(sid, key, state);
      }
    }
  }

  private _warnBestEffort(obj: object | string, msg?: string): void {
    try {
      this._logger.warn(obj, msg);
    } catch {
      // Goal verification is advisory at this terminal boundary. A broken
      // diagnostic sink must not prevent the prompt from settling.
    }
  }

  private _settleTerminalPrompt(
    sid: string,
    key: string,
    state: PromptState,
    outcome: NonNullable<PromptState['deferredOutcome']>,
    turnRevision: number,
  ): void {
    if (
      this._active.get(key) !== state ||
      state.aborting ||
      state.aborted ||
      state.turnId !== null ||
      state.turnRevision !== turnRevision ||
      state.deferredOutcome !== outcome ||
      this._steerReservations.get(key)?.target === state ||
      (state.agentId === MAIN_AGENT_ID && state.goalActive)
    ) {
      return;
    }

    state.deferredOutcome = null;
    if (outcome === 'cancelled') {
      this._finishAbortedPrompt(sid, key, state, true);
      return;
    }

    const reason =
      outcome === 'completed' && state.goalDriven && state.goalStatus === 'blocked'
        ? 'blocked'
        : outcome;
    this._completePrompt(sid, key, state, reason);
  }

  private _resumeTerminalAfterSteer(
    sid: string,
    key: string,
    state: PromptState,
  ): void {
    if (
      this._active.get(key) !== state ||
      state.aborting ||
      state.turnId !== null ||
      state.deferredOutcome === null
    ) {
      return;
    }
    if (state.agentId === MAIN_AGENT_ID && state.terminalCheckPending) {
      // The in-flight verification observes the reservation removal before it
      // attempts settlement. Re-querying after a completed verification would
      // create a TOCTOU window that can attach a newly-created goal to this old
      // prompt.
      return;
    }
    this._settleTerminalPrompt(
      sid,
      key,
      state,
      state.deferredOutcome,
      state.turnRevision,
    );
  }

  private _finishAbortedPrompt(
    sid: string,
    key: string,
    state: PromptState,
    terminalBarrier = false,
  ): void {
    if (this._active.get(key) !== state) return;
    state.aborting = false;
    if (terminalBarrier) this._signalWorkerTerminal(state);
    this._markAbortedPrompt(sid, key, state);
    if (terminalBarrier || !state.startupPending) {
      this._retireStartupState(sid, key, state);
    }
  }

  private _markAbortedPrompt(sid: string, key: string, state: PromptState): void {
    if (state.aborted) return;
    state.aborted = true;
    state.deferredOutcome = null;
    this._recordTerminalPromptId(key, state.promptId);
    if (state.submitted) this._publishAborted(sid, state.agentId, state.promptId);
  }

  private _completePrompt(
    sid: string,
    key: string,
    state: PromptState,
    reason: SyntheticPromptCompletedEvent['reason'],
  ): void {
    if (this._active.get(key) !== state) return;
    if (state.completed) return;
    this._signalWorkerTerminal(state);
    state.completed = true;
    const synth: SyntheticPromptCompletedEvent = {
      type: 'prompt.completed',
      agentId: state.agentId,
      sessionId: sid,
      promptId: state.promptId,
      finishedAt: new Date().toISOString(),
      reason,
    };
    this._recordTerminalPromptId(key, state.promptId);
    this._onDidComplete.fire(synth);
    this.eventService.publish(synth as unknown as Event);
    // A verified turn.ended is itself the terminal worker barrier. A delayed
    // response to the earlier prompt RPC cannot restart that worker, so it
    // must not pin this completed prompt or the queue.
    this._retireStartupState(sid, key, state);
  }

  /**
   * Test helper — peek at active prompt state.
   */
  _activeForTest(
    sid: string,
    agentId = MAIN_AGENT_ID,
  ): Readonly<PromptState> | undefined {
    const state = this._active.get(promptKey(sid, agentId));
    return state === undefined ? undefined : { ...state };
  }

  /**
   * Read the current runtime-controls shadow for a session, if it has been
   * bootstrapped. Returns a copy so callers cannot mutate internal state.
   */
  getAgentStateSnapshot(sid: string): AgentStateSnapshot | undefined {
    const snap = this._agentState.get(sid);
    return snap === undefined ? undefined : { ...snap };
  }

  /**
   * Test helper — peek at the per-session stateless-controls shadow.
   * Undefined before first submit on a session.
   */
  _agentStateForTest(sid: string): Readonly<AgentStateSnapshot> | undefined {
    return this.getAgentStateSnapshot(sid);
  }

  /**
   * Test / debug helper — return the per-session dispatch-log ring buffer
   * (newest-last). Returns `undefined` when the session has never
   * triggered a setter; an empty array means "saw submits but every
   * field matched the shadow". The daemon's `/debug/prompts/{sid}/dispatch-log`
   * route consumes this; unit tests assert against it directly.
   */
  _dispatchLogForTest(sid: string): readonly PromptDispatchLogEntry[] | undefined {
    const buf = this._dispatchLog.get(sid);
    if (buf === undefined) return undefined;
    // Defensive copy — callers may iterate while a parallel submit
    // pushes new entries.
    return buf.slice();
  }

  /**
   * Test helper — inject an active prompt record. Used by daemon e2e tests
   * that need to exercise the lifecycle-synthesis path WITHOUT driving a
   * real `core.rpc.prompt(...)` call (which would require an in-memory
   * KimiCore loaded with provider credentials). Not part of the public
   * contract; the underscore prefix is a "do not use in prod" signal.
   */
  _injectActiveForTest(sid: string, promptId: string, turnId: number | null): void {
    const terminal = createWorkerTerminalBarrier();
    this._active.set(promptKey(sid, MAIN_AGENT_ID), {
      agentId: MAIN_AGENT_ID,
      promptId,
      userMessageId: `msg_${sid}_pending_${promptId}`,
      body: { content: [{ type: 'text', text: 'test' }] },
      createdAt: new Date().toISOString(),
      sessionRevision: this._getSessionRevision(sid),
      preflightPending: false,
      readyToStart: true,
      submitted: true,
      startupPending: false,
      startupFailed: false,
      promptDispatched: turnId !== null,
      dispatchAcknowledged: turnId !== null,
      acceptedTurnId: turnId,
      acceptedDeferredPromptId: null,
      dispatchController: new AbortController(),
      workerTerminal: false,
      workerTerminalBarrier: terminal.promise,
      resolveWorkerTerminal: terminal.resolve,
      turnId,
      turnObserved: turnId !== null,
      turnRevision: 0,
      goalStatus: null,
      goalActive: false,
      goalDriven: false,
      goalRevision: 0,
      deferredOutcome: null,
      terminalCheckPending: false,
      aborting: false,
      abortPromise: undefined,
      completed: false,
      aborted: false,
    });
  }

  // --- internals -----------------------------------------------------------

  private _createPromptState(
    sid: string,
    promptId: string,
    body: PromptSubmission,
    sessionRevision: number,
  ): PromptState {
    const terminal = createWorkerTerminalBarrier();
    return {
      agentId: body.agent_id ?? MAIN_AGENT_ID,
      promptId,
      userMessageId: `msg_${sid}_pending_${promptId}`,
      body,
      createdAt: new Date().toISOString(),
      sessionRevision,
      preflightPending: true,
      readyToStart: false,
      submitted: false,
      startupPending: false,
      startupFailed: false,
      promptDispatched: false,
      dispatchAcknowledged: false,
      acceptedTurnId: null,
      acceptedDeferredPromptId: null,
      dispatchController: new AbortController(),
      workerTerminal: false,
      workerTerminalBarrier: terminal.promise,
      resolveWorkerTerminal: terminal.resolve,
      turnId: null,
      turnObserved: false,
      turnRevision: 0,
      goalStatus: null,
      goalActive: false,
      goalDriven: false,
      goalRevision: 0,
      deferredOutcome: null,
      terminalCheckPending: false,
      aborting: false,
      abortPromise: undefined,
      completed: false,
      aborted: false,
    };
  }

  private _getSessionRevision(sid: string): number {
    return this._sessionRevision.get(sid) ?? 0;
  }

  private _getSessionLifecycleSignal(sid: string, revision: number): AbortSignal {
    if (this._disposed || this._getSessionRevision(sid) !== revision) {
      const retired = new AbortController();
      retired.abort(
        abortError(this._disposed ? 'Prompt service disposed' : 'Session closed'),
      );
      return retired.signal;
    }
    const existing = this._sessionLifecycleControllers.get(sid);
    if (existing?.revision === revision) return existing.controller.signal;
    const controller = new AbortController();
    this._sessionLifecycleControllers.set(sid, { revision, controller });
    return controller.signal;
  }

  private _assertSessionRevision(sid: string, expected: number): void {
    if (this._disposed || this._getSessionRevision(sid) !== expected) {
      throw new SessionNotFoundError(sid);
    }
  }

  private _assertSubmissionAuthority(sid: string, key: string, state: PromptState): void {
    this._assertSessionRevision(sid, state.sessionRevision);
    const registered =
      this._active.get(key) === state ||
      (this._queued.get(key) ?? []).some((item) => item === state);
    if (!registered || state.aborting || state.aborted) {
      throw new PromptAlreadyCompletedError(sid, state.promptId);
    }
  }

  private _assertPromptAuthority(sid: string, key: string, state: PromptState): void {
    if (
      this._getSessionRevision(sid) !== state.sessionRevision ||
      this._active.get(key) !== state ||
      state.aborting ||
      state.aborted
    ) {
      throw new PromptAlreadyCompletedError(sid, state.promptId);
    }
  }

  private _retireStartupState(sid: string, key: string, state: PromptState): void {
    if (this._active.get(key) !== state) return;
    this._active.delete(key);
    this._startNextQueued(sid, state.agentId);
  }

  private _discardRegisteredSubmission(sid: string, key: string, state: PromptState): void {
    if (this._active.get(key) === state) {
      if ((state.aborted || state.aborting) && state.startupPending) return;
      this._retireStartupState(sid, key, state);
      return;
    }
    const queue = this._queued.get(key);
    const index = queue?.indexOf(state) ?? -1;
    if (queue === undefined || index < 0) return;
    queue.splice(index, 1);
    if (queue.length === 0) this._queued.delete(key);
  }

  private _settleStartupFailure(sid: string, key: string, state: PromptState): void {
    if (this._active.get(key) !== state) return;
    if (state.aborted || state.completed) {
      this._retireStartupState(sid, key, state);
      return;
    }
    if (state.submitted) {
      this._completePrompt(sid, key, state, 'failed');
      return;
    }
    this._retireStartupState(sid, key, state);
  }

  private _signalWorkerTerminal(state: PromptState): void {
    if (state.workerTerminal) return;
    state.workerTerminal = true;
    state.resolveWorkerTerminal();
  }

  private _trackStartupFence<T>(
    sid: string,
    key: string,
    state: PromptState,
    request: Promise<T>,
  ): Promise<T> {
    state.startupPending = true;
    return request.finally(() => {
      state.startupPending = false;
      if ((state.aborted || state.completed) && !state.aborting) {
        this._retireStartupState(sid, key, state);
      }
    });
  }

  private _enqueue(sid: string, state: PromptState): void {
    const key = promptKey(sid, state.agentId);
    let queue = this._queued.get(key);
    if (queue === undefined) {
      queue = [];
      this._queued.set(key, queue);
    }
    queue.push(state);
  }

  private _replaceQueue(sid: string, agentId: string, queue: PromptState[]): void {
    const key = promptKey(sid, agentId);
    if (queue.length === 0) {
      this._queued.delete(key);
      return;
    }
    this._queued.set(key, queue);
  }

  private _cancelReservedPrompts(
    sid: string,
    key: string,
    selected: readonly PromptState[],
  ): void {
    const selectedSet = new Set(selected);
    const queue = this._queued.get(key) ?? [];
    const claimed = selected.filter((state) => queue.includes(state));

    // Commit the entire batch before the first abort signal or event exposes a
    // synchronous re-entrancy point. Observers must never see an aborted prompt
    // still present in the authoritative queue, nor a half-cancelled selection.
    this._replaceQueue(
      sid,
      MAIN_AGENT_ID,
      queue.filter((state) => !selectedSet.has(state)),
    );
    for (const state of claimed) {
      state.aborted = true;
      state.preflightPending = false;
      this._recordTerminalPromptId(key, state.promptId);
    }
    for (const state of claimed) {
      state.dispatchController.abort(userCancellationReason());
    }
    for (const state of claimed) {
      if (state.submitted) this._publishAborted(sid, state.agentId, state.promptId);
    }
  }

  private _startNextQueued(sid: string, agentId = MAIN_AGENT_ID): void {
    if (this._disposed) return;
    const key = promptKey(sid, agentId);
    if (this._steerReservations.has(key)) return;
    const active = this._active.get(key);
    if (active !== undefined) return;
    const queue = this._queued.get(key);
    const next = queue?.shift();
    if (queue !== undefined && queue.length === 0) {
      this._queued.delete(key);
    }
    if (next === undefined) return;
    this._active.set(key, next);
    if (!next.readyToStart) return;
    void this._startPrompt(sid, next).catch(() => undefined);
  }

  private async _requireSession(sid: string, signal?: AbortSignal): Promise<void> {
    if (this._disposed) throw abortError('Prompt service disposed');
    const request = this.core.rpc.listSessions({ sessionId: sid }, { signal });
    const matches = signal === undefined ? await request : await abortable(request, signal);
    if (this._disposed) throw abortError('Prompt service disposed');
    if (matches.length === 0) {
      throw new SessionNotFoundError(sid);
    }
  }

  private async _withSessionControlLock<T>(
    sid: string,
    sessionRevision: number,
    operation: () => Promise<T>,
  ): Promise<T> {
    const key = `${sid}\u0000revision:${String(sessionRevision)}`;
    const previous = this._controlTails.get(key) ?? Promise.resolve();
    let release!: () => void;
    const barrier = new Promise<void>((resolve) => {
      release = resolve;
    });
    const tail = previous.then(
      () => barrier,
      () => barrier,
    );
    this._controlTails.set(key, tail);
    await previous.catch(() => undefined);
    try {
      return await operation();
    } finally {
      release();
      if (this._controlTails.get(key) === tail) this._controlTails.delete(key);
    }
  }

  private _findActivePrompt(sid: string, pid: string): [string, PromptState] | undefined {
    const prefix = `${sid}\u0000`;
    for (const [key, state] of this._active) {
      if (key.startsWith(prefix) && state.promptId === pid) return [key, state];
    }
    return undefined;
  }

  private _findQueuedPrompt(
    sid: string,
    pid: string,
  ): [string, PromptState[], number, PromptState] | undefined {
    const prefix = `${sid}\u0000`;
    for (const [key, queue] of this._queued) {
      if (!key.startsWith(prefix)) continue;
      const index = queue.findIndex((state) => state.promptId === pid);
      if (index >= 0) return [key, queue, index, queue[index]!];
    }
    return undefined;
  }

  private _findSteerReservation(
    sid: string,
    pid: string,
  ): [string, SteerReservation] | undefined {
    const prefix = `${sid}\u0000`;
    for (const [key, reservation] of this._steerReservations) {
      if (
        key.startsWith(prefix) &&
        reservation.selected.some((state) => !state.aborted && state.promptId === pid)
      ) {
        return [key, reservation];
      }
    }
    return undefined;
  }

  private _hasTerminalPrompt(sid: string, pid: string): boolean {
    const prefix = `${sid}\u0000`;
    for (const [key, terminalIds] of this._terminalPromptIds) {
      if (key.startsWith(prefix) && terminalIds.has(pid)) return true;
    }
    return false;
  }

  private _recordTerminalPromptId(key: string, pid: string): void {
    let terminalIds = this._terminalPromptIds.get(key);
    if (terminalIds === undefined) {
      terminalIds = new Set<string>();
      this._terminalPromptIds.set(key, terminalIds);
    }
    terminalIds.delete(pid);
    terminalIds.add(pid);
    while (terminalIds.size > TERMINAL_PROMPT_CAP) {
      const oldest = terminalIds.values().next().value;
      if (oldest === undefined) break;
      terminalIds.delete(oldest);
    }
  }

  override dispose(): void {
    if (this._store.isDisposed || this._disposed) return;
    this._disposed = true;
    const reason = abortError('Prompt service disposed');
    for (const state of this._active.values()) {
      state.dispatchController.abort(reason);
      this._signalWorkerTerminal(state);
    }
    for (const queue of this._queued.values()) {
      for (const state of queue) {
        state.dispatchController.abort(reason);
        this._signalWorkerTerminal(state);
      }
    }
    for (const reservation of this._steerReservations.values()) {
      reservation.controller.abort(reason);
    }
    for (const { controller } of this._sessionLifecycleControllers.values()) {
      controller.abort(reason);
    }
    this._active.clear();
    this._queued.clear();
    this._steerReservations.clear();
    this._terminalPromptIds.clear();
    this._sessionRevision.clear();
    this._controlTails.clear();
    this._sessionLifecycleControllers.clear();
    this._agentState.clear();
    this._agentStatusRevision.clear();
    this._agentStatusFieldRevision.clear();
    this._latestAgentStatus.clear();
    this._dispatchLog.clear();
    // `_onDidComplete` and `_onDidAbort` are registered via `this._register(...)`,
    // so `super.dispose()` flushes their listeners.
    super.dispose();
  }
}

// Self-register under the global singleton registry. All ctor deps are
// `@I…`-injected (@ICoreProcessService / @IEventService / @IAuthSummaryService);
// `staticArguments = []`. `supportsDelayedInstantiation = false` preserves
// current reverse-dispose semantics.
registerSingleton(IPromptService, PromptService, InstantiationType.Delayed);
