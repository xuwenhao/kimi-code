import { createHash, randomUUID } from 'node:crypto';

import { createControlledPromise, type ControlledPromise } from '@antfu/utils';
import {
  APIConnectionError,
  APIContextOverflowError,
  APIEmptyResponseError,
  APIStatusError,
  APITimeoutError,
  inputTotal,
  isContextOverflowStatusError,
  type ContentPart,
  type Message,
  type TokenUsage,
} from '@moonshot-ai/kosong';
import { basename } from 'pathe';

import type { Agent } from '..';
import {
  ErrorCodes,
  KimiError,
  type KimiErrorPayload,
  isKimiError,
  makeErrorPayload,
  toKimiErrorPayload,
} from '#/errors';
import { isAbortError, isMaxStepsExceededError } from '../../loop/errors';
import {
  createLoopEventDispatcher,
  runTurn,
  type ExecutableToolResult,
  type LoopEvent,
  type LoopRecordedEvent,
  type LoopTurnInterruptedEvent,
  type LoopTurnStopReason,
} from '../../loop/index';
import type { AgentEvent, PromptResult, TurnEndedEvent, TurnEndReason } from '../../rpc';
import type { TelemetryPropertyValue } from '../../telemetry';
import { gateImageFormatParts } from '../../tools/support/image-compress';
import { abortable, isUserCancellation, userCancellationReason } from '../../utils/abort';
import {
  USER_PROMPT_ORIGIN,
  type PromptOrigin,
  type TurnInputConsumption,
} from '../context';
import { agentRecordAppendAccepted } from '../records/append-error';
import {
  captureMediaStripSnapshot,
  stripMediaPartsBySnapshot,
  type MediaStripSnapshot,
} from '../context/projector';
import { renderUserPromptHookBlockResult, renderUserPromptHookResult } from '../../session/hooks';
import { canonicalTelemetryArgs, isPlainRecord } from './canonical-args';
import {
  renderTurnCancellationReminder,
  renderTurnFailureReminder,
  TURN_OUTCOME_REMINDER_VARIANT,
} from './outcome-reminder';
import { ToolCallDeduplicator } from './tool-dedup';
import { budgetToolResultForModel } from './tool-result-budget';

interface ActiveTurn {
  readonly turnId: number;
  readonly promptId?: string;
  readonly admission?: Pick<TurnInputConsumption, 'kind' | 'id'>;
  readonly controller: AbortController;
  readonly promise: Promise<TurnEndResult>;
  readonly firstRequest: ControlledPromise<void>;
  admitting: boolean;
  terminalizing: boolean;
  steerFlushFailed: boolean;
  steerFlushError: unknown;
}

interface BufferedSteer {
  /** Stable identity shared by turn.steer and its context consumption marker. */
  readonly admissionId?: string;
  readonly input: readonly ContentPart[];
  readonly origin: PromptOrigin;
  /** Prompt that owns this queued steer; absent for background/cron notifications. */
  readonly expectedPromptId?: string;
  /** Fallback owner for ordinary SDK prompts that do not supply promptId. */
  readonly ownerTurnId?: number;
  /** Fallback owner while such a prompt is deferred behind compaction. */
  readonly ownerDeferredPromptId?: string;
  /** Replay-only: this steer was known to still be behind a deferred prompt. */
  restorePending?: boolean;
}

export type PromptLaunchResult =
  | PromptResult
  | { readonly kind: 'busy'; readonly error: KimiErrorPayload };

interface DeferredPrompt {
  /** Public identity used only when manual compaction deferred the prompt. */
  readonly id?: string;
  /** Stable identity shared by turn.prompt and its context consumption marker. */
  readonly admissionId?: string;
  /** Turn allocated before a crash, used to replay owner-qualified cancellation. */
  turnId?: number;
  readonly input: readonly ContentPart[];
  readonly origin: PromptOrigin;
  readonly promptId?: string;
}

interface PendingTurnOutcome {
  readonly id: string;
  readonly turnId: number;
  readonly content: string;
}

interface TurnCancelOptions {
  readonly expectedPromptId?: string;
  readonly requireActive?: boolean;
  /** Owner persisted on turn.cancel; replay uses it without live-target validation. */
  readonly restoredPromptId?: string;
  readonly restoredOwnerTurnId?: number;
  readonly restoredOwnerDeferredPromptId?: string;
  readonly restoredOutcomeId?: string;
  readonly restoredOutcomeTurnId?: number;
  readonly restoredOutcomeContent?: string;
  readonly restoredCancelledTurnInputs?: readonly TurnInputConsumption[];
}

export interface TurnEndResult {
  readonly event: TurnEndedEvent;
  readonly stopReason?: LoopTurnStopReason;
  readonly blockedByUserPromptHook?: boolean;
}

interface PromptHookEndResult {
  readonly event: TurnEndedEvent;
  readonly blocked: boolean;
}

const LLM_NOT_SET_MESSAGE = 'LLM not set, send "/login" to login';

/** Origin tag for the synthetic "continue" prompt that drives each goal turn. */
const GOAL_CONTINUATION_ORIGIN: PromptOrigin = { kind: 'system_trigger', name: 'goal_continuation' };
const GOAL_RATE_LIMIT_PAUSE_REASON = 'Paused after provider rate limit';
const GOAL_PROVIDER_CONNECTION_PAUSE_PREFIX = 'Paused after provider connection error';
const GOAL_PROVIDER_AUTH_PAUSE_PREFIX = 'Paused after provider authentication error';
const GOAL_PROVIDER_API_PAUSE_PREFIX = 'Paused after provider API error';
const GOAL_MODEL_CONFIG_PAUSE_PREFIX = 'Paused after model configuration error';
const GOAL_RUNTIME_PAUSE_PREFIX = 'Paused after runtime error';
const GOAL_PROVIDER_FILTERED_PAUSE_REASON = 'Paused after provider safety policy block';

/**
 * The prompt the goal driver appends to start each continuation turn — the
 * autonomous stand-in for the user typing "continue". The model decides when to
 * stop by calling `UpdateGoal`; otherwise the driver runs another turn.
 */
const GOAL_CONTINUATION_PROMPT = [
  'Continue working toward the active goal.',
  'Keep the self-audit brief. Do not explore unrelated interpretations once the goal can be',
  'decided. If the objective is simple, already answered, impossible, unsafe, or contradictory,',
  'do not run another goal turn. Explain briefly if useful, then call UpdateGoal with `complete`',
  'or `blocked` in the same turn. Otherwise, weigh the objective and any completion criteria',
  'against the work done so far, choose one bounded, useful slice of work, and use the existing',
  'conversation context and your tools. Do not try to finish a broad goal in one turn unless the',
  'whole goal is genuinely small. Most goal turns should not call UpdateGoal: after completing a',
  'useful slice, if material work remains, end the turn normally without calling UpdateGoal so',
  'the runtime can continue the goal in the next turn. Call UpdateGoal with `complete` only when',
  'all required work is done, any stated validation has passed, and there is no useful next',
  'action. Completion audit: before calling `complete`, verify the current state against the',
  'actual objective and every explicit requirement. Treat weak or indirect evidence as not',
  'complete. Do not mark complete after only producing a plan, summary, first pass, or partial',
  'result. Do not mark complete merely because a budget is nearly exhausted or you want to stop.',
  'Blocked audit: do not call UpdateGoal with `blocked` the first time you hit a blocker. Use',
  '`blocked` only for a genuine impasse: an external condition, required user input, missing',
  'credentials or permissions, or a persistent technical failure. For those non-terminal',
  'blockers, the same blocking condition must repeat for at least 3 consecutive goal turns before',
  'you call `blocked`, counting the original/user-triggered turn and automatic continuations.',
  'If a previously blocked goal is resumed, treat the resumed run as a fresh blocked audit.',
  'Exception: if the objective itself is impossible, unsafe, or contradictory, call UpdateGoal',
  'with `blocked` in the same turn; do not run more goal turns just to satisfy the audit. Do not',
  'use `blocked` because the work is large, hard, slow, uncertain, incomplete, still needs',
  'validation, would benefit from clarification, or needs more goal turns. Once the 3-turn',
  'threshold is met and you cannot make meaningful progress without user input or an',
  'external-state change, call UpdateGoal with `blocked`; do not keep reporting the blocker while',
  'leaving the goal active. Do not ask the user for input unless a real blocker prevents progress.',
].join(' ');

export class TurnFlow {
  private steerBuffer: BufferedSteer[] = [];
  private deferredPrompts: DeferredPrompt[] = [];
  private readonly pendingOutcomes = new Map<string, PendingTurnOutcome>();
  private turnId = -1;
  private activeTurn: 'resuming' | ActiveTurn | null = null;
  private closed = false;
  private shutdownPromise: Promise<void> | null = null;
  private cancelSetupPromise: ControlledPromise | null = null;
  private readonly toolCallStartedAt = new Map<string, { name: string; startedAt: number }>();
  private readonly toolCallDupType = new Map<string, 'normal' | 'cross_step'>();
  private readonly stepToolCallKeys = new Map<number, Set<string>>();
  private readonly telemetryModeByTurn = new Map<number, 'agent' | 'plan'>();
  private readonly currentStepByTurn = new Map<number, number>();
  private readonly interruptedTelemetryTurnIds = new Set<number>();
  private readonly stepFailureByTurn = new Map<number, LoopTurnInterruptedEvent>();
  private currentStep = 0;

  constructor(protected readonly agent: Agent) {}

  private bestEffortWarn(message: string, payload?: unknown): void {
    try {
      this.agent.log.warn(message, payload);
    } catch {
      // These diagnostics report work that is already durable or a control
      // transition that must continue. A broken log sink cannot roll either
      // back, and surfacing its error would invite an unsafe retry.
    }
  }

  private bestEffortError(message: string, payload?: unknown): void {
    try {
      this.agent.log.error(message, payload);
    } catch {
      // Same fail-open boundary as bestEffortWarn. In these call sites the
      // original failure is already being contained; the diagnostic sink must
      // not replace it with a second failure or interrupt terminal cleanup.
    }
  }

  /** Best-effort agent id (main / generated id) derived from the agent homedir. */
  private get agentId(): string {
    return this.agent.homedir ? basename(this.agent.homedir) : this.agent.type;
  }

  // Returns the new turnId, or null if the prompt was deferred or rejected as busy.
  // Existing in-process callers intentionally retain the historical API; RPC
  // callers use submitPrompt() to distinguish those two null outcomes.
  prompt(input: readonly ContentPart[], origin: PromptOrigin = USER_PROMPT_ORIGIN): number | null {
    const result = this.submitPrompt(input, origin);
    return result.kind === 'started' ? result.turnId : null;
  }

  /**
   * Submit a prompt while preserving the distinction between an accepted
   * prompt deferred by manual compaction and a prompt rejected by an active
   * turn. Rejected input is never written to the replay log.
   */
  submitPrompt(
    input: readonly ContentPart[],
    origin: PromptOrigin = USER_PROMPT_ORIGIN,
    promptId?: string,
  ): PromptLaunchResult {
    // The last funnel before a prompt lands in the session history: images
    // in formats providers reject (AVIF, HEIC, …) become text notices here,
    // so no caller — the SDK/RPC prompt path included — can poison the
    // session. Upstream ingestion points already gate; this is the backstop.
    const gated = gateImageFormatParts(input);
    const admission = { kind: 'prompt', id: randomUUID() } as const;
    return this.launch(gated, origin, promptId, admission, (deferredPromptId, turnId) => {
      this.agent.records.logRecord({
        type: 'turn.prompt',
        input: gated,
        origin,
        admissionId: admission.id,
        turnId,
        deferredPromptId,
        promptId,
      });
    });
  }

  // Returns the new turnId, or null if the input was buffered as a steer
  // message or the turn was marked as resuming.
  steer(
    input: readonly ContentPart[],
    origin: PromptOrigin = USER_PROMPT_ORIGIN,
    expectedTurnId?: number,
    expectedPromptId?: string,
    requireActive = false,
  ): number | null {
    if (this.closed) {
      if (expectedTurnId !== undefined || expectedPromptId !== undefined || requireActive) {
        throw new KimiError(ErrorCodes.SESSION_CLOSED, 'Cannot steer after the agent has closed');
      }
      return null;
    }
    const active = this.activeTurn;
    const steerAdmissionClosed =
      active !== null && active !== 'resuming' && active.terminalizing;
    if (
      steerAdmissionClosed &&
      (origin.kind === 'user' ||
        expectedTurnId !== undefined ||
        expectedPromptId !== undefined ||
        requireActive)
    ) {
      throw new KimiError(
        ErrorCodes.TURN_AGENT_BUSY,
        expectedPromptId !== undefined
          ? `Cannot steer prompt ${expectedPromptId} because it is no longer active`
          : expectedTurnId !== undefined
            ? `Cannot steer turn ${String(expectedTurnId)} because it is no longer active`
            : 'Cannot steer because the active turn is ending',
        {
          details: {
            expectedTurnId,
            activeTurnId: active.turnId,
            expectedPromptId,
            activePromptId: active.promptId,
          },
        },
      );
    }
    if (expectedTurnId !== undefined || expectedPromptId !== undefined || requireActive) {
      const activeTurnId = active === null || active === 'resuming' ? undefined : this.currentId;
      const activePromptId =
        active === null
          ? this.deferredPrompts[0]?.promptId
          : active === 'resuming'
            ? undefined
            : active.promptId;
      if (
        (requireActive && activeTurnId === undefined && activePromptId === undefined) ||
        (expectedTurnId !== undefined && activeTurnId !== expectedTurnId) ||
        (expectedPromptId !== undefined && activePromptId !== expectedPromptId)
      ) {
        throw new KimiError(
          ErrorCodes.TURN_AGENT_BUSY,
          expectedPromptId !== undefined
            ? `Cannot steer prompt ${expectedPromptId} because it is no longer active`
            : expectedTurnId !== undefined
              ? `Cannot steer turn ${String(expectedTurnId)} because it is no longer active`
              : 'Cannot steer because no prompt is active',
          { details: { expectedTurnId, activeTurnId, expectedPromptId, activePromptId } },
        );
      }
    }
    // Same format gate as prompt() — steer input enters the history too.
    const gated = gateImageFormatParts(input);
    // Direct user steers (the public session.steer() path) do not carry an
    // expectedPromptId, but they still belong to the prompt being interrupted.
    // Bind them to that logical owner so Esc/cancel cannot leak them into the
    // next prompt. Background/cron origins intentionally remain unowned.
    const inferredPromptId =
      active === null
        ? this.deferredPrompts[0]?.promptId
        : active === 'resuming'
          ? undefined
          : active.promptId;
    const ownerPromptId =
      expectedPromptId ?? (origin.kind === 'user' ? inferredPromptId : undefined);
    const hasUserOwner =
      origin.kind === 'user' || expectedPromptId !== undefined || expectedTurnId !== undefined;
    const ownerTurnId =
      hasUserOwner && active !== null && active !== 'resuming' ? active.turnId : undefined;
    const ownerDeferredPromptId =
      hasUserOwner && active === null ? this.deferredPrompts[0]?.id : undefined;
    const admission = { kind: 'steer', id: randomUUID() } as const;
    // Buffer while a turn is active OR a manual compaction holds the context;
    // `onCompactionFinished` replays the buffer once compaction's full lifecycle
    // (summary + reinjection) is done. Returning null means "buffered" — which is
    // exactly what fire-and-forget callers (background notifications, cron) assume.
    if (
      this.activeTurn ||
      this.agent.fullCompaction.isCompacting ||
      this.deferredPrompts.length > 0
    ) {
      // Install the reservation before persistence. A synchronous persistence
      // observer may cancel the owner; it must be able to find and remove this
      // steer before the outer admission returns success.
      const reservation: BufferedSteer = {
        admissionId: admission.id,
        input: gated,
        origin,
        expectedPromptId: ownerPromptId,
        ownerTurnId,
        ownerDeferredPromptId,
      };
      this.steerBuffer.push(reservation);
      try {
        this.agent.records.logRecord({
          type: 'turn.steer',
          input: gated,
          origin,
          admissionId: admission.id,
          expectedPromptId: ownerPromptId,
          ownerTurnId,
          ownerDeferredPromptId,
        });
      } catch (error) {
        if (agentRecordAppendAccepted(error) !== true) {
          const index = this.steerBuffer.indexOf(reservation);
          if (index !== -1) this.steerBuffer.splice(index, 1);
          throw error;
        }
        this.bestEffortWarn('steer admission committed with observer failure', { error });
      }
      if (!this.steerBuffer.includes(reservation)) {
        throw new KimiError(
          this.closed ? ErrorCodes.SESSION_CLOSED : ErrorCodes.TURN_AGENT_BUSY,
          this.closed
            ? 'Cannot steer after the agent has closed'
            : 'Cannot steer because cancellation won the admission race',
          {
            details: {
              expectedTurnId,
              expectedPromptId,
              ownerTurnId,
              ownerDeferredPromptId,
            },
          },
        );
      }
      return null;
    }
    // Idle steer becomes a standalone turn. Persist from startTurn's accepted
    // callback so the active reservation exists before any synchronous record
    // observer can re-enter cancel/prompt.
    const result = this.launch(gated, origin, undefined, admission, (_deferredPromptId, turnId) => {
      this.agent.records.logRecord({
        type: 'turn.steer',
        input: gated,
        origin,
        admissionId: admission.id,
        turnId,
        expectedPromptId: ownerPromptId,
        ownerTurnId,
        ownerDeferredPromptId,
      });
    });
    return result.kind === 'started' ? result.turnId : null;
  }

  retry(trigger?: string): number | null {
    return this.prompt([], { kind: 'retry', trigger });
  }

  private launch(
    input: readonly ContentPart[],
    origin: PromptOrigin,
    promptId?: string,
    admission?: Pick<TurnInputConsumption, 'kind' | 'id'>,
    onAccepted?: (deferredPromptId: string | undefined, turnId: number | undefined) => void,
  ): PromptLaunchResult {
    if (this.closed) {
      return {
        kind: 'busy',
        error: makeErrorPayload(
          ErrorCodes.SESSION_CLOSED,
          'Cannot launch a turn after the agent has closed',
        ),
      };
    }
    if (this.activeTurn) {
      const error = makeErrorPayload(
        ErrorCodes.TURN_AGENT_BUSY,
        `Cannot launch a new turn while another turn (ID ${this.turnId}) is active`,
        { details: { turnId: this.turnId } },
      );
      this.agent.emitEvent({
        type: 'error',
        ...error,
      });
      return { kind: 'busy', error };
    }
    if (this.deferredPrompts.length > 0) {
      const error = makeErrorPayload(
        ErrorCodes.TURN_AGENT_BUSY,
        'Cannot launch a new turn while another prompt is waiting for compaction to finish',
        { details: { deferredPromptId: this.deferredPrompts[0]!.id } },
      );
      this.agent.emitEvent({ type: 'error', ...error });
      return { kind: 'busy', error };
    }

    // While a manual/SDK compaction holds the context, defer the launch instead
    // of rejecting it: persist a separately identified prompt and replay it
    // from `onCompactionFinished`
    // once compaction's full lifecycle (summary + reinjection) completes. The
    // deferred turn's eventual `turn.started` lets PromptService associate the
    // pending prompt, so a prompt submitted mid-compaction completes normally
    // rather than getting stuck "running". (Auto compaction runs inside an active
    // turn, so the `activeTurn` check above already covers it.)
    if (this.agent.fullCompaction.isCompacting) {
      const deferredPromptId = randomUUID();
      const reservation = {
        id: deferredPromptId,
        admissionId: admission?.id,
        input,
        origin,
        promptId,
      };
      this.deferredPrompts.push(reservation);
      try {
        onAccepted?.(deferredPromptId, undefined);
      } catch (error) {
        if (agentRecordAppendAccepted(error) !== true) {
          const index = this.deferredPrompts.indexOf(reservation);
          if (index !== -1) this.deferredPrompts.splice(index, 1);
          throw error;
        }
        this.bestEffortWarn('deferred prompt admission committed with observer failure', {
          error,
        });
      }
      if (!this.deferredPrompts.includes(reservation)) {
        throw new KimiError(
          this.closed ? ErrorCodes.SESSION_CLOSED : ErrorCodes.TURN_AGENT_BUSY,
          this.closed
            ? 'Cannot prompt after the agent has closed'
            : 'Cannot prompt because cancellation won the admission race',
          { details: { promptId, deferredPromptId } },
        );
      }
      return { kind: 'deferred', deferredPromptId };
    }

    return this.startTurn(input, origin, undefined, promptId, admission, onAccepted);
  }

  private startTurn(
    input: readonly ContentPart[],
    origin: PromptOrigin,
    deferredPromptId?: string,
    promptId?: string,
    admission?: Pick<TurnInputConsumption, 'kind' | 'id'>,
    onAccepted?: (deferredPromptId: string | undefined, turnId: number | undefined) => void,
  ): PromptResult {
    // Per-turn setup (telemetry, usage window, `turn.started`, appending the
    // prompt) now lives in `runOneTurn`, so a goal-driven run emits a clean
    // start/end pair per continuation turn rather than one mega-turn.
    const turnId = this.allocateTurnId();
    const controller = new AbortController();
    const promise = createControlledPromise<TurnEndResult>();
    const firstRequest = createControlledPromise<void>();
    const reservation: ActiveTurn = {
      turnId,
      promptId,
      admission,
      controller,
      promise,
      firstRequest,
      admitting: true,
      terminalizing: false,
      steerFlushFailed: false,
      steerFlushError: undefined,
    };
    this.activeTurn = reservation;

    void firstRequest.catch(() => undefined);
    void promise.then(firstRequest.reject, firstRequest.reject);
    try {
      // A previous abnormal turn may have durably reserved, but not yet
      // appended, its outcome reminder. The active reservation is installed
      // first so a synchronous persistence observer cannot re-enter launch and
      // overtake or duplicate this convergence barrier.
      try {
        this.materializePendingOutcomes();
      } catch (error) {
        if (agentRecordAppendAccepted(error) !== true) throw error;
        this.bestEffortWarn('turn outcome reminder committed with observer failure', { error });
      }
      if (this.closed) {
        throw new KimiError(ErrorCodes.SESSION_CLOSED, 'Cannot launch after the agent has closed');
      }
      // The prompt admission is the next durable operation. From this point a
      // synchronous observer may cancel: its record will necessarily follow
      // the prompt/activation record it owns.
      reservation.admitting = false;
      // The active reservation is already installed, but no turn event has
      // fired yet. Persistence callbacks may re-enter prompt/cancel safely.
      if (deferredPromptId !== undefined) {
        try {
          this.agent.records.logRecord({
            type: 'turn.deferred_prompt_started',
            deferredPromptId,
            turnId,
          });
        } catch (error) {
          if (agentRecordAppendAccepted(error) !== true) throw error;
          this.bestEffortWarn('deferred prompt activation committed with observer failure', {
            error,
            turnId,
          });
        }
      }
      try {
        onAccepted?.(undefined, turnId);
      } catch (error) {
        if (agentRecordAppendAccepted(error) !== true) throw error;
        this.bestEffortWarn('turn admission committed with observer failure', { error, turnId });
      }
    } catch (error) {
      const active = this.activeTurn;
      if (active !== null && active.controller === controller) {
        this.activeTurn = null;
        this.turnId -= 1;
      }
      controller.abort(error);
      promise.reject(error);
      throw error;
    }
    const worker = this.turnWorker(
      turnId,
      input,
      origin,
      controller.signal,
      deferredPromptId,
      promptId,
      admission,
    );
    void worker.then(promise.resolve, promise.reject);

    return { kind: 'started', turnId };
  }

  /** Allocates the next monotonic turn id. */
  private allocateTurnId(): number {
    this.turnId += 1;
    return this.turnId;
  }

  restorePrompt(
    input: readonly ContentPart[],
    origin: PromptOrigin,
    deferredPromptId?: string,
    promptId?: string,
    admissionId?: string,
    turnId?: number,
  ): void {
    if (admissionId !== undefined) {
      if (!this.deferredPrompts.some((prompt) => prompt.admissionId === admissionId)) {
        this.deferredPrompts.push({
          id: deferredPromptId,
          admissionId,
          turnId,
          input,
          origin,
          promptId,
        });
      }
      if (turnId !== undefined) this.observeRestoredTurnId(turnId);
      return;
    }
    if (deferredPromptId !== undefined) {
      if (!this.deferredPrompts.some((prompt) => prompt.id === deferredPromptId)) {
        this.deferredPrompts.push({ id: deferredPromptId, input, origin, promptId });
      }
      return;
    }
    this.restoreStartedTurn();
  }

  /** Marks a replayed deferred prompt as durably activated, even when its input is empty. */
  restoreDeferredPromptStarted(deferredPromptId: string, turnId: number): void {
    const index = this.deferredPrompts.findIndex((prompt) => prompt.id === deferredPromptId);
    const prompt = index === -1 ? undefined : this.deferredPrompts[index];
    if (prompt?.admissionId !== undefined) {
      // Activation is intentionally not consumption: a crash after this record
      // but before context append must retry the admitted prompt. The context
      // record's consumption marker is the only durable acknowledgement.
      prompt.turnId = turnId;
      this.observeRestoredTurnId(turnId);
      return;
    }
    if (index !== -1) this.deferredPrompts.splice(index, 1);
    // Activation means every steer queued behind this deferred prompt now
    // belongs to the started turn. Do not replay it as fresh work after a
    // crash; subsequent context records contain anything that was consumed.
    for (const steer of this.steerBuffer) steer.restorePending = false;
    this.observeRestoredTurnId(turnId);
    this.activeTurn ??= 'resuming';
  }

  private restoreStartedTurn(): void {
    if (this.activeTurn) {
      return;
    }
    this.turnId += 1;
    this.activeTurn = 'resuming';
  }

  /**
   * Raise the turn counter to cover a turnId observed in a replayed loop event.
   * This is the authoritative source of the restored counter: every turn that
   * ran — a prompted turn, a goal continuation, or a steer-launched turn —
   * emits loop events carrying its real turnId, even though only prompted turns
   * write a `turn.prompt` record. Resuming then continues from `max + 1`. Only
   * ever raises the counter, never lowers it, so the live path (where `turnId`
   * is already allocated before any loop event) is unaffected.
   */
  observeRestoredTurnId(turnId: number): void {
    if (Number.isInteger(turnId) && turnId > this.turnId) {
      this.turnId = turnId;
    }
  }

  restoreSteer(
    input: readonly ContentPart[],
    origin: PromptOrigin,
    expectedPromptId?: string,
    ownerTurnId?: number,
    ownerDeferredPromptId?: string,
    admissionId?: string,
    turnId?: number,
  ): void {
    if (admissionId !== undefined) {
      if (!this.steerBuffer.some((steer) => steer.admissionId === admissionId)) {
        this.steerBuffer.push({
          admissionId,
          input,
          origin,
          expectedPromptId,
          ownerTurnId,
          ownerDeferredPromptId,
        });
      }
      if (turnId !== undefined) this.observeRestoredTurnId(turnId);
      return;
    }
    if (this.activeTurn || this.deferredPrompts.length > 0) {
      this.steerBuffer.push({
        input,
        origin,
        expectedPromptId,
        ownerTurnId,
        ownerDeferredPromptId,
        restorePending: this.deferredPrompts.length > 0,
      });
      return;
    }
    this.turnId += 1;
    this.activeTurn = 'resuming';
  }

  /** Apply a durable admission acknowledgement during live append or replay. */
  consumeTurnInput(consumption: TurnInputConsumption): void {
    this.observeRestoredTurnId(consumption.turnId);
    if (consumption.kind === 'prompt') {
      this.deferredPrompts = this.deferredPrompts.filter(
        (prompt) => prompt.admissionId !== consumption.id,
      );
      return;
    }
    this.steerBuffer = this.steerBuffer.filter(
      (steer) => steer.admissionId !== consumption.id,
    );
  }

  restoreOutcome(id: string, turnId: number, content: string): void {
    this.observeRestoredTurnId(turnId);
    this.pendingOutcomes.set(id, { id, turnId, content });
  }

  consumeOutcome(id: string): void {
    this.pendingOutcomes.delete(id);
  }

  /** A destructive user context operation also tombstones unseen outcomes. */
  clearPendingOutcomes(): void {
    this.pendingOutcomes.clear();
  }

  cancel(
    turnId?: number,
    reason?: unknown,
    options: TurnCancelOptions = {},
  ): Promise<void> {
    if (this.closed) return this.shutdownPromise ?? Promise.resolve();
    if (this.activeTurn === 'resuming' && this.agent.records.restoring === null) {
      return Promise.resolve();
    }
    if (
      this.activeTurn !== null &&
      this.activeTurn !== 'resuming' &&
      this.activeTurn.admitting
    ) {
      // The caller has not yet received an admission acknowledgement and no
      // owner record exists to order a cancellation after. Treat this tiny
      // launch window as not-yet-active; once admission is durable, ordinary
      // cancellation is allowed synchronously from its persistence observer.
      return Promise.resolve();
    }
    if (options.restoredPromptId === undefined) {
      this.assertCancellationTarget(turnId, options.expectedPromptId, options.requireActive);
    }
    if (this.cancelSetupPromise !== null) return this.cancelSetupPromise;
    if (
      this.activeTurn !== null &&
      this.activeTurn !== 'resuming' &&
      this.activeTurn.terminalizing
    ) {
      return this.activeTurn.promise.then(() => undefined);
    }

    // Install a synchronous re-entrancy guard before logging. Record
    // persistence hooks are allowed to call back into the engine; a nested
    // cancel must join the outer operation instead of recursively logging.
    const completion = createControlledPromise<void>();
    this.cancelSetupPromise = completion;
    void completion.catch(() => undefined);
    const releaseSetup = (): void => {
      if (this.cancelSetupPromise === completion) this.cancelSetupPromise = null;
    };
    void completion.then(releaseSetup, releaseSetup);
    try {
      const barrier = this.startCancellation(
        turnId,
        reason,
        options.restoredPromptId,
        options.restoredOwnerTurnId,
        options.restoredOwnerDeferredPromptId,
        options.restoredOutcomeId,
        options.restoredOutcomeTurnId,
        options.restoredOutcomeContent,
        options.restoredCancelledTurnInputs,
      );
      void barrier.then(completion.resolve, completion.reject);
    } catch (error) {
      completion.reject(error);
    }
    return completion;
  }

  assertCancellationTarget(
    turnId?: number,
    expectedPromptId?: string,
    requireActive = false,
  ): void {
    if (expectedPromptId === undefined && !requireActive) return;
    const active = this.activeTurn;
    const activeTurnId = active === null || active === 'resuming' ? undefined : this.currentId;
    const activePromptId =
      active === null
        ? this.deferredPrompts[0]?.promptId
        : active === 'resuming'
          ? undefined
          : active.promptId;
    const hasLogicalPrompt =
      (active !== null && active !== 'resuming') ||
      (active === null && this.deferredPrompts.length > 0);
    const ownsTurn = turnId === undefined || this.isActiveTurn(turnId);
    if (
      (requireActive && !hasLogicalPrompt) ||
      !ownsTurn ||
      (expectedPromptId !== undefined && activePromptId !== expectedPromptId)
    ) {
      throw new KimiError(
        ErrorCodes.TURN_AGENT_BUSY,
        expectedPromptId !== undefined
          ? `Cannot cancel prompt ${expectedPromptId} because it is no longer active`
          : 'Cannot cancel because no prompt is active',
        { details: { expectedTurnId: turnId, activeTurnId, expectedPromptId, activePromptId } },
      );
    }
  }

  private startCancellation(
    turnId?: number,
    reason?: unknown,
    restoredPromptId?: string,
    restoredOwnerTurnId?: number,
    restoredOwnerDeferredPromptId?: string,
    restoredOutcomeId?: string,
    restoredOutcomeTurnId?: number,
    restoredOutcomeContent?: string,
    restoredCancelledTurnInputs?: readonly TurnInputConsumption[],
  ): Promise<void> {
    const ownsActiveTurn =
      turnId === undefined || turnId === this.currentId || this.activeWorkerOwnsTurn(turnId);
    const active = this.activeTurn;
    const terminalizingBeforeCancellation =
      active === null || active === 'resuming' ? undefined : active.terminalizing;
    const canCreateCancellationOutcome =
      ownsActiveTurn &&
      active !== null &&
      active !== 'resuming' &&
      !active.terminalizing;
    if (ownsActiveTurn && active !== null && active !== 'resuming') {
      active.terminalizing = true;
    }
    const inferredPromptId =
      restoredPromptId ??
      (ownsActiveTurn && active !== null && active !== 'resuming'
        ? active.promptId
        : ownsActiveTurn && active === null && turnId === undefined
          ? this.deferredPrompts[0]?.promptId
          : undefined);
    const inferredOwnerTurnId =
      restoredOwnerTurnId ??
      (ownsActiveTurn && active !== null && active !== 'resuming' ? active.turnId : undefined);
    const inferredOwnerDeferredPromptId =
      restoredOwnerDeferredPromptId ??
      (ownsActiveTurn && active === null && turnId === undefined
        ? this.deferredPrompts[0]?.id
        : undefined);
    const cancelledTurnInputs =
      restoredCancelledTurnInputs ??
      this.collectIncompleteTurnInputsOwnedBy(
        ownsActiveTurn,
        inferredPromptId,
        inferredOwnerTurnId,
        inferredOwnerDeferredPromptId,
      );
    // A direct cancel (RPC / replay) is the user pressing stop. When the cancel
    // is propagated from an aborting signal (e.g. a subagent's deadline via
    // waitForCurrentTurn), carry that original reason instead so a timeout is
    // not mislabeled to the model as a deliberate user interruption.
    const cancelReason = reason ?? userCancellationReason();
    let outcome: PendingTurnOutcome | undefined;
    let outcomeReservedByCancellation = false;
    if (
      restoredOutcomeId !== undefined &&
      restoredOutcomeTurnId !== undefined &&
      restoredOutcomeContent !== undefined
    ) {
      outcome = {
        id: restoredOutcomeId,
        turnId: restoredOutcomeTurnId,
        content: restoredOutcomeContent,
      };
      outcomeReservedByCancellation = !this.pendingOutcomes.has(outcome.id);
      this.pendingOutcomes.set(outcome.id, outcome);
    } else if (ownsActiveTurn && active !== null && active !== 'resuming') {
      // The reminder intent is part of the cancel record itself. A process may
      // die after that record is durable but before the worker appends the
      // reminder to context; replay can therefore finish the materialization
      // before releasing any pending follow-up prompt.
      try {
        outcome = [...this.pendingOutcomes.values()].find(
          (candidate) => candidate.turnId === this.currentId,
        );
        if (outcome === undefined && canCreateCancellationOutcome) {
          outcome = {
            id: randomUUID(),
            turnId: this.currentId,
            content: renderTurnCancellationReminder(
              cancelReason,
              isUserCancellation(cancelReason),
            ),
          };
        }
        if (outcome !== undefined) {
          outcomeReservedByCancellation = !this.pendingOutcomes.has(outcome.id);
          this.pendingOutcomes.set(outcome.id, outcome);
        }
      } catch (error) {
        // Rendering diagnostics must never prevent the cancellation itself.
        this.bestEffortWarn('failed to render turn cancellation reminder', { error });
      }
    }
    let recordFailed = false;
    let recordError: unknown;
    try {
      this.agent.records.logRecord({
        type: 'turn.cancel',
        turnId,
        promptId: inferredPromptId,
        ownerTurnId: inferredOwnerTurnId,
        ownerDeferredPromptId: inferredOwnerDeferredPromptId,
        outcomeId: outcome?.id,
        outcomeTurnId: outcome?.turnId,
        outcomeContent: outcome?.content,
        cancelledTurnInputs:
          cancelledTurnInputs.length === 0 ? undefined : cancelledTurnInputs,
      });
    } catch (error) {
      if (agentRecordAppendAccepted(error) === false) {
        // Nothing durable owns the destructive transition. Restore every
        // reservation made before persistence and leave the admitted work
        // running; otherwise replay would execute a prompt that live state
        // had already aborted or tombstoned.
        if (outcomeReservedByCancellation && outcome !== undefined) {
          this.pendingOutcomes.delete(outcome.id);
        }
        if (
          ownsActiveTurn &&
          active !== null &&
          active !== 'resuming' &&
          this.activeTurn === active &&
          terminalizingBeforeCancellation !== undefined
        ) {
          active.terminalizing = terminalizingBeforeCancellation;
        }
        return Promise.reject(error);
      }
      recordFailed = true;
      recordError = error;
    }
    try {
      this.agent.context.cancelTurnInputExpansions(cancelledTurnInputs);
    } catch (error) {
      recordFailed = true;
      recordError ??= error;
    }
    if (
      turnId !== undefined &&
      !ownsActiveTurn
    ) {
      return recordFailed
        ? Promise.reject(recordError)
        : Promise.resolve(); // Ignore cancel for non-active turn
    }
    // A prompt accepted during manual compaction has no turn id yet. Only an
    // unscoped user stop may own that pending prompt; a delayed explicit id can
    // belong to an older worker. Steers live in their own buffer and survive —
    // cancelling a user prompt must not erase a background notification.
    if (
      !this.hasActiveTurn &&
      turnId === undefined &&
      isUserCancellation(cancelReason)
    ) {
      const cancelled = this.deferredPrompts.shift();
      this.discardSteersOwnedBy(
        inferredPromptId ?? cancelled?.promptId,
        inferredOwnerTurnId,
        inferredOwnerDeferredPromptId ?? cancelled?.id,
      );
      this.discardPromptsOwnedBy(
        inferredPromptId ?? cancelled?.promptId,
        inferredOwnerTurnId,
        inferredOwnerDeferredPromptId ?? cancelled?.id,
      );
    } else {
      this.discardSteersOwnedBy(
        inferredPromptId,
        inferredOwnerTurnId,
        inferredOwnerDeferredPromptId,
      );
      this.discardPromptsOwnedBy(
        inferredPromptId,
        inferredOwnerTurnId,
        inferredOwnerDeferredPromptId,
      );
    }
    this.abortTurn(cancelReason);
    let subagentCancellation: Promise<void>;
    try {
      subagentCancellation = this.agent.subagentHost?.cancelAll(cancelReason) ?? Promise.resolve();
    } catch (error) {
      subagentCancellation = Promise.reject(error);
    }
    return Promise.allSettled([subagentCancellation]).then((settlements) => {
      if (recordFailed) throw recordError;
      const rejected = settlements.find(
        (result): result is PromiseRejectedResult => result.status === 'rejected',
      );
      if (rejected !== undefined) throw rejected.reason;
    });
  }

  /** Permanently stop turn/compaction work before the owning session is disposed. */
  shutdown(reason: unknown): Promise<void> {
    if (this.shutdownPromise !== null) return this.shutdownPromise;

    const completion = createControlledPromise<void>();
    this.shutdownPromise = completion;
    void completion.catch(() => undefined);

    const active =
      this.activeTurn !== null && this.activeTurn !== 'resuming'
        ? this.activeTurn
        : undefined;
    const admittedActive = active?.admitting === false ? active : undefined;
    const deferred = active === undefined ? this.deferredPrompts[0] : undefined;
    const cancelledTurnInputs =
      admittedActive === undefined
        ? []
        : this.collectIncompleteTurnInputsOwnedBy(
            true,
            admittedActive.promptId,
            admittedActive.turnId,
            undefined,
          );
    const closedBeforeShutdown = this.closed;
    const deferredPromptsBeforeShutdown = [...this.deferredPrompts];
    const steerBufferBeforeShutdown = [...this.steerBuffer];
    this.closed = true;
    this.deferredPrompts.length = 0;
    this.steerBuffer.length = 0;
    let outcome: PendingTurnOutcome | undefined;
    let outcomeReservedByShutdown = false;
    if (admittedActive !== undefined) {
      try {
        outcome = [...this.pendingOutcomes.values()].find(
          (candidate) => candidate.turnId === this.currentId,
        );
        if (outcome === undefined && !admittedActive.terminalizing) {
          outcome = {
            id: randomUUID(),
            turnId: this.currentId,
            content: renderTurnCancellationReminder(reason, isUserCancellation(reason)),
          };
        }
        if (outcome !== undefined) {
          outcomeReservedByShutdown = !this.pendingOutcomes.has(outcome.id);
          this.pendingOutcomes.set(outcome.id, outcome);
        }
      } catch (error) {
        this.bestEffortWarn('failed to render turn shutdown reminder', { error });
      }
    }
    let recordFailed = false;
    let recordError: unknown;
    const cleanup: Promise<unknown>[] = [];
    if (active !== undefined) cleanup.push(active.promise);
    if (admittedActive !== undefined || deferred !== undefined) {
      try {
        this.agent.records.logRecord({
          type: 'turn.cancel',
          promptId: admittedActive?.promptId ?? deferred?.promptId,
          ownerTurnId: admittedActive?.turnId,
          ownerDeferredPromptId: deferred?.id,
          outcomeId: outcome?.id,
          outcomeTurnId: outcome?.turnId,
          outcomeContent: outcome?.content,
          cancelledTurnInputs:
            cancelledTurnInputs.length === 0 ? undefined : cancelledTurnInputs,
        });
      } catch (error) {
        if (agentRecordAppendAccepted(error) === false) {
          // A rejected shutdown record cannot authorize any local teardown.
          // Restore the exact pre-shutdown reservations so the live worker and
          // a future replay continue from the same admitted state.
          if (outcomeReservedByShutdown && outcome !== undefined) {
            this.pendingOutcomes.delete(outcome.id);
          }
          this.closed = closedBeforeShutdown;
          this.deferredPrompts = deferredPromptsBeforeShutdown;
          this.steerBuffer = steerBufferBeforeShutdown;
          if (this.shutdownPromise === completion) this.shutdownPromise = null;
          completion.reject(error);
          return completion;
        }
        recordFailed = true;
        recordError = error;
      }
    }
    try {
      this.agent.context.cancelTurnInputExpansions(cancelledTurnInputs);
    } catch (error) {
      recordFailed = true;
      recordError ??= error;
    }
    if (active !== undefined) {
      try {
        active.controller.abort(reason);
      } catch (error) {
        cleanup.push(Promise.reject(error));
      }
    } else if (this.activeTurn === 'resuming') {
      this.activeTurn = null;
    }

    try {
      cleanup.push(this.agent.fullCompaction.cancel());
    } catch (error) {
      cleanup.push(Promise.reject(error));
    }
    try {
      cleanup.push(this.agent.subagentHost?.cancelAll(reason) ?? Promise.resolve());
    } catch (error) {
      cleanup.push(Promise.reject(error));
    }
    const barrier = Promise.allSettled(cleanup).then((settlements) => {
      if (recordFailed) throw recordError;
      const rejected = settlements.find(
        (result): result is PromiseRejectedResult => result.status === 'rejected',
      );
      if (rejected !== undefined) throw rejected.reason;
    });
    void barrier.then(completion.resolve, completion.reject);
    return completion;
  }

  /**
   * Whether the requested turn id belongs to the worker that is active now.
   * Goal continuations allocate multiple ids inside one worker, so callers
   * must use this predicate both when cancelling and when capturing the
   * worker's terminal barrier.
   */
  isActiveTurn(turnId?: number): boolean {
    if (!this.hasActiveTurn) return false;
    return turnId === undefined || this.activeWorkerOwnsTurn(turnId);
  }

  /**
   * A goal drive keeps one worker alive while allocating a new id for each
   * continuation turn. An RPC issued for an earlier turn in that same worker
   * must still cancel it, but the same stale id must not cancel a replacement
   * worker launched after the original worker settled.
   */
  private activeWorkerOwnsTurn(turnId: number): boolean {
    const active = this.activeTurn;
    return (
      active !== null &&
      active !== 'resuming' &&
      turnId >= active.turnId &&
      turnId <= this.currentId
    );
  }

  get currentId() {
    return this.turnId;
  }

  get hasActiveTurn(): boolean {
    return this.activeTurn !== null && this.activeTurn !== 'resuming';
  }

  get isClosed(): boolean {
    return this.closed;
  }

  private ensureActiveTurn(): ActiveTurn {
    if (this.activeTurn === null || this.activeTurn === 'resuming') {
      throw new Error('No active turn');
    }
    return this.activeTurn;
  }

  waitForCurrentTurn(signal?: AbortSignal): Promise<TurnEndResult> {
    const active = this.ensureActiveTurn();
    if (signal === undefined) return active.promise;

    const onAbort = (): void => {
      // A goal drive reuses one active worker across multiple turn IDs. Cancel
      // that captured worker rather than the ID that happened to be current
      // when this wait began, while also avoiding a later replacement turn.
      if (this.activeTurn === active) {
        void this.agent.turn.cancel(undefined, signal.reason).catch((error: unknown) => {
          this.bestEffortError('failed to cancel turn after waiter abort', { error });
        });
      }
    };
    if (signal.aborted) {
      onAbort();
      signal.throwIfAborted();
    }
    signal.addEventListener('abort', onAbort, { once: true });

    return abortable(active.promise, signal).finally(() => {
      signal.removeEventListener('abort', onAbort);
    });
  }

  waitForTurnFirstRequest(): Promise<void> {
    return this.ensureActiveTurn().firstRequest;
  }

  private abortTurn(reason: unknown): void {
    if (this.activeTurn === 'resuming') {
      this.activeTurn = null;
      return;
    }
    if (this.activeTurn === null) return;

    // Keep the turn active until its worker has closed any open tool exchange,
    // appended the terminal outcome reminder, and emitted turn.ended. This
    // lets the RPC cancellation wait on a true terminal barrier and prevents a
    // follow-up turn from overlapping the cancelled turn's asynchronous cleanup.
    const cancelledTurn = this.activeTurn;
    // The reason (a user cancellation by default, or the originating signal's
    // reason when propagated) travels as signal.reason so tools settling on
    // this signal can report a deliberate user interruption distinctly from a
    // timeout/system abort. linkAbortSignal forwards it to linked subagents.
    cancelledTurn.controller.abort(reason);
  }

  private flushSteerBuffer(): boolean {
    if (this.steerBuffer.length === 0) return false;
    while (this.steerBuffer.length > 0) {
      const steer = this.steerBuffer[0]!;
      try {
        this.agent.context.appendUserMessage(
          steer.input,
          steer.origin,
          steer.admissionId === undefined
            ? undefined
            : { kind: 'steer', id: steer.admissionId, turnId: this.currentId },
        );
      } catch (error) {
        const active = this.activeTurn;
        if (active !== null && active !== 'resuming') {
          active.steerFlushFailed = true;
          active.steerFlushError = error;
        }
        throw error;
      }
      // Remove only after append succeeds. If a later item fails, earlier
      // committed items are not replayed by a retry. A persistence observer
      // may synchronously cancel and replace the buffer while append runs, so
      // remove the captured item by identity rather than shifting a new head.
      const index = this.steerBuffer.indexOf(steer);
      if (index !== -1) this.steerBuffer.splice(index, 1);
    }
    return true;
  }

  private sealAndFlushSteers(turnId: number): { error: unknown } | undefined {
    this.closeSteerAdmission(turnId);
    const active = this.activeTurn;
    if (active !== null && active !== 'resuming' && active.steerFlushFailed) {
      this.steerBuffer.length = 0;
      return { error: active.steerFlushError };
    }
    try {
      this.flushSteerBuffer();
      return undefined;
    } catch (error) {
      // A terminal worker cannot carry unpersisted input into a later prompt:
      // that would reorder it behind newer user input. The durable turn.steer
      // record remains available for diagnostics, while this turn reports the
      // persistence failure and still emits its terminal event.
      this.steerBuffer.length = 0;
      return { error };
    }
  }

  private closeSteerAdmission(turnId: number): void {
    const active = this.activeTurn;
    if (
      active !== null &&
      active !== 'resuming' &&
      this.activeWorkerOwnsTurn(turnId)
    ) {
      active.terminalizing = true;
    }
  }

  private discardSteersOwnedBy(
    promptId: string | undefined,
    ownerTurnId: number | undefined,
    ownerDeferredPromptId: string | undefined,
  ): void {
    if (
      promptId === undefined &&
      ownerTurnId === undefined &&
      ownerDeferredPromptId === undefined
    ) {
      return;
    }
    this.steerBuffer = this.steerBuffer.filter(
      (steer) =>
        !this.isSteerOwnedBy(steer, promptId, ownerTurnId, ownerDeferredPromptId),
    );
  }

  private collectIncompleteTurnInputsOwnedBy(
    ownsActiveTurn: boolean,
    promptId: string | undefined,
    ownerTurnId: number | undefined,
    ownerDeferredPromptId: string | undefined,
  ): TurnInputConsumption[] {
    const inputs = new Map<string, TurnInputConsumption>();
    const add = (kind: TurnInputConsumption['kind'], id: string | undefined): void => {
      if (id === undefined) return;
      inputs.set(`${kind}:${id}`, { kind, id, turnId: this.currentId });
    };
    const active = this.activeTurn;
    if (ownsActiveTurn && active !== null && active !== 'resuming') {
      if (
        active.admission !== undefined &&
        this.agent.context.hasIncompleteTurnInput(active.admission)
      ) {
        add(active.admission.kind, active.admission.id);
      }
    }
    for (const steer of this.steerBuffer) {
      if (
        steer.admissionId !== undefined &&
        this.isSteerOwnedBy(steer, promptId, ownerTurnId, ownerDeferredPromptId) &&
        this.agent.context.hasIncompleteTurnInput({ kind: 'steer', id: steer.admissionId })
      ) {
        add('steer', steer.admissionId);
      }
    }
    return [...inputs.values()];
  }

  private isSteerOwnedBy(
    steer: BufferedSteer,
    promptId: string | undefined,
    ownerTurnId: number | undefined,
    ownerDeferredPromptId: string | undefined,
  ): boolean {
    if (
      promptId === undefined &&
      ownerTurnId === undefined &&
      ownerDeferredPromptId === undefined
    ) {
      return false;
    }
    if (promptId !== undefined) return steer.expectedPromptId === promptId;
    if (ownerTurnId !== undefined) {
      return steer.expectedPromptId === undefined && steer.ownerTurnId === ownerTurnId;
    }
    return (
      steer.expectedPromptId === undefined &&
      steer.ownerDeferredPromptId === ownerDeferredPromptId
    );
  }

  private discardPromptsOwnedBy(
    promptId: string | undefined,
    ownerTurnId: number | undefined,
    ownerDeferredPromptId: string | undefined,
  ): void {
    if (
      promptId === undefined &&
      ownerTurnId === undefined &&
      ownerDeferredPromptId === undefined
    ) {
      return;
    }
    this.deferredPrompts = this.deferredPrompts.filter((prompt) => {
      if (promptId !== undefined) return prompt.promptId !== promptId;
      if (ownerTurnId !== undefined) return prompt.turnId !== ownerTurnId;
      return prompt.id !== ownerDeferredPromptId;
    });
  }

  /**
   * Replay inputs (prompts or steers) that were deferred while a manual compaction
   * held the context. Called by `FullCompaction` once the compaction lifecycle
   * (summary + reinjection) is done — and on cancel/failure — so deferred input is
   * never lost or stuck. If a turn is somehow already active (e.g. one that raced
   * and cancelled the compaction), let it consume the buffer like any other steer;
   * otherwise launch a fresh turn from the first buffered item, with the rest
   * draining into it via `flushSteerBuffer`.
   */
  onCompactionFinished(): void {
    if (this.closed) {
      this.deferredPrompts.length = 0;
      this.steerBuffer.length = 0;
      return;
    }
    // Resume may rebuild pending admissions before a committed compaction has
    // finished its post-commit recovery. Keep them parked behind that durable
    // reservation; converting a steer through launch() here would misclassify
    // its consumption as a prompt and could deliver it twice.
    if (this.agent.fullCompaction.isCompacting) return;
    if (this.startNextDeferredPrompt()) return;
    if (this.steerBuffer.length === 0) return;
    if (this.activeTurn !== null) {
      this.flushSteerBuffer();
      return;
    }
    const next = this.steerBuffer[0]!;
    if (next.admissionId === undefined) this.steerBuffer.shift();
    const result = this.launch(
      next.input,
      next.origin,
      undefined,
      next.admissionId === undefined
        ? undefined
        : { kind: 'steer', id: next.admissionId },
    );
    if (result.kind === 'busy' && next.admissionId === undefined) {
      this.steerBuffer.unshift(next);
    }
  }

  finishResume(): void {
    // Hold a synchronous launch/cancel reservation while converging outcome
    // acknowledgements. Persistence observers may re-enter public APIs; none
    // may overtake this barrier or recursively materialize the same outcome.
    this.activeTurn ??= 'resuming';
    // New records have explicit admission/consumption identities: every
    // remaining identified steer is genuinely pending. Preserve the old
    // heuristic only for legacy records that predate those identities.
    this.steerBuffer = this.steerBuffer.filter(
      (steer) => steer.admissionId !== undefined || steer.restorePending === true,
    );
    for (const steer of this.steerBuffer) steer.restorePending = false;
    // A crash can leave a durable cancellation/failure intent immediately
    // before its context append. Complete those appends before any recovered
    // prompt or steer is allowed to start a model request.
    this.materializePendingOutcomes();
    if (this.activeTurn === 'resuming') this.activeTurn = null;
    if (this.startNextDeferredPrompt()) return;
    this.onCompactionFinished();
  }

  private startNextDeferredPrompt(): boolean {
    if (
      this.closed ||
      this.activeTurn !== null ||
      this.agent.fullCompaction.isCompacting ||
      this.deferredPrompts.length === 0
    ) {
      return false;
    }
    const next = this.deferredPrompts[0]!;
    const legacy = next.admissionId === undefined;
    if (legacy) this.deferredPrompts.shift();
    try {
      this.startTurn(
        next.input,
        next.origin,
        next.id,
        next.promptId,
        next.admissionId === undefined
          ? undefined
          : { kind: 'prompt', id: next.admissionId },
      );
    } catch (error) {
      if (legacy && !this.closed) this.deferredPrompts.unshift(next);
      throw error;
    }
    return true;
  }

  /**
   * The body of the single in-flight `activeTurn`. Routes to the goal driver
   * (sequential continuation turns) when a goal is active, otherwise runs exactly
   * one turn. Clears `activeTurn` when the whole run finishes (identified by the
   * launch signal, so a superseding turn is never clobbered).
   */
  private async turnWorker(
    firstTurnId: number,
    input: readonly ContentPart[],
    origin: PromptOrigin,
    signal: AbortSignal,
    deferredPromptId?: string,
    promptId?: string,
    admission?: Pick<TurnInputConsumption, 'kind' | 'id'>,
  ): Promise<TurnEndResult> {
    const ownsActiveTurn = (): boolean =>
      this.activeTurn !== null &&
      this.activeTurn !== 'resuming' &&
      this.activeTurn.controller.signal === signal;
    try {
      const initialGoalStatus = this.agent.goal.getGoal().goal?.status;
      if (initialGoalStatus === 'active') {
        return await this.driveGoal(
          firstTurnId,
          input,
          origin,
          signal,
          deferredPromptId,
          promptId,
          admission,
        );
      }
      const end = await this.runOneTurn(
        firstTurnId,
        input,
        origin,
        signal,
        true,
        deferredPromptId,
        promptId,
        admission,
      );
      // A goal can become active during an ordinary turn: the model creates one
      // with CreateGoal, or resumes a paused/blocked goal via UpdateGoal. Either
      // way, hand the now-active goal to the driver so it is actually pursued,
      // instead of stopping after the turn that merely started it. (The
      // already-active case took the early return above.)
      const goalBecameActive = this.agent.goal.getGoal().goal?.status === 'active';
      if (
        goalBecameActive &&
        end.event.reason !== 'cancelled' &&
        end.event.reason !== 'failed' &&
        end.event.reason !== 'blocked'
      ) {
        // The ordinary turn created or resumed the goal, so it counts as the
        // first active goal turn before the continuation driver takes over.
        const countedGoal = await this.agent.goal.incrementTurn();
        if (countedGoal?.budget.overBudget === true) {
          await this.agent.goal.markBlocked({ reason: 'A configured budget was reached' });
          return end;
        }
        return await this.driveGoal(
          this.allocateTurnId(),
          [{ type: 'text', text: GOAL_CONTINUATION_PROMPT }],
          GOAL_CONTINUATION_ORIGIN,
          signal,
        );
      }
      return end;
    } finally {
      if (ownsActiveTurn()) {
        this.activeTurn = null;
        // A deferred/buffered admission whose context commit failed remains in
        // memory for durable recovery. Do not spin by immediately retrying the
        // same failing append; all normally consumed admissions can drain the
        // next prompt/background notification now.
        if (!this.isAdmissionPending(admission)) this.onCompactionFinished();
      }
    }
  }

  private isAdmissionPending(
    admission: Pick<TurnInputConsumption, 'kind' | 'id'> | undefined,
  ): boolean {
    if (admission === undefined) return false;
    return admission.kind === 'prompt'
      ? this.deferredPrompts.some((prompt) => prompt.admissionId === admission.id)
      : this.steerBuffer.some((steer) => steer.admissionId === admission.id);
  }

  /**
   * Drives an active goal as a sequence of ordinary turns — the autonomous
   * equivalent of the user repeatedly typing "continue". Each iteration runs one
   * full turn, then reads the goal status the model set via `UpdateGoal`:
   * `complete` (the record is cleared) / `blocked` stop the loop; `active`
   * (the model didn't decide) re-injects the goal reminder and runs the
   * next continuation turn. Aborted or failed turns pause the goal. Goal-state
   * blockers, such as explicit `UpdateGoal('blocked')`, prompt-hook blocks, and
   * budget limits, block it (all resumable). Returns the final turn's result.
   */
  private async driveGoal(
    firstTurnId: number,
    input: readonly ContentPart[],
    origin: PromptOrigin,
    signal: AbortSignal,
    firstDeferredPromptId?: string,
    firstPromptId?: string,
    firstAdmission?: Pick<TurnInputConsumption, 'kind' | 'id'>,
  ): Promise<TurnEndResult> {
    let turnId = firstTurnId;
    let turnInput = input;
    let turnOrigin = origin;
    let deferredPromptId = firstDeferredPromptId;
    let promptId = firstPromptId;
    let admission = firstAdmission;
    while (true) {
      const goalBeforeTurn = this.agent.goal.getGoal().goal;
      if (goalBeforeTurn?.status === 'active' && goalBeforeTurn.budget.overBudget) {
        // No model step will run, so close targeted steer admission before the
        // asynchronous goal update creates a window for a falsely-ACKed steer.
        this.closeSteerAdmission(turnId);
        await this.agent.goal.markBlocked({ reason: 'A configured budget was reached' });
        const ended = await this.endGoalTurnWithoutModel(
          turnId,
          turnInput,
          turnOrigin,
          deferredPromptId,
          promptId,
          admission,
        );
        return { event: ended };
      }

      // Count the turn about to run (no-op if the goal isn't active), so the
      // completion stats include the turn in which the model reports `complete`.
      // Wall-clock is tracked live by the store (anchored while `active`), so the
      // timer is correct even when the model completes mid-turn.
      await this.agent.goal.incrementTurn();
      const end = await this.runOneTurn(
        turnId,
        turnInput,
        turnOrigin,
        signal,
        false,
        deferredPromptId,
        promptId,
        admission,
      );

      if (end.event.reason === 'cancelled') {
        await this.agent.goal.pauseOnInterrupt({ reason: 'Paused after interruption' });
        return end;
      }
      if (end.event.reason === 'failed') {
        await this.agent.goal.pauseActiveGoal({ reason: goalFailurePauseReason(end.event.error) });
        return end;
      }
      if (end.event.reason === 'blocked' || end.blockedByUserPromptHook === true) {
        await this.agent.goal.markBlocked({ reason: 'Blocked by UserPromptSubmit hook' });
        return end;
      }

      // The model decides via UpdateGoal: a cleared record means `complete`;
      // `blocked` remains as a non-active record. Runtime failures and user
      // interrupts can still leave the goal paused. Only a still `active` goal
      // continues to another turn.
      const goal = this.agent.goal.getGoal().goal;
      if (goal === null || goal.status !== 'active') {
        return end;
      }
      // Hard budgets (turn / token / wall-clock, set via the SDK) are a
      // deterministic ceiling: block when reached. `blocked` is resumable.
      if (goal.budget.overBudget) {
        await this.agent.goal.markBlocked({ reason: 'A configured budget was reached' });
        return end;
      }

      // Background/cron notifications may arrive while the completed goal
      // turn is awaiting goal-store bookkeeping. Commit them before appending
      // the synthetic continuation prompt so durable arrival order is kept.
      this.flushSteerBuffer();

      turnId = this.allocateTurnId();
      turnInput = [{ type: 'text', text: GOAL_CONTINUATION_PROMPT }];
      turnOrigin = GOAL_CONTINUATION_ORIGIN;
      deferredPromptId = undefined;
      promptId = undefined;
      admission = undefined;
    }
  }

  private async endGoalTurnWithoutModel(
    turnId: number,
    input: readonly ContentPart[],
    origin: PromptOrigin,
    deferredPromptId?: string,
    promptId?: string,
    admission?: Pick<TurnInputConsumption, 'kind' | 'id'>,
  ): Promise<TurnEndedEvent> {
    this.agent.usage.beginTurn();
    const startedAt = Date.now();
    this.agent.emitEvent({ type: 'turn.started', turnId, origin, promptId, deferredPromptId });
    this.agent.context.appendUserMessage(
      input,
      origin,
      admission === undefined ? undefined : { ...admission, turnId },
    );
    let ended: TurnEndedEvent = {
      type: 'turn.ended',
      turnId,
      reason: 'completed',
      durationMs: Date.now() - startedAt,
    };
    let errorEvent: AgentEvent | undefined;
    const steerFlushError = this.sealAndFlushSteers(turnId);
    if (steerFlushError !== undefined) {
      const summary = summarizeTurnError(steerFlushError.error, turnId);
      ended = {
        type: 'turn.ended',
        turnId,
        reason: 'failed',
        error: summary,
        durationMs: Date.now() - startedAt,
      };
      errorEvent = { type: 'error', ...summary };
      try {
        this.appendTurnOutcomeReminder(ended, undefined);
      } catch (error) {
        this.bestEffortError('goal turn terminal reminder failed', { error, turnId });
      }
    }
    this.agent.usage.endTurn();
    this.agent.emitEvent(ended);
    if (errorEvent !== undefined) this.agent.emitEvent(errorEvent);
    return ended;
  }

  /**
   * Runs exactly one logical turn end to end: per-turn bookkeeping, `turn.started`,
   * the prompt + goal reminder, the step loop, and `turn.ended`. Goal-agnostic —
   * the driver layers goal semantics on top. Never throws; abnormal ends are
   * mapped to a `cancelled`/`failed` `turn.ended` and returned.
   */
  private async runOneTurn(
    turnId: number,
    input: readonly ContentPart[],
    origin: PromptOrigin,
    signal: AbortSignal,
    standalone: boolean,
    deferredPromptId?: string,
    promptId?: string,
    admission?: Pick<TurnInputConsumption, 'kind' | 'id'>,
  ): Promise<TurnEndResult> {
    const active = this.activeTurn;
    if (
      active !== null &&
      active !== 'resuming' &&
      this.activeWorkerOwnsTurn(turnId)
    ) {
      active.terminalizing = false;
    }
    this.currentStep = 0;
    this.stepToolCallKeys.clear();
    this.toolCallDupType.clear();
    const telemetryMode = this.telemetryMode();
    this.telemetryModeByTurn.set(turnId, telemetryMode);
    this.currentStepByTurn.set(turnId, 0);
    this.agent.telemetry.track('turn_started', { turn_id: turnId, mode: telemetryMode, ...this.requestProtocolProps() });
    this.agent.fullCompaction.resetForTurn();
    this.agent.usage.beginTurn();
    this.agent.emitEvent({ type: 'turn.started', turnId, origin, promptId, deferredPromptId });
    this.agent.context.appendUserMessage(
      input,
      origin,
      admission === undefined ? undefined : { ...admission, turnId },
    );

    const startedAt = Date.now();
    let ended: TurnEndedEvent;
    let blockedByUserPromptHook = false;
    let completedStopReason: LoopTurnStopReason | undefined;
    // Emitted after turn.ended (preserving prior ordering), so the error event
    // sits just past the turn.ended boundary that consumers watch for.
    let errorEvent: AgentEvent | undefined;
    try {
      const promptHookEnded = await this.applyUserPromptHook(turnId, input, origin, signal, startedAt);
      if (promptHookEnded !== undefined) {
        ended = promptHookEnded.event;
        blockedByUserPromptHook = promptHookEnded.blocked;
      } else {
        const stopReason = await this.runStepLoop(turnId, signal);
        completedStopReason = stopReason;
        if (stopReason === 'filtered') {
          const summary = providerFilteredPayload(turnId);
          ended = {
            type: 'turn.ended',
            turnId,
            reason: 'failed',
            error: summary,
            durationMs: Date.now() - startedAt,
          };
          errorEvent = { type: 'error', ...summary };
        } else {
          // A synchronous context/persistence observer can cancel while the
          // loop is flushing accepted steers after a provider response. The
          // loop may still report its prior stop reason, so the signal is the
          // final authority at this boundary.
          const reason: TurnEndReason =
            stopReason === 'aborted' || signal.aborted ? 'cancelled' : 'completed';
          ended = {
            type: 'turn.ended',
            turnId,
            reason,
            durationMs: Date.now() - startedAt,
          };
        }
      }
    } catch (error) {
      if (isAbortError(error) || signal.aborted) {
        ended = { type: 'turn.ended', turnId, reason: 'cancelled', durationMs: Date.now() - startedAt };
      } else {
        const summary = summarizeTurnError(error, turnId);
        void this.agent.hooks?.fireAndForgetTrigger('StopFailure', {
          matcherValue: summary.name,
          inputData: { errorType: summary.name, errorMessage: summary.message },
        });
        ended = { type: 'turn.ended', turnId, reason: 'failed', error: summary, durationMs: Date.now() - startedAt };
        errorEvent = { type: 'error', ...summary };
        if (this.shouldTrackApiError(turnId)) {
          const classification = classifyApiError(error, summary);
          const properties: Record<string, TelemetryPropertyValue> = {
            error_type: classification.errorType,
            model: this.agent.config.model,
            alias: this.agent.config.modelAlias,
            ...this.requestProtocolProps(),
            retryable: summary.retryable,
            duration_ms: Date.now() - startedAt,
          };
          if (classification.statusCode !== undefined) {
            properties['status_code'] = classification.statusCode;
          }
          const inputTokens = currentTurnInputTokens(this.agent.usage.data().currentTurn);
          if (inputTokens !== undefined) {
            properties['input_tokens'] = inputTokens;
          }
          this.agent.telemetry.track('api_error', properties);
        }
      }
    }
    if (ended.reason === 'failed' || ended.reason === 'cancelled') {
      try {
        this.ensureTurnOutcomeIntent(ended, signal.reason);
      } catch (error) {
        // The worker still has a chance to materialize the reserved intent
        // below. Never let reminder persistence suppress turn terminalization.
        this.bestEffortError('turn outcome intent persistence failed', { error, turnId });
      }
    }
    // A live turn must never end with recorded tool calls still awaiting
    // results; if one does (a dispatch failure mid-batch broke the "every
    // recorded call gets a result" invariant), close the exchange now so the
    // context state machine cannot strand later messages in deferredMessages.
    this.closeAbandonedToolExchange(ended);
    // Seal steer admission at the unified terminal boundary, including paths
    // that never entered shouldContinueAfterStop (provider failure, prompt-hook
    // block/failure, cancellation). Anything already acknowledged belongs to
    // this turn and must land before its outcome reminder; otherwise a later
    // prompt would reorder older steer input behind itself.
    const recordTerminalizationFailure = (error: unknown): void => {
      if (ended.reason === 'completed' || ended.reason === 'blocked') {
        const summary = summarizeTurnError(error, turnId);
        ended = {
          type: 'turn.ended',
          turnId,
          reason: 'failed',
          error: summary,
          durationMs: Date.now() - startedAt,
        };
        errorEvent = { type: 'error', ...summary };
        try {
          this.ensureTurnOutcomeIntent(ended, signal.reason);
        } catch (outcomeError) {
          this.bestEffortError('turn outcome intent persistence failed', {
            error: outcomeError,
            turnId,
          });
        }
        return;
      }
      this.bestEffortError('turn terminal context finalization failed', {
        error,
        turnId,
      });
    };
    const steerFlushError = this.sealAndFlushSteers(turnId);
    if (steerFlushError !== undefined) {
      recordTerminalizationFailure(steerFlushError.error);
    }
    try {
      this.appendTurnOutcomeReminder(ended, signal.reason);
    } catch (error) {
      recordTerminalizationFailure(error);
    }
    // Emit the terminal turn.ended and (for a standalone turn) release the active
    // turn in the SAME synchronous frame, so the session is observably idle the
    // instant turn.ended fires. A goal drive keeps the active turn across its
    // continuation turns and releases it in `turnWorker` instead (`standalone`
    // is false for those).
    if (this.currentId === turnId) {
      this.agent.usage.endTurn();
    }
    // A user interrupt (e.g. Esc) aborts the turn without the normal Stop hook
    // firing, so external tooling that tracks status from hooks would otherwise
    // never see the turn stop. Emit an observation-only Interrupt event for it.
    // Gate on isUserCancellation: a `cancelled` turn can also come from a
    // programmatic abort (e.g. a subagent deadline timeout, which shares this
    // hook engine), and those must not be misreported as a user interrupt.
    if (ended.reason === 'cancelled' && isUserCancellation(signal.reason)) {
      void this.agent.hooks?.fireAndForgetTrigger('Interrupt', {
        inputData: { turnId, reason: 'cancelled' },
      });
    }
    this.agent.telemetry.track('turn_ended', {
      turn_id: turnId,
      reason: ended.reason,
      duration_ms: ended.durationMs,
      mode: this.telemetryModeByTurn.get(turnId) ?? this.telemetryMode(),
      ...this.requestProtocolProps(),
    });
    this.agent.emitEvent(ended);
    // Release the active turn in the same frame as turn.ended for a standalone
    // turn, so the session is observably idle the instant turn.ended fires.
    // Exception: if the model turned the goal active during this turn (e.g.
    // CreateGoal), the session is NOT idle — turnWorker is about to drive the
    // goal. Keep the active turn alive (as the already-active goal path does) so
    // those autonomous continuations stay cancelable and exclude concurrent
    // turns; turnWorker releases it after the drive.
    if (
      standalone &&
      this.currentId === turnId &&
      this.agent.goal.getGoal().goal?.status !== 'active'
    ) {
      this.activeTurn = null;
    }
    if (this.agent.swarmMode.shouldAutoExit) {
      this.agent.swarmMode.exit();
    }
    if (errorEvent !== undefined) {
      this.agent.emitEvent(errorEvent);
    }
    if (ended.reason !== 'completed') {
      // Fallback for turns that end abnormally without a `turn.interrupted`
      // loop event reaching `trackLoopTelemetry` (e.g. a user-prompt hook block
      // or an abort that bypasses the step loop). `ended.reason` maps onto the
      // same interrupt-reason taxonomy the loop-event path uses; for a
      // `cancelled` end the signal's reason decides user_cancelled vs aborted.
      const interruptReason = telemetryInterruptReason(ended.reason, isUserCancellation(signal.reason));
      this.trackTurnInterrupted(
        turnId,
        this.currentStepByTurn.get(turnId) ?? this.currentStep,
        interruptReason,
      );
    }
    this.telemetryModeByTurn.delete(turnId);
    this.currentStepByTurn.delete(turnId);
    this.interruptedTelemetryTurnIds.delete(turnId);
    this.stepFailureByTurn.delete(turnId);
    return { event: ended, stopReason: completedStopReason, blockedByUserPromptHook };
  }

  private appendTurnOutcomeReminder(ended: TurnEndedEvent, cancellationReason: unknown): void {
    const outcome = this.ensureTurnOutcomeIntent(ended, cancellationReason);
    if (outcome === undefined) return;
    this.materializeTurnOutcome(outcome);
  }

  /**
   * Persist the fact that an abnormal turn still needs a model-visible
   * reminder. Cancellation normally installs this intent atomically on
   * `turn.cancel`; failures get their own record as soon as their terminal
   * outcome is known.
   */
  private ensureTurnOutcomeIntent(
    ended: TurnEndedEvent,
    cancellationReason: unknown,
  ): PendingTurnOutcome | undefined {
    if (ended.reason !== 'failed' && ended.reason !== 'cancelled') return undefined;
    const existing = [...this.pendingOutcomes.values()].find(
      (outcome) => outcome.turnId === ended.turnId,
    );
    if (existing !== undefined) return existing;

    const content =
      ended.reason === 'failed'
        ? renderTurnFailureReminder(ended.error)
        : renderTurnCancellationReminder(
            cancellationReason,
            isUserCancellation(cancellationReason),
          );
    const outcome = { id: randomUUID(), turnId: ended.turnId, content };
    // Reserve before persistence so a synchronous record observer that
    // re-enters terminalization sees one stable identity, never two.
    this.pendingOutcomes.set(outcome.id, outcome);
    this.agent.records.logRecord({
      type: 'turn.outcome',
      outcomeId: outcome.id,
      turnId: outcome.turnId,
      content: outcome.content,
    });
    return outcome;
  }

  private materializeTurnOutcome(outcome: PendingTurnOutcome): void {
    this.agent.context.appendSystemReminder(
      outcome.content,
      {
        kind: 'injection',
        variant: TURN_OUTCOME_REMINDER_VARIANT,
      },
      undefined,
      outcome.id,
    );
  }

  private materializePendingOutcomes(): void {
    for (const outcome of this.pendingOutcomes.values()) {
      this.materializeTurnOutcome(outcome);
    }
  }

  private async applyUserPromptHook(
    turnId: number,
    input: readonly ContentPart[],
    origin: PromptOrigin,
    signal: AbortSignal,
    startedAt: number,
  ): Promise<PromptHookEndResult | undefined> {
    if (origin.kind !== 'user') return undefined;
    signal.throwIfAborted();
    const promptHookResults = await this.agent.hooks?.trigger('UserPromptSubmit', {
      matcherValue: input,
      signal,
      inputData: { prompt: input },
    });
    signal.throwIfAborted();
    const blockResult = renderUserPromptHookBlockResult(promptHookResults);
    if (blockResult !== undefined) {
      this.agent.context.appendMessage({
        role: 'assistant',
        content: [{ type: 'text', text: blockResult.text }],
        toolCalls: [],
        origin: { kind: 'hook_result', event: 'UserPromptSubmit', blocked: true },
      });
      this.agent.emitEvent({
        type: 'hook.result',
        turnId,
        hookEvent: blockResult.event,
        content: blockResult.message,
        blocked: true,
      });
      // The terminal turn.ended is emitted by runOneTurn (synchronously with the
      // activeTurn clear), not here, so the session is idle the moment it fires.
      return {
        event: { type: 'turn.ended', turnId, reason: 'blocked', durationMs: Date.now() - startedAt },
        blocked: true,
      };
    }

    const hookResult = renderUserPromptHookResult(promptHookResults);
    if (hookResult === undefined) return undefined;

    this.agent.context.appendUserMessage([{ type: 'text', text: hookResult.text }], {
      kind: 'hook_result',
      event: 'UserPromptSubmit',
    });
    this.agent.emitEvent({
      type: 'hook.result',
      turnId,
      hookEvent: hookResult.event,
      content: hookResult.message,
    });
    return undefined;
  }

  private async runStepLoop(turnId: number, signal: AbortSignal): Promise<LoopTurnStopReason> {
    let stopHookContinuationUsed = false;
    let goalOutcomeMessageContinuationUsed = false;
    let goalOutcomeToolResultPending = false;
    const deduper = new ToolCallDeduplicator({ telemetry: this.agent.telemetry });
    await this.agent.mcp?.waitForInitialLoad(signal);
    // Surface the active goal at the start of the turn (append-only; no-op when
    // there is no active goal). Each goal continuation is its own turn, so this
    // re-injects the reminder once per turn rather than per step, preserving prompt caching.
    await this.agent.injection.injectGoal();
    // Announce loadable-tool changes at the same boundary cadence: a diff is
    // appended only when the loadable set actually changed, so quiet turns
    // keep the prompt cache fully warm.
    this.agent.injection.injectToolsDiff();
    let mediaStripSnapshot: MediaStripSnapshot | undefined;
    const buildMessagesMediaStripped = (): Message[] => {
      const messages = this.agent.context.messages;
      mediaStripSnapshot ??= captureMediaStripSnapshot(messages);
      return stripMediaPartsBySnapshot(messages, mediaStripSnapshot);
    };
    let turnLlm = this.agent.llm;
    let systemPromptRefreshSnapshot = this.agent.fullCompaction.systemPromptRefreshSnapshot;
    while (true) {
      signal.throwIfAborted();
      const loopControl = this.agent.kimiConfig?.loopControl;
      let stopForGoalBudget = false;
      try {
        const result = await runTurn({
          turnId: String(turnId),
          signal,
          llm: turnLlm,
          buildLlm: () => {
            const snapshot = this.agent.fullCompaction.systemPromptRefreshSnapshot;
            if (snapshot.revision !== systemPromptRefreshSnapshot.revision) {
              systemPromptRefreshSnapshot = snapshot;
              if (snapshot.systemPrompt !== undefined) {
                turnLlm = turnLlm.withSystemPrompt(snapshot.systemPrompt);
              }
            }
            return turnLlm;
          },
          buildMessages: () => this.agent.context.messages,
          buildMessagesStrict: () => this.agent.context.strictMessages,
          buildMessagesMediaDegraded: () => this.agent.context.mediaDegradedMessages,
          buildMessagesMediaStripped,
          dispatchEvent: this.buildDispatchEvent(turnId),
          // Re-read per step (not snapshotted per turn) so a select_tools load
          // is dispatchable on the very next step of the same turn.
          buildTools: () => this.agent.tools.loopTools,
          describeMissingTool: (name) => this.agent.tools.missingToolMessage(name),
          log: this.agent.log,
          maxSteps: loopControl?.maxStepsPerTurn,
          maxRetryAttempts: loopControl?.maxRetriesPerStep,
          recordStepUsage: async (usage) => {
            try {
              const snapshot = await this.agent.goal.recordTokenUsage(usage.output);
              stopForGoalBudget = snapshot?.budget.overBudget === true;
            } catch (error) {
              this.agent.log.warn('goal token accounting failed', { error });
            }
          },
          hooks: {
            beforeStep: async ({ signal: stepSignal }) => {
              this.agent.microCompaction.detect();
              await this.agent.fullCompaction.beforeStep(stepSignal);
              // Flush steered messages (background-task / cron notifications,
              // user interrupts) AFTER compaction so they land in the
              // post-compaction context instead of being dropped by it. The
              // keep/drop decision lives in
              // `compactionUserMessageDisposition()`; these origins are not
              // re-injected later, so append them only after compaction runs.
              this.flushSteerBuffer();
              await this.agent.injection.inject();
              deduper.beginStep();
              return;
            },
            afterStep: async ({ usage, llm }) => {
              this.agent.usage.record(llm.modelName, usage, 'turn');
              await this.agent.fullCompaction.afterStep();
              deduper.endStep();
              return stopForGoalBudget ? { stopTurn: true } : undefined;
            },
            // oxlint-disable-next-line no-loop-func -- stop hook continuation state is scoped to this turn.
            shouldContinueAfterStop: async (ctx) => {
              const { signal } = ctx;
              const flushedSteeredMessages = this.flushSteerBuffer();
              // 0. A reached hard goal budget is a deterministic ceiling. While
              //    the goal is still active, never extend the turn — neither a
              //    steered message nor a Stop-hook continuation — past it; end
              //    the turn so the goal driver blocks the goal at the boundary.
              //    Buffered steers are still flushed above so real-time user
              //    input is preserved in context even when the budget stops the
              //    turn. A goal the model just marked terminal is no longer
              //    active, so its final outcome message (step 2 below) still runs.
              if (stopForGoalBudget && this.agent.goal.getActiveGoal() !== null) {
                this.closeSteerAdmission(turnId);
                return { continue: false };
              }
              // 1. If steered user messages were flushed and no active-goal
              //    budget stopped the turn, let the model react to them.
              if (flushedSteeredMessages) return { continue: true };
              signal.throwIfAborted();

              // Print-mode drain: when `kimi -p` ends a turn while background
              // subagents are still running, hold the turn open and idle-wait
              // until they finish. Their completions steer into the buffer
              // during the wait and are flushed afterward, so the model gets
              // one wrap-up step to react (nominate, backfill, ...) before the
              // turn ends. The wait is bounded by each subagent's own timeout,
              // not by a separate drain deadline, so late-spawned or long-
              // running subagents are still observed. Gated on a session flag
              // so interactive / goal modes are unaffected.
              if (this.agent.printDrainAgentTasksOnStop) {
                const hasActiveAgentTask = this.agent.background
                  .list(true)
                  .some((task) => task.kind === 'agent');
                if (hasActiveAgentTask) {
                  await this.agent.background.waitForActiveTasks(
                    (task) => task.kind === 'agent',
                    { signal },
                  );
                  this.flushSteerBuffer();
                  return { continue: true };
                }
              }

              // 2. After UpdateGoal marks a goal terminal, its tool result carries
              //    the final-message reminder. Let the model read that result and
              //    produce one user-facing outcome message before the turn ends.
              if (
                !goalOutcomeMessageContinuationUsed &&
                goalOutcomeToolResultPending
              ) {
                goalOutcomeMessageContinuationUsed = true;
                goalOutcomeToolResultPending = false;
                if (!hasStepBudgetRemaining(loopControl?.maxStepsPerTurn, ctx.stepNumber)) {
                  this.closeSteerAdmission(turnId);
                  return { continue: false };
                }
                return { continue: true };
              }

              // 3. The external Stop hook gets exactly one continuation; the cap
              //    is intentionally separate from (and does not cap) goal mode.
              if (!stopHookContinuationUsed) {
                const stopBlock = await this.agent.hooks?.triggerBlock('Stop', {
                  signal,
                  inputData: { stopHookActive: stopHookContinuationUsed },
                });
                signal.throwIfAborted();
                if (stopBlock !== undefined) {
                  stopHookContinuationUsed = true;
                  this.agent.context.appendUserMessage(
                    [{ type: 'text', text: stopBlock.reason }],
                    {
                      kind: 'system_trigger',
                      name: 'stop_hook',
                    },
                  );
                  return { continue: true };
                }
              }

              // 4. Otherwise stop. Goal continuation is no longer driven here:
              //    each goal turn is an ordinary turn, and the goal driver decides
              //    whether to run another after this one ends.
              // Re-check after the asynchronous Stop hook. A steer accepted
              // while that hook was pending must either get a model step in
              // this turn or be rejected once the terminal gate closes; it
              // must never receive an ACK and leak into a future prompt.
              if (this.flushSteerBuffer()) return { continue: true };
              this.closeSteerAdmission(turnId);
              return { continue: false };
            },
            prepareToolExecution: async (ctx) => {
              const cached = deduper.checkSameStep(
                ctx.toolCall.id,
                ctx.toolCall.name,
                ctx.args,
              );
              if (cached !== null) return { syntheticResult: cached };
              return undefined;
            },
            authorizeToolExecution: async (ctx) => {
              return this.agent.permission.beforeToolCall(ctx);
            },
            finalizeToolResult: async (ctx) => {
              // Resolve dedup BEFORE firing the PostToolUse hook so same-step
              // dups (whose ctx.result is the dedup placeholder) report the
              // original's real outcome, not an empty success.
              const finalResult = await deduper.finalizeResult(
                ctx.toolCall.id,
                ctx.toolCall.name,
                ctx.args,
                ctx.result,
              );
              const { isError, output } = finalResult;
              const event = isError === true ? 'PostToolUseFailure' : 'PostToolUse';
              void this.agent.hooks?.fireAndForgetTrigger(event, {
                matcherValue: ctx.toolCall.name,
                inputData: {
                  toolName: ctx.toolCall.name,
                  toolInput: toolInputRecord(ctx.args),
                  toolCallId: ctx.toolCall.id,
                  error: isError === true ? toKimiErrorPayload(toolOutputText(output)) : undefined,
                  toolOutput: isError === true ? undefined : toolOutputText(output).slice(0, 2000),
                },
              });
              const modelResult = await budgetToolResultForModel({
                homedir: this.agent.homedir,
                toolName: ctx.toolCall.name,
                toolCallId: ctx.toolCall.id,
                result: finalResult,
              });
              if (isTerminalUpdateGoalResult(ctx.toolCall.name, ctx.args, finalResult)) {
                goalOutcomeToolResultPending = true;
              }
              return modelResult;
            },
          },
        });

        return result.stopReason;
      } catch (error) {
        const isContextOverflow =
          error instanceof APIContextOverflowError ||
          (isKimiError(error) && error.code === ErrorCodes.CONTEXT_OVERFLOW);
        const estimatedRequestTokens = isContextOverflow
          ? this.agent.fullCompaction.estimateCurrentRequestTokens()
          : undefined;
        if (
          isContextOverflow ||
          this.agent.fullCompaction.shouldRecoverFromContextOverflow(error, estimatedRequestTokens)
        ) {
          this.agent.fullCompaction.observeContextOverflow(
            estimatedRequestTokens ?? this.agent.fullCompaction.estimateCurrentRequestTokens(),
          );
          await this.agent.fullCompaction.handleOverflowError(signal, error);
          continue; // Retry with compacted context
        }
        if (isMaxStepsExceededError(error)) {
          this.agent.log.warn('turn hit max steps', {
            turnId,
            steps: this.currentStepByTurn.get(turnId) ?? this.currentStep,
            limit: isKimiError(error) ? error.details?.['maxSteps'] : undefined,
          });
        } else {
          this.agent.log.error('turn failed', { turnId, error });
        }
        throw error;
      }
    }
  }

  // Guarded so this repair can never turn a finished turn into a crash: a
  // failure to close (e.g. record persistence still broken) is logged and the
  // projection-level safeguards remain the last line of defense.
  private closeAbandonedToolExchange(ended: TurnEndedEvent): void {
    try {
      const closed = this.agent.context.closeAbandonedToolExchange(
        abandonedToolResultOutput(ended),
      );
      if (closed === 0) return;
      this.bestEffortWarn('closed abandoned tool exchange at turn end', {
        turnId: ended.turnId,
        reason: ended.reason,
        closed,
      });
      this.agent.telemetry.track('tool_exchange_abandoned', {
        reason: ended.reason,
        closed,
      });
    } catch (error) {
      this.bestEffortWarn('failed to close abandoned tool exchange', { error });
    }
  }

  private buildDispatchEvent(turnId: number) {
    return createLoopEventDispatcher({
      appendTranscriptRecord: async (event: LoopRecordedEvent) => {
        this.agent.context.appendLoopEvent(event);
      },
      emitLiveEvent: (event: LoopEvent) => {
        this.noteFirstRequestEvent(event);
        this.trackLoopTelemetry(event, turnId);
        const mapped = mapLoopEvent(event, turnId);
        if (mapped !== undefined) this.agent.emitEvent(mapped);
      },
    });
  }

  private noteFirstRequestEvent(event: LoopEvent): void {
    switch (event.type) {
      case 'step.end':
      case 'content.part':
      case 'tool.call':
      case 'text.delta':
      case 'thinking.delta':
      case 'tool.call.delta': {
        const active = this.activeTurn;
        if (active === null || active === 'resuming') return;
        active.firstRequest.resolve();
        return;
      }
      default:
        return;
    }
  }

  private trackLoopTelemetry(event: LoopEvent, turnId: number): void {
    if (event.type === 'step.begin') {
      this.beginTrackedStep(turnId, event.step);
      return;
    }
    if (event.type === 'turn.interrupted') {
      if (event.reason === 'error' && event.activeStep !== undefined) {
        this.stepFailureByTurn.set(turnId, event);
      }
      this.trackTurnInterrupted(
        turnId,
        interruptedStep(event),
        event.interruptReason ?? telemetryInterruptReason(event.reason, false),
      );
      return;
    }
    this.trackToolLifecycle(event, turnId);
  }

  private beginTrackedStep(turnId: number, step: number): void {
    this.currentStepByTurn.set(turnId, step);
    this.currentStep = step;
    if (!this.stepToolCallKeys.has(step)) {
      this.stepToolCallKeys.set(step, new Set());
    }
  }

  private trackToolLifecycle(event: LoopEvent, turnId: number): void {
    if (event.type === 'tool.call') {
      const dupType = this.trackDuplicateToolCall(turnId, event.step, event.name, event.args);
      this.toolCallDupType.set(
        event.toolCallId,
        dupType === 'cross_step' ? 'cross_step' : 'normal',
      );
      this.toolCallStartedAt.set(event.toolCallId, {
        name: event.name,
        startedAt: Date.now(),
      });
      return;
    }
    if (event.type === 'tool.result') {
      const started = this.toolCallStartedAt.get(event.toolCallId);
      if (started === undefined) return;
      this.toolCallStartedAt.delete(event.toolCallId);
      const dupType = this.toolCallDupType.get(event.toolCallId) ?? 'normal';
      this.toolCallDupType.delete(event.toolCallId);
      const outcome = telemetryToolOutcome(event.result);
      const properties: Record<string, TelemetryPropertyValue> = {
        turn_id: turnId,
        tool_name: started.name,
        outcome,
        duration_ms: Date.now() - started.startedAt,
        dup_type: dupType,
      };
      const errorType = outcome === 'error' ? telemetryToolErrorType(event.result) : undefined;
      if (errorType !== undefined) {
        properties['error_type'] = errorType;
      }
      this.agent.telemetry.track('tool_call', properties);
    }
  }

  private trackDuplicateToolCall(
    turnId: number,
    step: number,
    toolName: string,
    args: unknown,
  ): 'normal' | 'same_step' | 'cross_step' {
    const argsText = canonicalTelemetryArgs(args);
    const key = `${toolName}\u0000${argsText}`;
    const stepKeys = this.stepToolCallKeys.get(step) ?? new Set<string>();
    this.stepToolCallKeys.set(step, stepKeys);

    let dupType: 'same_step' | 'cross_step' | undefined;
    if (stepKeys.has(key)) {
      dupType = 'same_step';
    } else if (this.hasPriorStepToolCallKey(step, key)) {
      dupType = 'cross_step';
    }

    stepKeys.add(key);
    if (dupType === undefined) return 'normal';

    this.agent.telemetry.track('tool_call_dedup_detected', {
      turn_id: turnId,
      step_no: step,
      tool_name: toolName,
      dup_type: dupType,
      args_hash: createHash('sha256').update(argsText).digest('hex').slice(0, 8),
    });
    return dupType;
  }

  private hasPriorStepToolCallKey(step: number, key: string): boolean {
    for (const [seenStep, keys] of this.stepToolCallKeys) {
      if (seenStep !== step && keys.has(key)) return true;
    }
    return false;
  }

  private trackTurnInterrupted(
    turnId: number,
    atStep: number,
    interruptReason: TelemetryInterruptReason,
  ): void {
    if (this.interruptedTelemetryTurnIds.has(turnId)) return;
    this.interruptedTelemetryTurnIds.add(turnId);
    this.agent.telemetry.track('turn_interrupted', {
      turn_id: turnId,
      mode: this.telemetryModeByTurn.get(turnId) ?? this.telemetryMode(),
      at_step: atStep,
      interrupt_reason: interruptReason,
      ...this.requestProtocolProps(),
    });
  }

  private telemetryMode(): 'agent' | 'plan' {
    return this.agent.planMode.isActive ? 'plan' : 'agent';
  }

  /**
   * Resolve the current model's provider wire type and any model-level protocol
   * override for request telemetry. Never throws — telemetry must not break a
   * turn over an unresolvable provider config (the step loop will surface that
   * error on its own).
   */
  private requestProtocolProps(): { provider_type?: string; protocol?: string } {
    const model = this.agent.config.modelAlias;
    if (model === undefined) return {};
    try {
      const resolved = this.agent.modelProvider?.resolveProviderConfig(model);
      if (resolved === undefined) return {};
      return {
        provider_type: resolved.type,
        protocol: resolved.protocol ?? resolved.type,
      };
    } catch {
      return {};
    }
  }

  private shouldTrackApiError(turnId: number): boolean {
    const failure = this.stepFailureByTurn.get(turnId);
    return failure?.reason === 'error' && failure.activeStep !== undefined;
  }
}

function hasStepBudgetRemaining(maxSteps: number | undefined, currentStep: number): boolean {
  return maxSteps === undefined || maxSteps <= 0 || currentStep < maxSteps;
}

function isTerminalUpdateGoalResult(
  toolName: string,
  args: unknown,
  result: ExecutableToolResult,
): boolean {
  if (toolName !== 'UpdateGoal' || result.isError === true || result.stopTurn !== true) {
    return false;
  }
  if (!isPlainRecord(args)) return false;
  const status = args['status'];
  return status === 'complete' || status === 'blocked';
}

function mapLoopEvent(event: LoopEvent, turnId: number): AgentEvent | undefined {
  switch (event.type) {
    case 'step.begin':
      return {
        type: 'turn.step.started',
        turnId,
        step: event.step,
        stepId: event.uuid,
      };
    case 'step.end':
      return {
        type: 'turn.step.completed',
        turnId,
        step: event.step,
        stepId: event.uuid,
        usage: event.usage,
        finishReason: event.finishReason,
        llmFirstTokenLatencyMs: event.llmFirstTokenLatencyMs,
        llmStreamDurationMs: event.llmStreamDurationMs,
        llmRequestBuildMs: event.llmRequestBuildMs,
        llmServerFirstTokenMs: event.llmServerFirstTokenMs,
        llmServerDecodeMs: event.llmServerDecodeMs,
        llmClientConsumeMs: event.llmClientConsumeMs,
        providerFinishReason: event.providerFinishReason,
        rawFinishReason: event.rawFinishReason,
      };
    case 'step.retrying':
      return {
        type: 'turn.step.retrying',
        turnId,
        step: event.step,
        stepId: event.stepUuid,
        failedAttempt: event.failedAttempt,
        nextAttempt: event.nextAttempt,
        maxAttempts: event.maxAttempts,
        delayMs: event.delayMs,
        errorName: event.errorName,
        errorMessage: event.errorMessage,
        statusCode: event.statusCode,
      };
    case 'content.part':
      return undefined;
    case 'tool.call':
      return {
        type: 'tool.call.started',
        turnId,
        toolCallId: event.toolCallId,
        name: event.name,
        args: event.args,
        description: event.description,
        display: event.display,
      };
    case 'tool.result':
      return {
        type: 'tool.result',
        turnId,
        toolCallId: event.toolCallId,
        output: event.result.output,
        isError: event.result.isError,
      };
    case 'turn.interrupted':
      if (event.activeStep === undefined) return undefined;
      return {
        type: 'turn.step.interrupted',
        turnId,
        step: event.activeStep,
        reason: event.reason,
        message: event.message,
      };
    case 'text.delta':
      return {
        type: 'assistant.delta',
        turnId,
        delta: event.delta,
      };
    case 'thinking.delta':
      return {
        type: 'thinking.delta',
        turnId,
        delta: event.delta,
      };
    case 'tool.call.delta':
      return {
        type: 'tool.call.delta',
        turnId,
        toolCallId: event.toolCallId,
        name: event.name,
        argumentsPart: event.argumentsPart,
      };
    case 'tool.progress':
      return {
        type: 'tool.progress',
        turnId,
        toolCallId: event.toolCallId,
        update: event.update,
      };
  }
}

function summarizeTurnError(error: unknown, turnId: number): KimiErrorPayload {
  const payload = toKimiErrorPayload(error);
  const details = { ...payload.details, turnId };

  // Substitute a friendlier TUI-aware message for model-not-configured.
  // The raw "Model not set" / "Provider not set" text is not actionable;
  // this string points the user at the login flow.
  if (payload.code === 'model.not_configured') {
    return { ...payload, message: LLM_NOT_SET_MESSAGE, details };
  }

  return { ...payload, details };
}

function providerFilteredPayload(turnId: number): KimiErrorPayload {
  return {
    code: ErrorCodes.PROVIDER_FILTERED,
    message: 'Provider safety policy blocked the response.',
    name: 'ProviderFilteredError',
    details: { finishReason: 'filtered', turnId },
    retryable: false,
  };
}

function goalFailurePauseReason(error: TurnEndedEvent['error']): string {
  if (error?.code === ErrorCodes.PROVIDER_RATE_LIMIT) return GOAL_RATE_LIMIT_PAUSE_REASON;
  if (error?.code === ErrorCodes.PROVIDER_CONNECTION_ERROR) {
    return pauseReasonWithMessage(GOAL_PROVIDER_CONNECTION_PAUSE_PREFIX, error.message);
  }
  if (error?.code === ErrorCodes.PROVIDER_AUTH_ERROR) {
    return pauseReasonWithMessage(GOAL_PROVIDER_AUTH_PAUSE_PREFIX, error.message);
  }
  if (error?.code === ErrorCodes.PROVIDER_FILTERED) {
    return GOAL_PROVIDER_FILTERED_PAUSE_REASON;
  }
  if (error?.code === ErrorCodes.PROVIDER_API_ERROR) {
    return pauseReasonWithMessage(GOAL_PROVIDER_API_PAUSE_PREFIX, error.message);
  }
  if (
    error?.code === ErrorCodes.MODEL_NOT_CONFIGURED ||
    error?.code === ErrorCodes.MODEL_CONFIG_INVALID
  ) {
    return pauseReasonWithMessage(GOAL_MODEL_CONFIG_PAUSE_PREFIX, error.message);
  }
  return pauseReasonWithMessage(GOAL_RUNTIME_PAUSE_PREFIX, error?.message);
}

function pauseReasonWithMessage(prefix: string, message: string | undefined): string {
  return message === undefined || message.length === 0 ? prefix : `${prefix}: ${message}`;
}

function toolInputRecord(args: unknown): Record<string, unknown> {
  return isPlainRecord(args) ? args : {};
}

function toolOutputText(output: ExecutableToolResult['output']): string {
  if (typeof output === 'string') return output;
  return output
    .filter((part): part is Extract<(typeof output)[number], { type: 'text' }> => {
      return typeof part === 'object' && part !== null && part.type === 'text';
    })
    .map((part) => part.text)
    .join('');
}

function interruptedStep(event: LoopTurnInterruptedEvent): number {
  return event.activeStep ?? event.attemptedSteps;
}

/**
 * Telemetry-facing interrupt reason. The loop reports `LoopInterruptReason`
 * (`aborted` | `max_steps` | `error`); we split `aborted` into a deliberate
 * user cancel vs. any other programmatic abort so telemetry can tell them
 * apart. `filtered` is folded in for the fallback path (turn ends flagged
 * `filtered` never emit a `turn.interrupted` loop event).
 */
type TelemetryInterruptReason =
  | 'user_cancelled'
  | 'aborted'
  | 'max_steps'
  | 'error'
  | 'filtered'
  | 'blocked';

function telemetryInterruptReason(
  reason: LoopTurnInterruptedEvent['reason'] | Exclude<TurnEndedEvent['reason'], 'completed'>,
  userCancelled: boolean,
): TelemetryInterruptReason {
  if ((reason === 'aborted' || reason === 'cancelled') && userCancelled) {
    return 'user_cancelled';
  }
  if (reason === 'aborted' || reason === 'cancelled') return 'aborted';
  if (reason === 'failed') return 'error';
  if (reason === 'blocked') return 'blocked';
  // Remaining values are `max_steps` | `error` | `filtered`, which match the
  // telemetry enum.
  return reason;
}

interface ApiErrorClassification {
  readonly errorType: string;
  readonly statusCode?: number;
}

function classifyApiError(error: unknown, summary: KimiErrorPayload): ApiErrorClassification {
  const statusCode = apiStatusCode(error) ?? summaryStatusCode(summary);
  if (statusCode !== undefined) {
    if (statusCode === 429) return { errorType: 'rate_limit', statusCode };
    if (statusCode === 401 || statusCode === 403) return { errorType: 'auth', statusCode };
    if (statusCode >= 500) return { errorType: '5xx_server', statusCode };
    if (isContextOverflowStatusError(statusCode, summary.message)) {
      return { errorType: 'context_overflow', statusCode };
    }
    if (statusCode >= 400) return { errorType: '4xx_client', statusCode };
    return { errorType: 'api', statusCode };
  }

  if (summary.code === ErrorCodes.PROVIDER_RATE_LIMIT) return { errorType: 'rate_limit' };
  if (summary.code === ErrorCodes.PROVIDER_AUTH_ERROR) return { errorType: 'auth' };
  if (summary.code === ErrorCodes.CONTEXT_OVERFLOW) return { errorType: 'context_overflow' };
  if (isApiConnectionError(error, summary)) return { errorType: 'network' };
  if (isApiTimeoutError(error, summary)) return { errorType: 'timeout' };
  if (isApiEmptyResponseError(error, summary)) return { errorType: 'empty_response' };
  return { errorType: 'other' };
}

function apiStatusCode(error: unknown): number | undefined {
  if (error instanceof APIStatusError) {
    const statusCode = (error as { readonly statusCode?: unknown }).statusCode;
    return typeof statusCode === 'number' ? statusCode : undefined;
  }
  if (typeof error !== 'object' || error === null) return undefined;
  const statusCode = (error as { readonly statusCode?: unknown }).statusCode;
  if (typeof statusCode === 'number') return statusCode;
  const status = (error as { readonly status?: unknown }).status;
  return typeof status === 'number' ? status : undefined;
}

function summaryStatusCode(summary: KimiErrorPayload): number | undefined {
  const statusCode = summary.details?.['statusCode'];
  return typeof statusCode === 'number' ? statusCode : undefined;
}

function isApiConnectionError(error: unknown, summary: KimiErrorPayload): boolean {
  return error instanceof APIConnectionError || summary.name === 'APIConnectionError';
}

function isApiTimeoutError(error: unknown, summary: KimiErrorPayload): boolean {
  return (
    error instanceof APITimeoutError ||
    summary.name === 'APITimeoutError' ||
    summary.name === 'TimeoutError'
  );
}

function isApiEmptyResponseError(error: unknown, summary: KimiErrorPayload): boolean {
  return error instanceof APIEmptyResponseError || summary.name === 'APIEmptyResponseError';
}

function currentTurnInputTokens(usage: TokenUsage | undefined): number | undefined {
  if (usage === undefined) return undefined;
  return inputTotal(usage);
}

type ToolTelemetryResult = Extract<LoopEvent, { type: 'tool.result' }>['result'];

function telemetryToolOutcome(result: ToolTelemetryResult): 'success' | 'error' | 'cancelled' {
  if (result.isError !== true) return 'success';
  const text = toolResultText(result).toLowerCase();
  return text.includes('aborted') ||
    text.includes('cancelled') ||
    text.includes('manually interrupted')
    ? 'cancelled'
    : 'error';
}

function telemetryToolErrorType(result: ToolTelemetryResult): string {
  const text = toolResultText(result);
  if (text.startsWith('Tool "') && text.includes('" not found')) return 'ToolNotFound';
  if (text.startsWith('Invalid args for tool "')) return 'ToolInputError';
  if (text.includes('prepareToolExecution hook failed')) return 'HookError';
  if (text.includes('finalizeToolResult hook failed')) return 'HookError';
  if (text.includes('blocked')) return 'ToolBlocked';
  return 'ToolError';
}

function toolResultText(result: ToolTelemetryResult): string {
  return toolOutputText(result.output);
}

// Output for a tool call abandoned by its turn (see closeAbandonedToolExchange):
// name the cause so the model treats the gap as an interruption to reason about,
// not a tool outcome. Mirrors the phrasing of the resume-time synthesis in
// `ContextMemory`.
function abandonedToolResultOutput(ended: TurnEndedEvent): string {
  const cause =
    ended.reason === 'cancelled'
      ? 'the turn was cancelled'
      : ended.reason === 'failed'
        ? `the turn failed${ended.error !== undefined ? ` (${ended.error.message})` : ''}`
        : 'the turn ended';
  return `Tool call did not complete: ${cause} before its result was recorded. Do not assume the tool completed successfully.`;
}
