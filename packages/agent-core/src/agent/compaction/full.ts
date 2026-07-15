import {
  ErrorCodes,
  KimiError,
  isKimiError,
  toKimiErrorPayload,
} from '#/errors';
import {
  APIEmptyResponseError,
  inputTotal,
  isRetryableGenerateError,
  type ContentPart,
  type GenerateResult,
  type Message,
  type TokenUsage,
  APIContextOverflowError,
  APIRequestTooLargeError,
  APIStatusError,
  createUserMessage,
  isImageFormatError,
} from '@moonshot-ai/kosong';
import { createControlledPromise, type ControlledPromise } from '@antfu/utils';

import type { Agent } from '..';
import type { GenerateOptionsWithRequestLogFields } from '../llm-request-logger';
import type { ContextMessage, TurnInputConsumption } from '../context/types';
import { stripDynamicToolContext } from '../context/dynamic-tools';
import type { PromptOrigin } from '../context';
import {
  retryBackoffDelays,
  sleepForRetry,
} from '../../loop/retry';
import {
  renderTodoList,
  TODO_STORE_KEY,
  type TodoItem,
} from '../../tools/builtin/state/todo-list';
import {
  estimateTokens,
  estimateTokensForMessage,
  estimateTokensForMessages,
  estimateTokensForTools,
} from '../../utils/tokens';
import {
  applyCompletionBudget,
  resolveCompletionBudget,
} from '../../utils/completion-budget';
import { renderPrompt } from '../../utils/render-prompt';
import type { AgentReplayRecord } from '../../rpc/resumed';
import { agentRecordAppendAccepted } from '../records';
import compactionInstructionTemplate from './compaction-instruction.md?raw';
import type { CompactionBeginData, CompactionResult } from './types';
import {
  DEFAULT_COMPACTION_CONFIG,
  DefaultCompactionStrategy,
  type CompactionStrategy,
} from './strategy';
import { buildCompactionSummaryText, isRealUserInput } from './handoff';

export const MAX_COMPACTION_RETRY_ATTEMPTS = 5;

const DEFAULT_COMPACTION_MAX_COMPLETION_TOKENS = 128 * 1024;
const OVERFLOW_CONTEXT_SAFETY_RATIO = 0.85;
const OVERFLOW_STATUS_RECOVERY_RATIO = 0.5;

class CompactionTruncatedError extends Error {
  constructor() {
    super('Compaction response was truncated before producing a complete summary.');
    this.name = 'CompactionTruncatedError';
  }
}

type CompactionPhase = 'running' | 'cancelling' | 'committing' | 'committed';

interface ActiveCompaction {
  readonly abortController: AbortController;
  readonly promise: ControlledPromise;
  blockedByTurn: boolean;
  phase: CompactionPhase;
  cancellation?: Promise<void>;
  committedResult?: CompactionResult;
  completionRecordWritten: boolean;
  completionEventEmitted: boolean;
  postCompactHookTriggered: boolean;
  released: boolean;
  postCommit?: Promise<void>;
}

type CompactionReplayRecord = Extract<AgentReplayRecord, { type: 'compaction' }>;

interface RestoredCompaction {
  readonly data: Readonly<CompactionBeginData>;
  readonly replay: CompactionReplayRecord;
}

interface RestoredCompactionRecovery {
  readonly active: ActiveCompaction;
  readonly data: Readonly<CompactionBeginData>;
  readonly result: CompactionResult;
  started: boolean;
}

interface PostCommitOptions {
  readonly emitCompletionEvent: boolean;
  readonly triggerPostCompactHook: boolean;
  readonly deferSystemPromptRefresh: boolean;
}

const LIVE_POST_COMMIT_OPTIONS: PostCommitOptions = {
  emitCompletionEvent: true,
  triggerPostCompactHook: true,
  deferSystemPromptRefresh: false,
};

const RECOVERED_POST_COMMIT_OPTIONS: PostCommitOptions = {
  // A resumed client reconstructs the completed compaction from replay. Emitting
  // a live completion without a matching live `compaction.started` would render
  // a second, phantom compaction row in clients.
  emitCompletionEvent: false,
  // Hooks are external side effects and cannot be replayed exactly once after a
  // crash. The durable context/reminder/terminal invariants are recovered; the
  // observational hook is deliberately not replayed.
  triggerPostCompactHook: false,
  // Session profile handles are restored by Session immediately after
  // Agent.resume(). Retry the refresh at the next request boundary, when that
  // handle is guaranteed to exist.
  deferSystemPromptRefresh: true,
};

const POST_COMMIT_ATTEMPTS = 2;

export class FullCompaction {
  protected compactionCountInTurn = 0;
  protected compacting: ActiveCompaction | null = null;
  private restoredCompaction: RestoredCompaction | null = null;
  private restoredCompactionRecovery: RestoredCompactionRecovery | null = null;
  private restoreRecoverySubscribed = false;
  private postCompactionInjectionPending = false;
  private systemPromptRefreshPending = false;
  private _systemPromptRefreshSnapshot: Readonly<{
    revision: number;
    systemPrompt: string | undefined;
  }> = { revision: 0, systemPrompt: undefined };
  private readonly observedMaxContextTokensByModel = new Map<string, number>();
  // Token count right after the last successful compaction. While no new
  // content has been appended (tokenCountWithPending <= this value), the
  // history is already in its minimal compacted form ([kept user prompts
  // (possibly split around an elision marker), summary]); re-compacting would
  // only nest summaries, so
  // checkAutoCompaction skips in that case even if an observed overflow
  // limit still flags the context as oversized.
  private lastCompactedTokenCount: number | null = null;
  // Counts provider-overflow recoveries in this turn that have not yet been
  // followed by a successful step. Trips MAX_OVERFLOW_COMPACTION_ATTEMPTS to
  // stop an overflow -> compact -> overflow loop when compaction can no
  // longer shrink the request below the model window.
  private consecutiveOverflowCompactions = 0;
  protected readonly strategy: CompactionStrategy;

  constructor(
    protected readonly agent: Agent,
    strategy?: CompactionStrategy,
  ) {
    this.strategy =
      strategy ??
      new DefaultCompactionStrategy(
        () => this.getEffectiveMaxContextTokens(),
        {
          ...DEFAULT_COMPACTION_CONFIG,
          reservedContextSize:
            agent.kimiConfig?.loopControl?.reservedContextSize ??
            DEFAULT_COMPACTION_CONFIG.reservedContextSize,
        },
      );
  }

  get isCompacting(): boolean {
    return this.compacting !== null;
  }

  getEffectiveMaxContextTokens(): number {
    const configured = this.agent.config.modelCapabilities.max_context_tokens;
    const modelAlias = this.agent.config.modelAlias;
    const observed =
      modelAlias === undefined ? undefined : this.observedMaxContextTokensByModel.get(modelAlias);
    if (observed === undefined) return configured;
    if (configured <= 0) return observed;
    return Math.min(configured, observed);
  }

  estimateCurrentRequestTokens(): number {
    return this.estimateRequestTokens(this.agent.context.messages);
  }

  shouldRecoverFromContextOverflow(
    error: unknown,
    estimatedRequestTokens = this.estimateCurrentRequestTokens(),
  ): boolean {
    if (error instanceof APIContextOverflowError) return true;
    if (!(error instanceof APIStatusError) || error.statusCode !== 413) return false;
    const effectiveMax = this.getEffectiveMaxContextTokens();
    return (
      effectiveMax > 0 && estimatedRequestTokens >= effectiveMax * OVERFLOW_STATUS_RECOVERY_RATIO
    );
  }

  observeContextOverflow(estimatedRequestTokens: number): void {
    if (!Number.isFinite(estimatedRequestTokens) || estimatedRequestTokens <= 0) return;
    const modelAlias = this.agent.config.modelAlias;
    if (modelAlias === undefined) return;
    const observed = Math.max(
      1,
      Math.floor(estimatedRequestTokens * OVERFLOW_CONTEXT_SAFETY_RATIO),
    );
    const current = this.getEffectiveMaxContextTokens();
    if (current > 0 && observed >= current) return;
    this.observedMaxContextTokensByModel.set(modelAlias, observed);
  }

  begin(data: Readonly<CompactionBeginData>): void {
    if (this.compacting) return;
    if (data.source === 'manual') {
      this.compactionCountInTurn = 0;
    } else {
      this.compactionCountInTurn += 1;
    }
    if (this.compactionCountInTurn > this.strategy.maxCompactionPerTurn) return;
    if (this.agent.records.restoring) {
      const replay = this.agent.replayBuilder.push({
        type: 'compaction',
        instruction: data.instruction,
      });
      if (replay?.type === 'compaction') {
        this.restoredCompaction = { data, replay };
        this.subscribeRestoreRecovery();
      }
      return;
    }
    if (this.agent.context.history.length === 0) {
      throw new KimiError(ErrorCodes.COMPACTION_UNABLE, 'No messages to compact in current history.');
    }
    // Manual (SDK/REST) compaction must not start while a turn is running: the
    // turn keeps mutating the context (streaming content, appending messages)
    // while the summarizer is in flight, and that output is then neither
    // summarized nor preserved by the rebuild. Auto compaction is exempt — it is
    // triggered from within the turn at a step boundary, which blocks the turn
    // for the duration. Refuse manual compaction here so it only runs at a clean
    // boundary; the caller can retry once the turn finishes.
    if (data.source === 'manual' && this.agent.turn.hasActiveTurn) {
      throw new KimiError(
        ErrorCodes.COMPACTION_UNABLE,
        'Cannot compact while a turn is active. Wait for it to finish, then retry.',
      );
    }
    const abortController = new AbortController();
    const promise = createControlledPromise<void>();
    const reservation: NonNullable<FullCompaction['compacting']> = {
      abortController,
      promise,
      blockedByTurn: false,
      phase: 'running',
      completionRecordWritten: false,
      completionEventEmitted: false,
      postCompactHookTriggered: false,
      released: false,
    };
    this.compacting = reservation;
    void promise.catch(() => undefined);
    try {
      this.agent.records.logRecord({
        type: 'full_compaction.begin',
        ...data,
      });
    } catch (error) {
      if (this.compacting === reservation) this.compacting = null;
      abortController.abort(error);
      promise.reject(error);
      throw error;
    }
    if (this.compacting !== reservation || abortController.signal.aborted) {
      if (this.compacting === reservation) this.compacting = null;
      promise.resolve();
      this.agent.turn.onCompactionFinished();
      return;
    }
    try {
      this.agent.emitEvent({
        type: 'compaction.started',
        trigger: data.source,
        instruction: data.instruction,
      });
    } catch (error) {
      if (this.compacting === reservation) this.compacting = null;
      abortController.abort(error);
      promise.reject(error);
      this.agent.turn.onCompactionFinished();
      throw error;
    }
    if (abortController.signal.aborted) {
      if (this.compacting === reservation) this.compacting = null;
      promise.resolve();
      this.agent.turn.onCompactionFinished();
      return;
    }
    const worker = this.compactionWorker(abortController.signal, data);
    void worker.then(promise.resolve, promise.reject);
  }

  cancel(): Promise<void> {
    if (this.agent.records.restoring) {
      this.restoredCompaction = null;
      this.agent.replayBuilder.patchLast('compaction', { result: 'cancelled' });
      return Promise.resolve();
    }
    const active = this.compacting;
    if (!active) {
      this.agent.replayBuilder.patchLast('compaction', { result: 'cancelled' });
      return Promise.resolve();
    }
    // Once context application begins, cancellation would create a half-
    // committed state by skipping prompt refresh/reinjection. From this point
    // callers join the terminal barrier without changing the outcome.
    if (active.phase === 'cancelling') return active.cancellation ?? active.promise;
    if (active.phase !== 'running') return active.promise;
    if (active.abortController.signal.aborted) return active.promise;
    // Publish the cancellation barrier before invoking persistence callbacks:
    // a synchronous observer may re-enter cancel(), and must join this exact
    // operation instead of recursively appending cancellation records.
    const cancellation = createControlledPromise<void>();
    active.cancellation = cancellation;
    active.phase = 'cancelling';
    void cancellation.catch(() => undefined);
    this.agent.replayBuilder.patchLast('compaction', { result: 'cancelled' });
    let recordFailed = false;
    let recordError: unknown;
    try {
      this.agent.records.logRecord({
        type: 'full_compaction.cancel',
      });
    } catch (error) {
      recordFailed = true;
      recordError = error;
    }
    active.abortController.abort();
    this.agent.emitEvent({ type: 'compaction.cancelled' });
    const barrier = active.promise.then(
      () => {
        if (recordFailed) throw recordError;
      },
      (error: unknown) => {
        throw recordFailed ? recordError : error;
      },
    );
    void barrier.then(cancellation.resolve, cancellation.reject);
    return cancellation;
  }

  private cancelInBackground(): void {
    void this.cancel().catch((error: unknown) => {
      this.logPostCommitFailure('failed to persist compaction cancellation', error);
    });
  }

  markCompleted(): void {
    if (this.agent.records.restoring) {
      this.restoredCompaction = null;
      return;
    }
    this.writeCompletionRecord(this.compacting);
  }

  private writeCompletionRecord(active: ActiveCompaction | null): void {
    if (active?.completionRecordWritten === true) return;
    try {
      this.agent.records.logRecord({
        type: 'full_compaction.complete',
      });
      if (active !== null) active.completionRecordWritten = true;
    } catch (error) {
      // Built-in persistence implementations report which side of the append
      // boundary failed. Retry only an explicit pre-accept rejection; unknown
      // third-party semantics stay conservative to avoid two durable terminals.
      if (active !== null && agentRecordAppendAccepted(error) !== false) {
        active.completionRecordWritten = true;
      }
      throw error;
    }
  }

  private get tokenCountWithPending(): number {
    return this.agent.context.tokenCountWithPending;
  }

  private estimateRequestTokens(messages: readonly Message[]): number {
    return (
      estimateTokens(this.agent.config.systemPrompt) +
      // Deferred tools never reach the outbound top-level tools[] (kosong
      // generate() strips them); keep the estimate aligned with the wire.
      estimateTokensForTools(this.agent.tools.loopTools.filter((t) => t.deferred !== true)) +
      estimateTokensForMessages(messages)
    );
  }

  resetForTurn(): void {
    this.compactionCountInTurn = 0;
    this.lastCompactedTokenCount = null;
    this.consecutiveOverflowCompactions = 0;
  }

  /**
   * Atomically identifies the prompt text produced by the latest successful
   * compaction refresh. Keeping the revision and text together prevents an
   * unrelated config update between compaction and the next step from being
   * mistaken for the prompt that compaction rendered.
   */
  get systemPromptRefreshSnapshot(): Readonly<{
    revision: number;
    systemPrompt: string | undefined;
  }> {
    return this._systemPromptRefreshSnapshot;
  }

  async handleOverflowError(signal: AbortSignal, error: unknown) {
    this.consecutiveOverflowCompactions += 1;
    const maxAttempts = this.strategy.maxOverflowCompactionAttempts;
    if (this.consecutiveOverflowCompactions > maxAttempts) {
      throw new KimiError(
        ErrorCodes.CONTEXT_OVERFLOW,
        `Compaction failed to bring the context under the model window after ${String(maxAttempts)} attempts.`,
        { cause: error instanceof Error ? error : undefined },
      );
    }
    const didStartCompaction = this.beginAutoCompaction();
    if (!didStartCompaction && !this.compacting) throw error;
    // Always block on overflow errors
    await this.block(signal);
  }

  async beforeStep(signal: AbortSignal): Promise<void> {
    await this.retryPendingPostCommitMaintenance();
    this.checkAutoCompaction();
    if (this.strategy.shouldBlock(this.tokenCountWithPending)) {
      await this.block(signal);
    }
  }

  async afterStep(): Promise<void> {
    // A completed step means a generate() succeeded, so any prior
    // overflow -> compact cycle produced a request that now fits; clear the
    // loop guard.
    this.consecutiveOverflowCompactions = 0;
    if (this.strategy.checkAfterStep) {
      this.checkAutoCompaction(false);
    }
    // Do not block after the step
  }

  private checkAutoCompaction(throwOnLimit: boolean = true): boolean {
    if (this.compacting) return true;
    if (
      this.lastCompactedTokenCount !== null &&
      this.tokenCountWithPending <= this.lastCompactedTokenCount
    ) {
      return false;
    }
    if (!this.strategy.shouldCompact(this.tokenCountWithPending)) return false;
    return this.beginAutoCompaction(throwOnLimit);
  }

  private beginAutoCompaction(throwOnLimit: boolean = true): boolean {
    if (this.compacting) return true;
    const maxCompactions = this.strategy.maxCompactionPerTurn;
    if (this.compactionCountInTurn >= maxCompactions) {
      if (throwOnLimit) {
        throw new KimiError(ErrorCodes.CONTEXT_OVERFLOW, `Compaction limit exceeded (${String(maxCompactions)})`, {
          details: { maxCompactions },
        });
      }
      return false;
    }
    this.begin({ source: 'auto', instruction: undefined });
    return this.compacting !== null;
  }

  private async block(signal: AbortSignal): Promise<void> {
    const active = this.compacting;
    if (active) {
      active.blockedByTurn = true;
      signal.addEventListener('abort', () => {
        if (this.compacting === active) {
          this.cancelInBackground();
        }
      });
      this.agent.emitEvent({
        type: 'compaction.blocked',
        turnId: this.agent.turn.currentId,
      });
      await active.promise;
    }
  }

  private async compactionWorker(
    signal: AbortSignal,
    data: Readonly<CompactionBeginData>,
  ): Promise<void> {
    const active = this.compacting;
    try {
      const result = await this.compactionRound(signal, data);
      if (!result) return;
      if (active === null || this.compacting !== active) return;
      await this.convergeCommittedCompaction(
        active,
        data,
        result,
        LIVE_POST_COMMIT_OPTIONS,
      );
    } catch (error) {
      // applyCompaction is the commit point. Anything that throws after it must
      // converge to the one completed terminal; treating it as cancellation is
      // both false (the context is already replaced) and used to self-wait on
      // active.promise from inside the worker that owns that promise.
      if (
        active !== null &&
        this.compacting === active &&
        active.phase === 'committed' &&
        active.committedResult !== undefined
      ) {
        this.logPostCommitFailure('post-commit compaction work failed', error);
        await this.convergeCommittedCompaction(
          active,
          data,
          active.committedResult,
          LIVE_POST_COMMIT_OPTIONS,
        );
        return;
      }
      // AbortError is not proof of cancellation: providers sometimes throw one
      // on their own. The controller is the source of truth for whether this
      // particular compaction was actually cancelled.
      if (signal.aborted) return;
      const blockedByTurn = active?.blockedByTurn === true;
      this.logCompactionFailure(error);
      if (blockedByTurn) {
        throw error;
      }
      this.agent.emitEvent({
        type: 'error',
        ...toKimiErrorPayload(error),
      });
    } finally {
      // Replay prompts/steers deferred while compaction held the context — on the
      // success path (after reinjection above), on an A1 prefix/tail cancel
      // (`!result`), and on failure/abort. A cancelled worker retains its
      // reservation through asynchronous cleanup; release that exact worker now
      // so replay cannot overlap it and can actually start instead of re-buffering.
      if (active !== null) this.releaseCompaction(active);
    }
  }

  private subscribeRestoreRecovery(): void {
    if (this.restoreRecoverySubscribed) return;
    this.restoreRecoverySubscribed = true;
    this.agent.records.onOpened(() => {
      try {
        this.prepareRestoredCompactionRecovery();
      } catch (error) {
        this.logPostCommitFailure('failed to prepare restored compaction recovery', error);
      }
    });
  }

  private prepareRestoredCompactionRecovery(): void {
    this.scheduleRestoredPostCompactionMaintenance();
    const restored = this.restoredCompaction;
    this.restoredCompaction = null;
    const result = restored?.replay.result;
    // A begin record without context.apply_compaction is still pre-commit. It
    // has no state to finalize and, crucially, must not be re-labelled as a
    // cancellation just because the process exited.
    if (restored === null || result === undefined || result === 'cancelled') return;
    if (this.compacting !== null) {
      this.logPostCommitFailure(
        'cannot recover committed compaction while another compaction is active',
        new Error('Compaction reservation collision during restore'),
      );
      return;
    }

    const abortController = new AbortController();
    const promise = createControlledPromise<void>();
    const active: ActiveCompaction = {
      abortController,
      promise,
      blockedByTurn: false,
      phase: 'committed',
      committedResult: result,
      completionRecordWritten: false,
      completionEventEmitted: false,
      postCompactHookTriggered: false,
      released: false,
    };
    this.compacting = active;
    void promise.catch(() => undefined);
    this.restoredCompactionRecovery = {
      active,
      data: restored.data,
      result,
      started: false,
    };
  }

  private scheduleRestoredPostCompactionMaintenance(): void {
    const hasCompactedContext = this.agent.context.history.some(
      (message) => message.origin?.kind === 'compaction_summary',
    );
    if (!hasCompactedContext) return;
    // Even a durable `full_compaction.complete` may have been followed by a
    // process exit before transient refresh/injection maintenance retried at
    // the next request boundary. These flags are intentionally reconstructed
    // from the surviving summary on every resume. The retry path deduplicates
    // reminders that were already durable, so this closes that crash window
    // without stacking a second copy after ordinary completed compactions.
    this.systemPromptRefreshPending = true;
    this.postCompactionInjectionPending = true;
  }

  /**
   * Finish a committed compaction discovered during record replay. Agent.resume
   * calls this only after background/cron/context/turn restore has settled. The
   * reservation prepared in onOpened keeps TurnFlow from starting restored
   * deferred work before that point.
   */
  async finishResume(): Promise<void> {
    const recovery = this.restoredCompactionRecovery;
    if (recovery === null) return;
    if (!recovery.started) {
      recovery.started = true;
      const worker = this.recoveredCompactionWorker(
        recovery.active,
        recovery.data,
        recovery.result,
      );
      void worker.then(recovery.active.promise.resolve, recovery.active.promise.reject);
    }
    await recovery.active.promise;
    if (this.restoredCompactionRecovery === recovery) {
      this.restoredCompactionRecovery = null;
    }
  }

  private async recoveredCompactionWorker(
    active: ActiveCompaction,
    data: Readonly<CompactionBeginData>,
    result: CompactionResult,
  ): Promise<void> {
    try {
      await this.convergeCommittedCompaction(
        active,
        data,
        result,
        RECOVERED_POST_COMMIT_OPTIONS,
      );
    } catch (error) {
      // convergeCommittedCompaction isolates every stage, but keep the recovery
      // worker fail-open so an unexpected diagnostic failure cannot strand the
      // restored agent in `isCompacting` forever.
      this.logPostCommitFailure('restored compaction convergence failed', error);
      await this.ensureCompletionRecord(active);
    } finally {
      this.releaseCompaction(active);
    }
  }

  private convergeCommittedCompaction(
    active: ActiveCompaction,
    data: Readonly<CompactionBeginData>,
    result: CompactionResult,
    options: PostCommitOptions,
  ): Promise<void> {
    if (active.postCommit !== undefined) return active.postCommit;
    // Defer the body by one microtask so the joinable promise is published
    // before any synchronous persistence callback can re-enter cancellation.
    const postCommit = Promise.resolve().then(async () => {
      if (options.deferSystemPromptRefresh) {
        this.systemPromptRefreshPending = true;
      } else {
        await this.refreshSystemPromptAfterCompaction();
      }
      await this.injectAfterCompactionWithRetry();
      this.capturePostCompactionTokenFloor();
      await this.ensureCompletionRecord(active);
      this.emitCompletionOnce(active, result, options.emitCompletionEvent);
      if (options.triggerPostCompactHook) {
        this.triggerPostCompactHookOnce(active, data, result);
      }
      // Keep the reservation through synchronous completion observers, then
      // release and replay deferred work before those observers' promises
      // resume on the next microtask.
      this.releaseCompaction(active);
    });
    active.postCommit = postCommit;
    return postCommit;
  }

  private async retryPendingPostCommitMaintenance(): Promise<void> {
    if (this.systemPromptRefreshPending) {
      await this.refreshSystemPromptAfterCompaction();
    }
    if (this.postCompactionInjectionPending) {
      await this.injectAfterCompactionWithRetry();
      this.capturePostCompactionTokenFloor();
    }
  }

  private async refreshSystemPromptAfterCompaction(): Promise<void> {
    this.systemPromptRefreshPending = true;
    for (let attempt = 1; attempt <= POST_COMMIT_ATTEMPTS; attempt += 1) {
      try {
        const systemPrompt = await this.agent.refreshSystemPrompt();
        if (systemPrompt !== undefined) {
          this._systemPromptRefreshSnapshot = {
            revision: this._systemPromptRefreshSnapshot.revision + 1,
            systemPrompt,
          };
        }
        this.systemPromptRefreshPending = false;
        return;
      } catch (error) {
        this.logPostCommitFailure(
          `failed to refresh system prompt after compaction (attempt ${String(attempt)})`,
          error,
        );
      }
    }
  }

  private async injectAfterCompactionWithRetry(): Promise<void> {
    this.postCompactionInjectionPending = true;
    for (let attempt = 1; attempt <= POST_COMMIT_ATTEMPTS; attempt += 1) {
      try {
        await this.injectAfterCompactionIdempotently();
        this.postCompactionInjectionPending = false;
        return;
      } catch (error) {
        this.logPostCommitFailure(
          `failed to inject post-compaction reminders (attempt ${String(attempt)})`,
          error,
        );
        // DynamicInjector advances its cursor before appendSystemReminder. If
        // persistence throws before accepting that append, the cursor would
        // otherwise make the retry look successful while permanently skipping
        // the missing reminder. Reset lifecycle cursors; exact durable reminders
        // are still consumed by injectAfterCompactionIdempotently on the retry.
        this.agent.injection.onContextCompacted();
      }
    }
  }

  /**
   * Re-running InjectionManager after a crash is necessary, but doing so
   * blindly duplicates any reminder records that made it to disk before the
   * crash. Interpose only on the manager's append boundary and consume exact
   * origin+content matches already present after the latest summary. Missing
   * reminders still use ContextMemory's normal append path, so persistence,
   * replay and injector bookkeeping stay aligned.
   */
  private async injectAfterCompactionIdempotently(): Promise<void> {
    const context = this.agent.context;
    const existing = this.postCompactionReminderCounts();
    // Kept unbound so the exact original property can be restored in finally;
    // calls below always provide ContextMemory explicitly.
    // oxlint-disable-next-line typescript-eslint/unbound-method
    const original = context.appendSystemReminder;
    const wrapper = (
      content: string,
      origin: PromptOrigin,
      consumedTurnInput?: TurnInputConsumption,
      materializedTurnOutcomeId?: string,
    ): void => {
      const signature = systemReminderSignature(content, origin);
      const remaining = existing.get(signature) ?? 0;
      if (remaining > 0) {
        existing.set(signature, remaining - 1);
        return;
      }
      original.call(
        context,
        content,
        origin,
        consumedTurnInput,
        materializedTurnOutcomeId,
      );
    };
    const mutableContext = context as unknown as {
      appendSystemReminder(
        content: string,
        origin: PromptOrigin,
        consumedTurnInput?: TurnInputConsumption,
        materializedTurnOutcomeId?: string,
      ): void;
    };
    mutableContext.appendSystemReminder = wrapper;
    try {
      await this.agent.injection.injectAfterCompaction();
    } finally {
      // Restore only our own interposition. If a test/host deliberately replaced
      // the method while injection awaited, do not overwrite that newer owner.
      if (mutableContext.appendSystemReminder === wrapper) {
        mutableContext.appendSystemReminder = original;
      }
    }
  }

  private postCompactionReminderCounts(): Map<string, number> {
    const history = this.agent.context.history;
    const summaryIndex = history.findLastIndex(
      (message) => message.origin?.kind === 'compaction_summary',
    );
    const counts = new Map<string, number>();
    for (const message of history.slice(summaryIndex + 1)) {
      const origin = message.origin;
      if (origin === undefined || message.role !== 'user') continue;
      if (message.content.length !== 1 || message.content[0]?.type !== 'text') continue;
      const content = unwrapSystemReminder(message.content[0].text);
      if (content === undefined) continue;
      const signature = systemReminderSignature(content, origin);
      counts.set(signature, (counts.get(signature) ?? 0) + 1);
    }
    return counts;
  }

  private capturePostCompactionTokenFloor(): void {
    try {
      // Reinjected reminders are part of the minimal post-compaction floor. A
      // lower baseline can immediately re-trigger a compaction that cannot make
      // the context any smaller.
      this.lastCompactedTokenCount = this.tokenCountWithPending;
    } catch (error) {
      this.logPostCommitFailure('failed to capture post-compaction token floor', error);
    }
  }

  private async ensureCompletionRecord(active: ActiveCompaction): Promise<void> {
    try {
      // Preserve the synchronous production path so persistence observers see
      // the reservation released in the same stack, while still accepting an
      // async fault-injected replacement in tests/hosts.
      const completion = (this.markCompleted as () => unknown)();
      if (isPromiseLike(completion)) await completion;
      return;
    } catch (error) {
      this.logPostCommitFailure('failed to mark compaction completed', error);
    }
    // A spy may throw before entering markCompleted. Bypass that replaceable
    // surface once; the per-reservation write guard still guarantees one record
    // when the first call accepted the record and then threw.
    if (!active.completionRecordWritten) {
      try {
        this.writeCompletionRecord(active);
      } catch (error) {
        this.logPostCommitFailure('failed to persist compaction completion fallback', error);
      }
    }
  }

  private emitCompletionOnce(
    active: ActiveCompaction,
    result: CompactionResult,
    enabled: boolean,
  ): void {
    if (!enabled || active.completionEventEmitted) return;
    active.completionEventEmitted = true;
    const { contextSummary: _contextSummary, ...eventResult } = result;
    void _contextSummary;
    this.agent.emitEvent({ type: 'compaction.completed', result: eventResult });
  }

  private triggerPostCompactHookOnce(
    active: ActiveCompaction,
    data: Readonly<CompactionBeginData>,
    result: CompactionResult,
  ): void {
    if (active.postCompactHookTriggered) return;
    active.postCompactHookTriggered = true;
    try {
      this.triggerPostCompactHook(data, result);
    } catch (error) {
      this.logPostCommitFailure('failed to trigger post-compaction hook', error);
    }
  }

  private releaseCompaction(active: ActiveCompaction): void {
    if (active.released) return;
    active.released = true;
    if (this.compacting === active) this.compacting = null;
    this.agent.turn.onCompactionFinished();
  }

  private logCompactionFailure(error: unknown): void {
    try {
      this.agent.log.error('compaction failed', { error });
    } catch {
      // Diagnostics must not alter the compaction terminal state.
    }
  }

  private logPostCommitFailure(message: string, error: unknown): void {
    try {
      this.agent.log.error(message, { error });
    } catch {
      // Diagnostics must not alter the compaction terminal state.
    }
  }

  private buildInstruction(customInstruction: string | undefined): string {
    return renderPrompt(compactionInstructionTemplate, {
      customInstruction: customInstruction?.trim() ?? '',
    }).trimEnd();
  }

  private postProcessSummary(summary: string): string {
    const storeData = this.agent.tools.storeData();
    const todos = (storeData[TODO_STORE_KEY] as readonly TodoItem[] | undefined) ?? [];
    if (todos.length === 0) {
      return summary;
    }
    const todoMarkdown = renderTodoList(todos, '## TODO List');
    return `${summary.trim()}\n\n${todoMarkdown}`;
  }

  private async compactionRound(
    signal: AbortSignal,
    data: Readonly<CompactionBeginData>,
  ): Promise<CompactionResult | undefined> {
    const startedAt = Date.now();
    const originalHistory = [...this.agent.context.history];
    const tokensBefore = estimateTokensForMessages(originalHistory);
    let retryCount = 0;
    try {
      await this.triggerPreCompactHook(data, tokensBefore, signal);

      const model = this.agent.config.model;
      const capability = this.agent.config.modelCapabilities;
      const maxContextTokens = capability.max_context_tokens;
      // When the model's context window is known and the user has not set
      // `maxOutputSize`, cap compaction output to a safe default so a large
      // context window does not push `max_tokens` past the provider's ceiling.
      // When the window is unknown (maxContextTokens === 0), leave
      // `maxOutputSize` unset so `resolveCompletionBudget` falls back to the
      // conservative unknown-context fallback.
      const defaultCompactionCap =
        maxContextTokens > 0
          ? Math.min(maxContextTokens, DEFAULT_COMPACTION_MAX_COMPLETION_TOKENS)
          : undefined;
      const provider = applyCompletionBudget({
        provider: this.agent.config.provider,
        budget: resolveCompletionBudget({
          maxOutputSize: this.agent.config.maxOutputSize ?? defaultCompactionCap,
          reservedContextSize: this.agent.kimiConfig?.loopControl?.reservedContextSize,
        }),
        capability,
      });
      const instruction = this.buildInstruction(data.instruction);

      const delays = retryBackoffDelays(MAX_COMPACTION_RETRY_ATTEMPTS);
      let usage: TokenUsage | null = null;
      let summary: string | undefined;
      // Compact the whole history, trimming old messages only when the
      // summarizer request itself cannot fit. Any trimmed messages are not
      // covered by the produced summary; `droppedCount` reports that blind spot.
      // Dynamic-tool protocol context (schema messages, loadable-tools
      // announcements) is excluded from the summarizer input entirely: it is
      // protocol state, not conversation — summarizing it wastes tokens and
      // risks schema text leaking into the summary. The post-compaction
      // boundary re-announces the manifest; the schemas themselves are
      // deliberately dropped (discard-on-compaction) and re-selectable on
      // demand. Must happen before project() (which strips the origin
      // anchor). `originalHistory` itself stays untouched for the
      // prefix-race check and `compactedCount`.
      let historyForModel: readonly ContextMessage[] = stripDynamicToolContext(originalHistory);
      let droppedCount = 0;
      let mediaStripAttempted = false;
      let overflowShrinkCount = 0;
      let emptyOrTruncatedShrinkCount = 0;
      while (true) {
        // A request-building projection: close still-open calls in the sliced
        // prefix (synthesizeMissing) and drop stray results with no call anywhere
        // (dropOrphanResults), so the summarizer request cannot be rejected by a
        // strict provider even when the history carries a legacy-restore orphan.
        const messages = [
          ...this.agent.context.project(historyForModel, {
            synthesizeMissing: true,
            dropOrphanResults: true,
          }),
          createUserMessage(instruction),
        ];
        const estimatedCompactionRequestTokens = this.estimateRequestTokens(messages);
        try {
          const generateOptions: GenerateOptionsWithRequestLogFields = {
            signal,
            requestLogFields: { kind: 'compaction', droppedCount },
          };
          const response = await this.agent.generate(
            provider,
            this.agent.config.systemPrompt,
            [...this.agent.tools.loopTools],
            messages,
            undefined,
            generateOptions,
          );
          if (response.finishReason === 'truncated') {
            throw new CompactionTruncatedError();
          }
          usage = response.usage;
          summary = extractCompactionSummary(response);
          break;
        } catch (error) {
          // A request-body-size rejection (HTTP 413) or an image-format
          // rejection is first retried with media parts replaced by text
          // markers: accumulated base64 payloads are the usual 413 culprit,
          // a poisoned image the format-rejection culprit, and a text summary
          // needs neither — the conversation already narrates what was seen,
          // and the ReadMediaFile `<image path="...">` text wrapper survives.
          // Only the summarizer input copy is rewritten; the real history
          // keeps its media. A rejection after the strip (or with no media to
          // strip) falls through to the overflow shrink below for a 413, and
          // propagates for a format error — dropping oldest messages cannot
          // fix a poisoned image's format.
          const mediaRejected =
            error instanceof APIRequestTooLargeError || isImageFormatError(error);
          if (mediaRejected && !mediaStripAttempted) {
            mediaStripAttempted = true;
            const stripped = replaceMediaPartsWithMarkers(historyForModel);
            if (stripped !== historyForModel) {
              historyForModel = stripped;
              retryCount = 0;
              continue;
            }
          }
          const isContextOverflow = this.shouldRecoverFromContextOverflow(
            error,
            estimatedCompactionRequestTokens,
          );
          if (isContextOverflow) {
            this.observeContextOverflow(estimatedCompactionRequestTokens);
          }
          const shouldShrinkAfterOverflow =
            isContextOverflow || error instanceof APIRequestTooLargeError;
          if (shouldShrinkAfterOverflow && historyForModel.length > 1) {
            overflowShrinkCount += 1;
            if (overflowShrinkCount > MAX_COMPACTION_OVERFLOW_SHRINK_ATTEMPTS) {
              throw error;
            }
            const before = historyForModel.length;
            historyForModel = shrinkCompactionHistoryAfterOverflow(
              historyForModel,
              overflowShrinkCount,
            );
            droppedCount += before - historyForModel.length;
            retryCount = 0;
            continue;
          }
          const shouldShrinkAfterEmptyOrTruncated =
            error instanceof CompactionTruncatedError ||
            error instanceof APIEmptyResponseError;
          if (shouldShrinkAfterEmptyOrTruncated && historyForModel.length > 1) {
            // Each empty/truncated summary drops the oldest message and retries,
            // but without its own bound this would issue ~one request per message
            // (resetting retryCount sidesteps the transient-error budget). Cap the
            // shrink attempts by the same retry budget so a model that keeps
            // returning empty cannot fan out into a request per history entry.
            emptyOrTruncatedShrinkCount += 1;
            if (emptyOrTruncatedShrinkCount > MAX_COMPACTION_RETRY_ATTEMPTS) {
              throw error;
            }
            const before = historyForModel.length;
            historyForModel = dropOldestMessageAndLeadingToolResults(historyForModel);
            droppedCount += before - historyForModel.length;
            retryCount = 0;
            continue;
          }
          if (!isRetryableGenerateError(error)) {
            throw error;
          }
          if (retryCount + 1 >= MAX_COMPACTION_RETRY_ATTEMPTS) {
            throw error;
          }
          await sleepForRetry(delays[retryCount]!, signal);
          retryCount += 1;
        }
      }

      // A provider may ignore AbortSignal and resolve after cancellation. Do
      // not record its usage or let its stale summary rebuild context while a
      // concurrently arriving prompt is waiting behind the cancellation
      // reservation.
      signal.throwIfAborted();
      if (usage !== null) {
        this.agent.usage.record(model, usage);
      }

      const newHistory = this.agent.context.history;
      for (let i = 0; i < originalHistory.length; i++) {
        if (newHistory[i] !== originalHistory[i]) {
          // The compacted prefix changed under us (e.g. undo). Bail.
          this.cancelInBackground();
          return undefined;
        }
      }
      // The prefix is intact, but the tail grew while the summarizer was in
      // flight (a live step racing a manual/SDK compaction). A real user message
      // is safe — the all-user rebuild picks recent user input back up from the
      // grown history — but anything compaction would drop (an assistant/tool
      // turn, or a user-role message like a background-task notification, hook/
      // cron reminder, or shell output) was neither summarized (the summary only
      // covers originalHistory) nor kept, so it would silently vanish. Cancel and
      // let a later clean-boundary compaction handle it.
      if (newHistory.slice(originalHistory.length).some((message) => !isRealUserInput(message))) {
        this.cancelInBackground();
        return undefined;
      }

      const rawSummary = this.postProcessSummary(summary ?? '');
      const contextSummary = buildCompactionSummaryText(rawSummary);
      signal.throwIfAborted();
      const active = this.compacting;
      if (!active || active.abortController.signal !== signal) return undefined;
      active.phase = 'committing';
      let result: CompactionResult;
      try {
        result = this.agent.context.applyCompaction({
          summary: rawSummary,
          contextSummary,
          compactedCount: originalHistory.length,
          tokensBefore,
          droppedCount: droppedCount === 0 ? undefined : droppedCount,
        });
      } catch (error) {
        if (this.compacting === active) active.phase = 'running';
        throw error;
      }
      if (this.compacting === active) {
        active.committedResult = result;
        active.phase = 'committed';
      }
      // Loaded dynamic tool schemas are deliberately NOT rebuilt: compaction
      // discards the loaded set entirely (the boundary announcement re-lists
      // every loadable name, and the model re-selects what it still needs).
      // Everything downstream already treats the empty loaded set as its
      // consistent base state — the ledger scan finds no schema messages, the
      // pending set was cleared by applyCompaction, deferred extras drop out
      // of the executable table, and a from-memory call is rejected by
      // preflight with select guidance.

      // Telemetry keys are snake_case, but the `context.apply_compaction`
      // record written below keeps its persisted camelCase field names
      // (consumed by external projectors). The two channels intentionally
      // diverge — don't rename the record side to match.
      try {
        this.agent.telemetry.track('compaction_finished', {
          source: data.source,
          tokens_before: result.tokensBefore,
          tokens_after: result.tokensAfter,
          duration_ms: Date.now() - startedAt,
          compacted_count: result.compactedCount,
          dropped_count: result.droppedCount,
          retry_count: retryCount,
          round: 1,
          thinking_effort: this.agent.config.thinkingEffort,
          ...(usage === null
            ? {}
            : { input_tokens: inputTotal(usage), output_tokens: usage.output }),
        });
      } catch (error) {
        // Telemetry is observational. Once context application committed, a
        // broken sink must not divert the state machine away from completion.
        this.logPostCommitFailure('failed to record compaction telemetry', error);
      }
      // Baseline the "nothing new since compaction" guard on the live counter
      // (== result.tokensAfter here, since nothing has been appended since
      // applyCompaction). compactionWorker raises it once more after
      // injectAfterCompaction so the reinjected reminders join the floor;
      // this earlier capture stays as the fallback when reinjection throws.
      this.lastCompactedTokenCount = this.tokenCountWithPending;
      return result;
    } catch (error) {
      if (
        this.compacting?.phase === 'committed' &&
        this.compacting.committedResult !== undefined
      ) {
        throw error;
      }
      if (signal.aborted) return undefined;
      this.agent.telemetry.track('compaction_failed', {
        source: data.source,
        tokens_before: tokensBefore,
        duration_ms: Date.now() - startedAt,
        round: 1,
        retry_count: retryCount,
        thinking_effort: this.agent.config.thinkingEffort,
        error_type: error instanceof Error ? error.name : 'Unknown',
      });
      if (
        isKimiError(error) &&
        (error.code === ErrorCodes.AUTH_LOGIN_REQUIRED ||
          error.code === ErrorCodes.PROVIDER_AUTH_ERROR)
      )
        throw error;
      throw new KimiError(ErrorCodes.COMPACTION_FAILED, String(error), { cause: error });
    }
  }

  private async triggerPreCompactHook(
    data: Readonly<CompactionBeginData>,
    tokenCount: number,
    signal: AbortSignal,
  ): Promise<void> {
    signal.throwIfAborted();
    await this.agent.hooks?.trigger('PreCompact', {
      matcherValue: data.source,
      signal,
      inputData: {
        trigger: data.source,
        tokenCount,
      },
    });
    signal.throwIfAborted();
  }

  private triggerPostCompactHook(
    data: Readonly<CompactionBeginData>,
    result: CompactionResult,
  ): void {
    void this.agent.hooks?.fireAndForgetTrigger('PostCompact', {
      matcherValue: data.source,
      inputData: {
        trigger: data.source,
        estimatedTokenCount: result.tokensAfter,
      },
    });
  }
}

function systemReminderSignature(content: string, origin: PromptOrigin): string {
  return JSON.stringify([
    origin,
    `<system-reminder>\n${content.trim()}\n</system-reminder>`,
  ]);
}

function unwrapSystemReminder(text: string): string | undefined {
  const prefix = '<system-reminder>\n';
  const suffix = '\n</system-reminder>';
  if (!text.startsWith(prefix) || !text.endsWith(suffix)) return undefined;
  return text.slice(prefix.length, -suffix.length);
}

function isPromiseLike(value: unknown): value is PromiseLike<unknown> {
  return (
    (typeof value === 'object' || typeof value === 'function') &&
    value !== null &&
    'then' in value &&
    typeof value.then === 'function'
  );
}

const MAX_COMPACTION_OVERFLOW_SHRINK_ATTEMPTS = 3;
const COMPACTION_OVERFLOW_SHRINK_RATIOS = [0.7, 0.5, 0.35] as const;

const MEDIA_PART_MARKERS = {
  image_url: '[image]',
  audio_url: '[audio]',
  video_url: '[video]',
} as const;

function isMediaPart(part: ContentPart): part is ContentPart & { type: keyof typeof MEDIA_PART_MARKERS } {
  return part.type in MEDIA_PART_MARKERS;
}

/**
 * Replace media parts (image/audio/video) with text markers in the summarizer
 * input, for the 413 strip-and-retry above. Messages without media are
 * returned by reference (keeping the per-message token-estimate cache warm),
 * and when nothing changed the input array itself is returned so the caller
 * can tell there was no media to strip.
 */
function replaceMediaPartsWithMarkers(
  messages: readonly ContextMessage[],
): readonly ContextMessage[] {
  let changed = false;
  const out = messages.map((message) => {
    if (!message.content.some(isMediaPart)) return message;
    changed = true;
    return {
      ...message,
      content: message.content.map((part): ContentPart =>
        isMediaPart(part) ? { type: 'text', text: MEDIA_PART_MARKERS[part.type] } : part,
      ),
    };
  });
  return changed ? out : messages;
}

function shrinkCompactionHistoryAfterOverflow<T extends Message>(
  messages: readonly T[],
  attempt: number,
): T[] {
  if (messages.length <= 1) return messages.slice();
  const ratio = COMPACTION_OVERFLOW_SHRINK_RATIOS[
    Math.min(attempt - 1, COMPACTION_OVERFLOW_SHRINK_RATIOS.length - 1)
  ]!;
  const tokenBudget = Math.floor(estimateTokensForMessages(messages) * ratio);
  return takeRecentMessagesWithinTokenBudget(messages, tokenBudget);
}

function takeRecentMessagesWithinTokenBudget<T extends Message>(
  messages: readonly T[],
  tokenBudget: number,
): T[] {
  let start = messages.length;
  let tokens = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const messageTokens = estimateTokensForMessage(messages[i]!);
    if (tokens + messageTokens > tokenBudget) break;
    tokens += messageTokens;
    start = i;
  }
  if (start === 0) start = 1;
  return dropLeadingToolResults(messages.slice(start));
}

function dropOldestMessageAndLeadingToolResults<T extends { readonly role: string }>(
  messages: readonly T[],
): T[] {
  if (messages.length <= 1) return messages.slice();
  return dropLeadingToolResults(messages.slice(1));
}

function dropLeadingToolResults<T extends { readonly role: string }>(messages: readonly T[]): T[] {
  let start = 0;
  while (start < messages.length && messages[start]!.role === 'tool') {
    start += 1;
  }
  return messages.slice(start);
}

function extractCompactionSummary(response: GenerateResult): string {
  const summary =
    typeof response.message.content === 'string'
      ? response.message.content
      : response.message.content.map((part) => (part.type === 'text' ? part.text : '')).join('');

  if (summary.trim().length === 0) {
    throw new APIEmptyResponseError(
      'The compaction response did not contain a non-empty summary.',
    );
  }
  return summary;
}
