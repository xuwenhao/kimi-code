import { Disposable, type IDisposable } from '#/_base/di/lifecycle';
import { InstantiationType } from '#/_base/di/extensions';
import { IInstantiationService } from '#/_base/di/instantiation';
import { LifecycleScope, registerScopedService } from '#/_base/di/scope';
import { ILogService } from '#/_base/log/log';
import { renderPrompt } from '#/_base/utils/render-prompt';
import {
  estimateTokens,
  estimateTokensForMessage,
  estimateTokensForMessages,
  estimateTokensForTools,
} from '#/_base/utils/tokens';
import {
  buildCompactionSummaryText,
  buildContextCompactionShape,
  isRealUserInput,
} from '#/agent/contextMemory/compactionHandoff';
import { IAgentContextInjectorService } from '#/agent/contextInjector/contextInjector';
import { IAgentContextMemoryService } from '#/agent/contextMemory/contextMemory';
import type { ContextMessage } from '#/agent/contextMemory/types';
import { IAgentContextSizeService } from '#/agent/contextSize/contextSize';
import { IAgentLLMRequesterService, type LLMRequestFinish } from '#/agent/llmRequester/llmRequester';
import { retryBackoffDelays, sleepForRetry } from '#/_base/utils/retry';
import { IAgentLoopService, type LoopErrorContext } from '#/agent/loop/loop';
import { isAbortError } from '#/_base/utils/abort';
import { IAgentProfileService, type ProfileModelContext } from '#/agent/profile/profile';
import { IAgentToolRegistryService } from '#/agent/toolRegistry/toolRegistry';
import { stripDynamicToolContext } from '#/agent/toolSelect/dynamicTools';
import { IAgentToolSelectService } from '#/agent/toolSelect/toolSelect';
import { IAgentActivityService } from '#/activity/activity';
import { ISessionTodoService } from '#/session/todo/sessionTodo';
import { renderTodoList, type TodoItem } from '#/session/todo/todoItem';
import {
  APIContextOverflowError,
  APIEmptyResponseError,
  APIStatusError,
  isRetryableGenerateError,
} from '#/app/llmProtocol/errors';
import { createUserMessage, type Message } from '#/app/llmProtocol/message';
import type { Tool } from '#/app/llmProtocol/tool';
import { inputTotal, type TokenUsage } from '#/app/llmProtocol/usage';
import { IEventBus } from '#/app/event/eventBus';
import type { CompactionFinishedEvent } from '#/app/telemetry/events';
import { ITelemetryService } from '#/app/telemetry/telemetry';
import {
  ErrorCodes,
  Error2,
  isCodedError,
  isError2,
  toKimiErrorPayload,
  unwrapErrorCause,
} from '#/errors';
import { IWireService } from '#/wire/wire';
import compactionInstructionTemplate from './compaction-instruction.md?raw';
import {
  IAgentFullCompactionService,
  type FullCompactionInput,
  type FullCompactionTask,
} from './fullCompaction';
import {
  RuntimeCompactionStrategy,
  type CompactionStrategy,
} from './strategy';
import {
  CompactionModel,
  fullCompactionBegin,
  fullCompactionCancel,
  fullCompactionComplete,
  fullCompactionFail,
} from './compactionOps';
import {
  type CompactionBeginData,
  type CompactionResult,
} from './types';
import { Emitter, type Event } from '#/_base/event';
import { OrderedHookSlot } from '#/hooks';

export const MAX_COMPACTION_RETRY_ATTEMPTS = 5;
const DEFAULT_COMPACTION_MAX_COMPLETION_TOKENS = 128 * 1024;
const OVERFLOW_CONTEXT_SAFETY_RATIO = 0.85;
const OVERFLOW_STATUS_RECOVERY_RATIO = 0.5;
const MAX_COMPACTION_OVERFLOW_SHRINK_ATTEMPTS = 3;
const COMPACTION_OVERFLOW_SHRINK_RATIOS = [0.7, 0.5, 0.35] as const;
const EMPTY_TOOL_PARAMETERS: Record<string, unknown> = {
  type: 'object',
  properties: {},
};

type CompactionTelemetryProperties = Pick<
  CompactionFinishedEvent,
  'input_tokens' | 'output_tokens' | 'input_cache_read' | 'input_cache_creation'
>;

interface ActiveCompaction extends FullCompactionTask {
  blockedByTurn: boolean;
  turnId?: number;
  phase: 'running' | 'cancelling' | 'committing' | 'committed' | 'finished';
  committedResult?: CompactionResult;
  bgRegistration?: IDisposable;
}

interface CompactionAttemptResult {
  readonly summary: string;
  readonly usage: TokenUsage | null;
}

class CompactionTruncatedError extends Error {
  constructor() {
    super('Compaction response was truncated before producing a complete summary.');
    this.name = 'CompactionTruncatedError';
  }
}

export class AgentFullCompactionService extends Disposable implements IAgentFullCompactionService {
  declare readonly _serviceBrand: undefined;
  readonly hooks: IAgentFullCompactionService['hooks'] = {
    onWillCompact: new OrderedHookSlot<FullCompactionTask>(),
  };
  private readonly _onDidFinishCompaction = this._register(new Emitter<FullCompactionTask>());
  readonly onDidFinishCompaction: Event<FullCompactionTask> = this._onDidFinishCompaction.event;

  private readonly strategy: CompactionStrategy;
  private compactionCountInTurn = 0;
  private _compacting: ActiveCompaction | null = null;
  private readonly observedMaxContextTokensByModel = new Map<string, number>();
  private lastCompactedTokenCount: number | null = null;
  private consecutiveOverflowCompactions = 0;
  private contextInjectorService: IAgentContextInjectorService | undefined;
  private systemPromptRefreshPending = false;
  private systemPromptRefreshTurnId: number | undefined;
  private contextReinjectionPending = false;
  private postCommitMaintenance: Promise<Error | undefined> | undefined;

  constructor(
    @IAgentContextMemoryService private readonly context: IAgentContextMemoryService,
    @IAgentContextSizeService private readonly contextSize: IAgentContextSizeService,
    @IAgentLLMRequesterService private readonly llmRequester: IAgentLLMRequesterService,
    @IAgentProfileService private readonly profile: IAgentProfileService,
    @IAgentToolRegistryService private readonly toolRegistry: IAgentToolRegistryService,
    @IAgentToolSelectService private readonly toolSelect: IAgentToolSelectService,
    @IInstantiationService private readonly instantiation: IInstantiationService,
    @ISessionTodoService private readonly todo: ISessionTodoService,
    @ITelemetryService private readonly telemetry: ITelemetryService,
    @IWireService private readonly wire: IWireService,
    @IEventBus private readonly eventBus: IEventBus,
    @IAgentActivityService private readonly activity: IAgentActivityService,
    @ILogService private readonly log: ILogService,
    @IAgentLoopService private readonly loopService: IAgentLoopService,
  ) {
    super();
    this.strategy = new RuntimeCompactionStrategy(() => this.resolveModelContextWithEffectiveMax());
    this._register(
      this.wire.hooks.onDidRestore.register('full-compaction', async (_ctx, next) => {
        await next();
        await this.normalizeAfterReplay();
      }),
    );
    this._register(
      this.eventBus.subscribe('turn.started', () => this.resetForTurn()),
    );
    this._register(
      this.loopService.hooks.onWillBeginStep.register('full-compaction', async (ctx, next) => {
        await this.beforeStep(ctx.signal, ctx.turnId);
        await next();
      }),
    );
    this._register(
      this.loopService.hooks.onDidFinishStep.register('full-compaction', async (ctx, next) => {
        await this.afterStep(ctx.turnId);
        await next();
      }),
    );
    this._register(
      this.loopService.registerLoopErrorHandler({
        id: 'full-compaction',
        match: (context) => this.shouldRecoverFromContextOverflow(context.error),
        handle: (context) => this.recoverFromContextOverflow(context),
      }),
    );
  }

  get compacting(): FullCompactionTask | null {
    return this._compacting;
  }

  private getEffectiveMaxContextTokens(): number {
    const configured = this.profile.data().modelCapabilities.max_context_tokens;
    const modelAlias = this.profile.data().modelAlias;
    const observed =
      modelAlias === undefined ? undefined : this.observedMaxContextTokensByModel.get(modelAlias);
    if (observed === undefined) return configured;
    if (configured <= 0) return observed;
    return Math.min(configured, observed);
  }

  private resolveModelContextWithEffectiveMax(): ProfileModelContext {
    const resolved = this.profile.resolveModelContext();
    return {
      ...resolved,
      modelCapabilities: {
        ...resolved.modelCapabilities,
        max_context_tokens: this.getEffectiveMaxContextTokens(),
      },
    };
  }

  private estimateCurrentRequestTokens(): number {
    return this.estimateRequestTokens(this.context.get());
  }

  private estimateRequestTokens(messages: readonly Message[]): number {
    return (
      estimateTokens(this.profile.getSystemPrompt()) +
      estimateTokensForTools(this.defaultTools().filter((tool) => tool.deferred !== true)) +
      estimateTokensForMessages(messages)
    );
  }

  private defaultTools(): readonly Tool[] {
    return this.toolSelect
      .shapeTools(this.toolRegistry.list())
      .map((tool) => ({
        name: tool.name,
        description: tool.description,
        parameters: tool.parameters ?? EMPTY_TOOL_PARAMETERS,
        deferred: tool.deferred,
      }));
  }

  private shouldRecoverFromContextOverflow(
    error: unknown,
    estimatedRequestTokens = this.estimateCurrentRequestTokens(),
  ): boolean {
    if (isCodedError(error) && error.code === ErrorCodes.CONTEXT_OVERFLOW) return true;
    const statusError = findAPIStatusError(error);
    if (statusError instanceof APIContextOverflowError) return true;
    if (statusError === undefined || statusError.statusCode !== 413) return false;
    const effectiveMax = this.getEffectiveMaxContextTokens();
    return (
      effectiveMax > 0 &&
      estimatedRequestTokens >= effectiveMax * OVERFLOW_STATUS_RECOVERY_RATIO
    );
  }

  private observeContextOverflow(estimatedRequestTokens: number): void {
    if (!Number.isFinite(estimatedRequestTokens) || estimatedRequestTokens <= 0) return;
    const modelAlias = this.profile.data().modelAlias;
    if (modelAlias === undefined) return;
    const observed = Math.max(
      1,
      Math.floor(estimatedRequestTokens * OVERFLOW_CONTEXT_SAFETY_RATIO),
    );
    const current = this.getEffectiveMaxContextTokens();
    if (current > 0 && observed >= current) return;
    this.observedMaxContextTokensByModel.set(modelAlias, observed);
  }

  begin(input: FullCompactionInput, turnId?: number): boolean {
    if (this._compacting || this.wire.getModel(CompactionModel).phase !== 'idle') return false;
    const data: CompactionBeginData = { source: input.source, instruction: input.instruction };
    if (!this.reserveCompactionSlot(data.source)) return false;

    const tokenCount = this.validateCompactionStart(data.source);
    this.wire.dispatch(fullCompactionBegin(data));

    const active = this.createActiveCompaction(data.source, tokenCount, turnId);
    this._compacting = active.task;
    active.task.abortController.signal.addEventListener(
      'abort',
      () => this.cancelActive(active.task),
      { once: true },
    );
    void this.compactionWorker(active.task, data).then(active.resolve, active.reject);
    void active.task.promise.catch(() => undefined);
    return true;
  }

  private reserveCompactionSlot(source: CompactionBeginData['source']): boolean {
    if (source === 'manual') {
      this.compactionCountInTurn = 0;
    } else {
      this.compactionCountInTurn += 1;
    }
    return this.compactionCountInTurn <= this.strategy.maxCompactionPerTurn;
  }

  private validateCompactionStart(source: CompactionBeginData['source']): number {
    const history = this.context.get();
    if (history.length === 0) {
      throw new Error2(ErrorCodes.COMPACTION_UNABLE, 'No messages to compact in current history.');
    }
    if (source === 'manual' && !this.activity.isIdle()) {
      throw new Error2(
        ErrorCodes.COMPACTION_UNABLE,
        'Cannot compact while a turn is active. Wait for it to finish, then retry.',
      );
    }
    return estimateTokensForMessages(history);
  }

  private createActiveCompaction(
    trigger: CompactionBeginData['source'],
    tokenCount: number,
    turnId?: number,
  ): {
    readonly task: ActiveCompaction;
    readonly resolve: (result: CompactionResult) => void;
    readonly reject: (reason: unknown) => void;
  } {
    const abortController = new AbortController();
    let resolve!: (result: CompactionResult) => void;
    let reject!: (reason: unknown) => void;
    const promise = new Promise<CompactionResult>((onResolve, onReject) => {
      resolve = onResolve;
      reject = onReject;
    });
    return {
      task: {
        abortController,
        promise,
        trigger,
        tokenCount,
        blockedByTurn: false,
        turnId,
        phase: 'running',
        bgRegistration: this.activity.registerBackground('compaction', abortController),
      },
      resolve,
      reject,
    };
  }

  private cancelActive(active: ActiveCompaction): boolean {
    if (this._compacting !== active || active.phase !== 'running') return false;
    active.phase = 'cancelling';
    this.wire.dispatch(fullCompactionCancel({}));
    this._compacting = null;
    active.bgRegistration?.dispose();
    if (!active.abortController.signal.aborted) {
      active.abortController.abort();
    }
    this.eventBus.publish({ type: 'compaction.cancelled' });
    return true;
  }

  private failActive(active: ActiveCompaction): boolean {
    if (
      this._compacting !== active ||
      (active.phase !== 'running' && active.phase !== 'committing')
    ) {
      return false;
    }
    active.phase = 'finished';
    this.wire.dispatch(fullCompactionFail());
    this._compacting = null;
    active.bgRegistration?.dispose();
    return true;
  }

  private releaseCommitted(active: ActiveCompaction): boolean {
    if (this._compacting !== active || active.phase !== 'committed') return false;
    active.phase = 'finished';
    this._compacting = null;
    active.bgRegistration?.dispose();
    return true;
  }

  private async normalizeAfterReplay(): Promise<void> {
    const phase = this.wire.getModel(CompactionModel).phase;
    if (phase === 'running') {
      this.wire.dispatch(fullCompactionFail());
      return;
    }
    if (phase !== 'committed') return;
    this.schedulePostCommitMaintenance();
    await this.retryPostCommitMaintenance(false);
  }

  private resetForTurn(): void {
    this.compactionCountInTurn = 0;
    this.lastCompactedTokenCount = null;
    this.consecutiveOverflowCompactions = 0;
  }

  private async recoverFromContextOverflow(
    context: LoopErrorContext,
  ): Promise<boolean> {
    this.recordOverflowRecovery(context.error);
    const didStartCompaction = this.beginAutoCompaction(true, context.turnId);
    if (!didStartCompaction && !this._compacting) return false;

    await this.block(context.signal, context.turnId);
    return this.retryFailedDriver(context);
  }

  private recordOverflowRecovery(error: unknown): void {
    this.observeContextOverflow(this.estimateCurrentRequestTokens());
    this.consecutiveOverflowCompactions += 1;
    const maxAttempts = this.strategy.maxOverflowCompactionAttempts;
    if (this.consecutiveOverflowCompactions <= maxAttempts) return;
    throw new Error2(
      ErrorCodes.CONTEXT_OVERFLOW,
      `Compaction failed to bring the context under the model window after ${String(maxAttempts)} attempts.`,
      { cause: error instanceof Error ? error : undefined },
    );
  }

  private retryFailedDriver(context: LoopErrorContext): boolean {
    const driver = context.failedDriver;
    if (driver === undefined || context.currentStep?.signal.aborted === true) return false;
    context.retry(driver, { at: 'head' });
    return true;
  }

  private async beforeStep(signal: AbortSignal, turnId?: number): Promise<void> {
    await this.retryPostCommitMaintenance(true);
    this.checkAutoCompaction(true, turnId);
    if (this.strategy.shouldBlock(this.tokenCountWithPending())) {
      await this.block(signal, turnId);
    }
    await this.retryPostCommitMaintenance(true);
  }

  private async afterStep(turnId: number): Promise<void> {
    this.consecutiveOverflowCompactions = 0;
    if (this.strategy.checkAfterStep) {
      this.checkAutoCompaction(false, turnId);
    }
  }

  private checkAutoCompaction(throwOnLimit = true, turnId?: number): boolean {
    if (this._compacting) return true;
    if (
      this.lastCompactedTokenCount !== null &&
      this.tokenCountWithPending() <= this.lastCompactedTokenCount
    ) {
      return false;
    }
    if (!this.strategy.shouldCompact(this.tokenCountWithPending())) return false;
    return this.beginAutoCompaction(throwOnLimit, turnId);
  }

  private beginAutoCompaction(throwOnLimit = true, turnId?: number): boolean {
    if (this._compacting) return true;
    const maxCompactions = this.strategy.maxCompactionPerTurn;
    if (this.compactionCountInTurn >= maxCompactions) {
      if (throwOnLimit) {
        throw new Error2(ErrorCodes.CONTEXT_OVERFLOW, `Compaction limit exceeded (${String(maxCompactions)})`, {
          details: { maxCompactions },
        });
      }
      return false;
    }
    return this.begin({ source: 'auto' }, turnId);
  }

  private async block(signal?: AbortSignal, turnId?: number): Promise<void> {
    const active = this._compacting;
    if (active === null) return;
    active.blockedByTurn = true;
    active.turnId ??= turnId;
    this.propagateBlockingAbort(active, signal);
    this.eventBus.publish({ type: 'compaction.blocked', turnId });
    try {
      await active.promise;
    } catch (error) {
      if (this.wasBlockingWaitAborted(active, signal, error)) return;
      throw error;
    }
  }

  private propagateBlockingAbort(active: ActiveCompaction, signal: AbortSignal | undefined): void {
    signal?.addEventListener(
      'abort',
      () => {
        if (this._compacting === active) active.abortController.abort();
      },
      { once: true },
    );
  }

  private wasBlockingWaitAborted(
    active: ActiveCompaction,
    signal: AbortSignal | undefined,
    error: unknown,
  ): boolean {
    return (
      signal?.aborted === true &&
      (active.abortController.signal.aborted || isAbortError(error))
    );
  }

  private async compactionWorker(
    active: ActiveCompaction,
    data: Readonly<CompactionBeginData>,
  ): Promise<CompactionResult> {
    try {
      const result = await this.compactionRound(active, data);
      if (this._compacting !== active) throw compactionCancelledReason(active);
      return await this.finishCommittedCompaction(active, result);
    } catch (error) {
      if (
        active.phase === 'committing' &&
        this.wire.getModel(CompactionModel).phase === 'committed'
      ) {
        active.phase = 'committed';
      }
      if (active.phase === 'committed' && active.committedResult !== undefined) {
        this.reportDiagnosticError('post-commit compaction work failed', error);
        return await this.finishCommittedCompaction(active, active.committedResult);
      }
      if (active.abortController.signal.aborted) {
        this.cancelActive(active);
        throw error;
      }
      const blockedByTurn = this._compacting === active && active.blockedByTurn;
      this.failActive(active);
      if (blockedByTurn) {
        throw error;
      }
      this.eventBus.publish({
        type: 'error',
        ...toKimiErrorPayload(error),
      });
      throw error;
    } finally {
      this._onDidFinishCompaction.fire(active);
    }
  }

  private async finishCommittedCompaction(
    active: ActiveCompaction,
    result: CompactionResult,
  ): Promise<CompactionResult> {
    this.lastCompactedTokenCount = result.tokensAfter;
    this.schedulePostCommitMaintenance(active.turnId);
    await this.retryPostCommitMaintenance(false);
    this.releaseCommitted(active);
    const { contextSummary: _contextSummary, ...eventResult } = result;
    void _contextSummary;
    this.eventBus.publish({ type: 'compaction.completed', result: eventResult });
    return result;
  }

  private async compactionRound(
    active: ActiveCompaction,
    data: Readonly<CompactionBeginData>,
  ): Promise<CompactionResult> {
    const startedAt = Date.now();
    const originalHistory = [...this.context.get()];
    const tokensBefore = estimateTokensForMessages(originalHistory);
    let retryCount = 0;
    let thinkingEffort = this.profile.data().thinkingLevel;

    try {
      const signal = active.abortController.signal;
      signal.throwIfAborted();

      await this.hooks.onWillCompact.run(active);

      const resolvedModel = this.profile.resolveModelContext();
      thinkingEffort = resolvedModel.thinkingLevel;
      const maxContextTokens = resolvedModel.modelCapabilities.max_context_tokens;
      const defaultCompactionCap =
        maxContextTokens > 0
          ? Math.min(maxContextTokens, DEFAULT_COMPACTION_MAX_COMPLETION_TOKENS)
          : undefined;
      const compactionMaxOutputSize = resolvedModel.maxOutputSize ?? defaultCompactionCap;

      const instruction = renderPrompt(compactionInstructionTemplate, {
        customInstruction: data.instruction?.trim() ?? '',
      }).trimEnd();

      const delays = retryBackoffDelays(MAX_COMPACTION_RETRY_ATTEMPTS);
      let attempt: CompactionAttemptResult | undefined;
      let historyForModel: readonly ContextMessage[] = stripDynamicToolContext(originalHistory);
      let droppedCount = 0;
      let overflowShrinkCount = 0;
      let emptyOrTruncatedShrinkCount = 0;
      while (true) {
        const messagesToCompact = historyForModel;
        const messages: Message[] = [...messagesToCompact, createUserMessage(instruction)];
        const estimatedCompactionRequestTokens = this.estimateRequestTokens(messages);

        try {
          attempt = collectSummary(
            await this.llmRequester.request(
              {
                messages,
                maxOutputSize: compactionMaxOutputSize,
                source: {
                  type: 'operation',
                  requestKind: 'full_compaction',
                  // Per-attempt count of messages dropped by overflow/empty
                  // shrinks so far; recorded on the llm.request wire op so a
                  // replay can see how much history each retry round blinded.
                  logFields: { droppedCount },
                },
              },
              undefined,
              signal,
            ),
          );
          break;
        } catch (error) {
          const isContextOverflow = this.shouldRecoverFromContextOverflow(
            error,
            estimatedCompactionRequestTokens,
          );
          if (isContextOverflow) {
            this.observeContextOverflow(estimatedCompactionRequestTokens);
            overflowShrinkCount += 1;
            if (
              overflowShrinkCount > MAX_COMPACTION_OVERFLOW_SHRINK_ATTEMPTS ||
              messagesToCompact.length <= 1
            ) {
              throw error;
            }
            const before = messagesToCompact.length;
            historyForModel = shrinkCompactionHistoryAfterOverflow(
              messagesToCompact,
              overflowShrinkCount,
            );
            droppedCount += before - historyForModel.length;
            retryCount = 0;
            continue;
          }
          if (
            (error instanceof CompactionTruncatedError || unwrapErrorCause(error) instanceof APIEmptyResponseError) &&
            messagesToCompact.length > 1
          ) {
            emptyOrTruncatedShrinkCount += 1;
            if (emptyOrTruncatedShrinkCount > MAX_COMPACTION_RETRY_ATTEMPTS) {
              throw error;
            }
            const reduced = dropOldestMessageAndLeadingToolResults(messagesToCompact);
            droppedCount += messagesToCompact.length - reduced.length;
            historyForModel = reduced;
            retryCount = 0;
            continue;
          }
          if (!isRetryableGenerateError(unwrapErrorCause(error))) {
            throw error;
          }
          if (retryCount + 1 >= MAX_COMPACTION_RETRY_ATTEMPTS) {
            throw error;
          }
          await sleepForRetry(delays[retryCount]!, signal);
          retryCount += 1;
        }
      }

      if (attempt === undefined) {
        throw new APIEmptyResponseError(
          'The compaction response did not contain a usable summary.',
        );
      }

      if (!historySafeToCompact(this.context.get(), originalHistory)) {
        const active = this._compacting;
        if (active !== null) {
          this.cancelActive(active);
        }
        throw compactionCancelledReason(active);
      }

      const summary = this.postProcessSummary(attempt.summary);
      const compactionInput = {
        summary,
        contextSummary: buildCompactionSummaryText(summary),
        compactedCount: originalHistory.length,
        tokensBefore,
        droppedCount: droppedCount === 0 ? undefined : droppedCount,
      };
      const prepared = buildContextCompactionShape(this.context.get(), compactionInput);
      const { messages: _messages, ...preparedResult } = prepared;
      void _messages;
      active.phase = 'committing';
      active.committedResult = preparedResult;
      const result = this.context.applyCompaction(compactionInput);
      active.phase = 'committed';
      active.committedResult = result;

      const properties: CompactionFinishedEvent = {
        source: data.source,
        tokens_before: result.tokensBefore,
        tokens_after: result.tokensAfter,
        duration_ms: Date.now() - startedAt,
        compacted_count: result.compactedCount,
        dropped_count: result.droppedCount,
        retry_count: retryCount,
        round: 1,
        thinking_effort: thinkingEffort,
        ...usageTelemetry(attempt.usage),
      };
      this.telemetry.track2('compaction_finished', properties);
      return result;
    } catch (error) {
      if (isAbortError(error) && active.abortController.signal.aborted) throw error;
      this.telemetry.track2('compaction_failed', {
        source: data.source,
        tokens_before: tokensBefore,
        duration_ms: Date.now() - startedAt,
        round: 1,
        retry_count: retryCount,
        thinking_effort: thinkingEffort,
        error_type: error instanceof Error ? error.name : 'Unknown',
      });
      if (
        isError2(error) &&
        (error.code === ErrorCodes.AUTH_LOGIN_REQUIRED ||
          error.code === ErrorCodes.PROVIDER_AUTH_ERROR)
      ) {
        throw error;
      }
      throw new Error2(ErrorCodes.COMPACTION_FAILED, String(error), { cause: error });
    }
  }

  private postProcessSummary(summary: string): string {
    const todos = this.currentTodos();
    if (todos.length === 0) {
      return summary;
    }
    return `${summary.trim()}\n\n${renderTodoList(todos, '## TODO List')}`;
  }

  private currentTodos(): readonly TodoItem[] {
    return this.todo.getTodos();
  }

  private tokenCountWithPending(): number {
    return this.contextSize.get().size;
  }

  private schedulePostCommitMaintenance(turnId?: number): void {
    this.systemPromptRefreshPending = true;
    this.systemPromptRefreshTurnId = turnId;
    this.contextReinjectionPending = true;
  }

  private async retryPostCommitMaintenance(throwOnFailure: boolean): Promise<void> {
    if (
      !this.systemPromptRefreshPending &&
      !this.contextReinjectionPending &&
      this.wire.getModel(CompactionModel).phase !== 'committed'
    ) {
      return;
    }
    let maintenance = this.postCommitMaintenance;
    if (maintenance === undefined) {
      maintenance = this.performPostCommitMaintenance();
      this.postCommitMaintenance = maintenance;
      void maintenance.finally(() => {
        if (this.postCommitMaintenance === maintenance) {
          this.postCommitMaintenance = undefined;
        }
      });
    }
    const error = await maintenance;
    if (throwOnFailure && error !== undefined) throw error;
  }

  private async performPostCommitMaintenance(): Promise<Error | undefined> {
    let firstError: Error | undefined;
    if (this.systemPromptRefreshPending) {
      try {
        const systemPrompt = await this.profile.refreshSystemPrompt();
        if (systemPrompt !== undefined && this.systemPromptRefreshTurnId !== undefined) {
          this.llmRequester.refreshTurnSystemPrompt(
            this.systemPromptRefreshTurnId,
            systemPrompt,
          );
        }
        this.systemPromptRefreshPending = false;
        this.systemPromptRefreshTurnId = undefined;
      } catch (error) {
        firstError = error instanceof Error ? error : new Error(String(error));
        this.reportDiagnosticError('failed to refresh system prompt after compaction', error);
      }
    }
    if (this.contextReinjectionPending) {
      try {
        await this.contextInjector.injectAfterCompaction();
        this.contextReinjectionPending = false;
      } catch (error) {
        firstError ??= error instanceof Error ? error : new Error(String(error));
        this.reportDiagnosticError('failed to inject context after compaction', error);
      }
    }
    this.lastCompactedTokenCount = this.tokenCountWithPending();
    if (
      !this.systemPromptRefreshPending &&
      !this.contextReinjectionPending &&
      this.wire.getModel(CompactionModel).phase === 'committed'
    ) {
      this.wire.dispatch(fullCompactionComplete({}));
    }
    return firstError;
  }

  private reportDiagnosticError(message: string, error: unknown): void {
    try {
      this.log.error(message, { error });
    } catch {
      // Post-commit diagnostics must not alter compaction convergence or make
      // an already-committed result appear to have failed.
    }
  }

  private get contextInjector(): IAgentContextInjectorService {
    if (this.contextInjectorService === undefined) {
      this.contextInjectorService = this.instantiation.invokeFunction((accessor) =>
        accessor.get(IAgentContextInjectorService),
      );
    }
    return this.contextInjectorService;
  }
}

function findAPIStatusError(error: unknown): APIStatusError | undefined {
  let current: unknown = error;
  const seen = new Set<unknown>();
  while (current !== undefined && current !== null && !seen.has(current)) {
    if (current instanceof APIStatusError) return current;
    seen.add(current);
    current = current instanceof Error ? current.cause : undefined;
  }
  return undefined;
}

function collectSummary(finish: LLMRequestFinish): CompactionAttemptResult {
  if (finish.providerFinishReason === 'truncated') {
    throw new CompactionTruncatedError();
  }

  const summary = finish.message.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('')
    .trim();
  if (summary.length === 0) {
    throw new APIEmptyResponseError(
      'The compaction response did not contain a non-empty summary.',
    );
  }

  return { summary, usage: finish.usage };
}

function historySafeToCompact(
  current: readonly ContextMessage[],
  original: readonly ContextMessage[],
): boolean {
  if (current.length < original.length) return false;
  if (!original.every((message, index) => message === current[index])) return false;
  return current.slice(original.length).every(isRealUserInput);
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

function usageTelemetry(usage: TokenUsage | null): CompactionTelemetryProperties {
  if (usage === null) return {};
  return {
    input_tokens: inputTotal(usage),
    output_tokens: usage.output,
    input_cache_read: usage.inputCacheRead,
    input_cache_creation: usage.inputCacheCreation,
  };
}

function compactionCancelledReason(active: ActiveCompaction | null): Error {
  const reason = active?.abortController.signal.reason;
  if (reason instanceof Error) return reason;
  const error = new Error('Compaction cancelled.');
  error.name = 'AbortError';
  return error;
}

registerScopedService(
  LifecycleScope.Agent,
  IAgentFullCompactionService,
  AgentFullCompactionService,
  InstantiationType.Eager,
  'fullCompaction',
);
