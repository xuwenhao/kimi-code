// Compaction scenario + probe tests.
//
// Two kinds of tests live here:
//   * GUARD tests lock in behavior we rely on (so future refactors can't
//     silently regress it).
//   * PROBE tests exercise the high-risk scenarios surfaced in review and in
//     our own audit, asserting the DESIRED behavior. Where the current
//     implementation does NOT meet that bar, the probe is marked `it.fails`:
//     the suite stays green, but the test documents the exact defect and will
//     start failing (forcing its removal) the day the behavior is fixed.
//
// Compaction is a hot path, so these intentionally drive the real
// Agent/ContextMemory/FullCompaction machinery through the test harness rather
// than mocking it.
import type { ContentPart, Message } from '@moonshot-ai/kosong';
import { createControlledPromise } from '@antfu/utils';
import { describe, expect, it, vi } from 'vitest';

import type { AgentOptions, AgentRecord, AgentRecordPersistence } from '../../../src/agent';
import type { BackgroundTaskInfo } from '../../../src/agent/background';
import { COMPACTION_ELISION_VARIANT, COMPACTION_SUMMARY_PREFIX } from '../../../src/agent/compaction';
import {
  AGENT_WIRE_PROTOCOL_VERSION,
  InMemoryAgentRecordPersistence,
  markAgentRecordAppendError,
} from '../../../src/agent/records';
import type { ContextMessage } from '../../../src/agent/context';
import { FLAG_DEFINITIONS, FlagResolver } from '../../../src/flags';
import type { ResolvedAgentProfile } from '../../../src/profile';
import { abortError } from '../../../src/utils/abort';
import { testAgent, type TestAgentContext } from '../harness/agent';

type GenerateFn = NonNullable<AgentOptions['generate']>;

const PROVIDER = { type: 'kimi', apiKey: 'test-key', model: 'kimi-code' } as const;
const CAPS = {
  image_in: true,
  video_in: true,
  audio_in: false,
  thinking: true,
  tool_use: true,
  max_context_tokens: 256_000,
} as const;

const RUNNING_BACKGROUND_TASK = {
  taskId: 'process-example',
  kind: 'process',
  description: 'important background work',
  status: 'running',
  detached: true,
  startedAt: 1,
  endedAt: null,
  command: 'example-command',
  pid: 42,
  exitCode: null,
} satisfies BackgroundTaskInfo;

function textResult(text: string): Awaited<ReturnType<GenerateFn>> {
  return {
    id: 'mock-compaction-summary',
    message: { role: 'assistant', content: [{ type: 'text', text }], toolCalls: [] },
    usage: { inputOther: 1, output: 1, inputCacheRead: 0, inputCacheCreation: 0 },
    finishReason: 'completed',
    rawFinishReason: 'stop',
  };
}

function historyTexts(ctx: TestAgentContext): string[] {
  return ctx.agent.context.history.map((message) =>
    message.content.map((part) => (part.type === 'text' ? part.text : `[${part.type}]`)).join(''),
  );
}

function summaryMessageText(ctx: TestAgentContext): string {
  const summary = ctx.agent.context.history.find(
    (message) => message.origin?.kind === 'compaction_summary',
  );
  return summary?.content.map((part) => (part.type === 'text' ? part.text : '')).join('') ?? '';
}

function recordsThrough(
  records: readonly AgentRecord[],
  predicate: (record: AgentRecord) => boolean,
): AgentRecord[] {
  const index = records.findIndex(predicate);
  if (index < 0) throw new Error('record boundary not found');
  return structuredClone(records.slice(0, index + 1));
}

describe('compaction — guard tests', () => {
  it('continues compaction when a synchronous started-event observer throws', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    ctx.emitter.once('compaction.started', () => {
      throw new Error('observer failed');
    });
    const completed = ctx.once('compaction.completed');

    await expect(ctx.rpc.beginCompaction({})).resolves.toBeUndefined();
    await completed;

    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
    expect(summaryMessageText(ctx)).toContain('Compacted summary.');
  });

  it('reserves compaction before its persistence callback can re-enter prompt', async () => {
    let ctx!: TestAgentContext;
    let reentrantAdmission: ReturnType<TestAgentContext['agent']['turn']['submitPrompt']> | undefined;
    const persistence = new InMemoryAgentRecordPersistence([], {
      onRecord: (record) => {
        if (record.type !== 'full_compaction.begin' || reentrantAdmission !== undefined) return;
        reentrantAdmission = ctx.agent.turn.submitPrompt([
          { type: 'text', text: 'PROMPT-FROM-COMPACTION-REENTRY' },
        ]);
      },
    });
    let generateCalls = 0;
    ctx = testAgent({
      persistence,
      generate: async () => {
        generateCalls += 1;
        return generateCalls === 1
          ? textResult('Compacted summary.')
          : textResult('Deferred prompt completed.');
      },
    });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    const ended = ctx.once('turn.ended');

    await ctx.rpc.beginCompaction({});
    expect(reentrantAdmission).toMatchObject({ kind: 'deferred' });
    await ended;

    expect(generateCalls).toBe(2);
    expect(historyTexts(ctx).join('\n')).toContain('PROMPT-FROM-COMPACTION-REENTRY');
  });

  it('keeps the successful compaction reservation through synchronous completed listeners', async () => {
    let generateCalls = 0;
    const ctx = testAgent({
      generate: async () => {
        generateCalls += 1;
        return generateCalls === 1
          ? textResult('Compacted summary.')
          : textResult('Deferred prompt completed.');
      },
    });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    let admission: ReturnType<typeof ctx.agent.turn.submitPrompt> | undefined;
    ctx.emitter.once('compaction.completed', () => {
      admission = ctx.agent.turn.submitPrompt([
        { type: 'text', text: 'PROMPT-FROM-COMPLETED-LISTENER' },
      ]);
    });
    const ended = ctx.once('turn.ended');

    await ctx.rpc.beginCompaction({});
    await ended;

    expect(admission).toMatchObject({ kind: 'deferred' });
    expect(generateCalls).toBe(2);
    expect(historyTexts(ctx).join('\n')).toContain('PROMPT-FROM-COMPLETED-LISTENER');
  });

  it('treats cancellation after context commit as a join on the completed outcome', async () => {
    const refreshStarted = createControlledPromise<void>();
    const releaseRefresh = createControlledPromise<void>();
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    vi.spyOn(ctx.agent, 'refreshSystemPrompt').mockImplementation(async () => {
      refreshStarted.resolve();
      await releaseRefresh;
      return 'refreshed system prompt';
    });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await refreshStarted;
    let cancelSettled = false;
    const cancel = ctx.rpc.cancelCompaction({}).then(() => {
      cancelSettled = true;
    });
    await Promise.resolve();
    expect(cancelSettled).toBe(false);
    expect(ctx.agent.fullCompaction.isCompacting).toBe(true);
    expect(ctx.allEvents.some((event) => event.event === 'compaction.cancelled')).toBe(false);

    releaseRefresh.resolve();
    await cancel;
    await completed;

    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.cancel'),
    ).toHaveLength(0);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
  });

  it('does not turn a completed-listener cancellation into a second terminal outcome', async () => {
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    let listenerCancellation: ReturnType<typeof ctx.rpc.cancelCompaction> | undefined;
    ctx.emitter.once('compaction.completed', () => {
      listenerCancellation = ctx.rpc.cancelCompaction({});
    });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;
    await listenerCancellation;

    expect(ctx.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.cancel'),
    ).toHaveLength(0);
  });

  it('converges to one completed terminal when system-prompt refresh rejects after commit', async () => {
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    vi.spyOn(ctx.agent, 'refreshSystemPrompt').mockRejectedValue(
      abortError('refresh failed independently of compaction cancellation'),
    );
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;

    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.cancel'),
    ).toHaveLength(0);
  });

  it('retries an async post-compaction injection failure without duplicating reminders', async () => {
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    vi.spyOn(ctx.agent.background, 'list').mockReturnValue([RUNNING_BACKGROUND_TASK]);
    const originalInjection = ctx.agent.injection.injectAfterCompaction.bind(ctx.agent.injection);
    let attempts = 0;
    vi.spyOn(ctx.agent.injection, 'injectAfterCompaction').mockImplementation(async () => {
      attempts += 1;
      await originalInjection();
      if (attempts === 1) throw new Error('injection failed after appending reminders');
    });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;

    expect(attempts).toBe(2);
    expect(
      ctx.agent.context.history.filter(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === 'background_task_status',
      ),
    ).toHaveLength(1);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
  });

  it('resets injector cursors when a reminder append throws before persistence', async () => {
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.agent.permission.setMode('auto');
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    const originalAppend = ctx.agent.context.appendSystemReminder.bind(ctx.agent.context);
    let permissionAppendAttempts = 0;
    vi.spyOn(ctx.agent.context, 'appendSystemReminder').mockImplementation((content, origin) => {
      if (origin.kind === 'injection' && origin.variant === 'permission_mode') {
        permissionAppendAttempts += 1;
        if (permissionAppendAttempts === 1) {
          throw new Error('append rejected before accepting the reminder');
        }
      }
      originalAppend(content, origin);
    });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;

    expect(permissionAppendAttempts).toBe(2);
    expect(
      ctx.agent.context.history.filter(
        (message) =>
          message.origin?.kind === 'injection' && message.origin.variant === 'permission_mode',
      ),
    ).toHaveLength(1);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
  });

  it('persists one completion when markCompleted throws before entering its implementation', async () => {
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    vi.spyOn(ctx.agent.fullCompaction, 'markCompleted').mockImplementationOnce(() => {
      throw new Error('completion callback failed before persistence');
    });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;

    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
  });

  it('retries completion after persistence explicitly rejects it before acceptance', async () => {
    const persistenceError = new Error('completion rejected before append');
    const accepted = new InMemoryAgentRecordPersistence();
    let completionAttempts = 0;
    const persistence: AgentRecordPersistence = {
      read: () => accepted.read(),
      append: (record) => {
        if (record.type === 'full_compaction.complete') {
          completionAttempts += 1;
          if (completionAttempts === 1) {
            throw markAgentRecordAppendError(persistenceError, false);
          }
        }
        accepted.append(record);
      },
      rewrite: (records) => {
        accepted.rewrite(records);
      },
      flush: () => accepted.flush(),
      close: () => accepted.close(),
    };
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;

    expect(completionAttempts).toBe(2);
    expect(
      accepted.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
  });

  it('does not duplicate completion when persistence throws after accepting the terminal record', async () => {
    const persistenceError = new Error('completion observer failed after append');
    let ctx!: TestAgentContext;
    let reentrantCancellation: Promise<void> | undefined;
    const persistence = new InMemoryAgentRecordPersistence([], {
      onRecord: (record) => {
        if (record.type !== 'full_compaction.complete') return;
        reentrantCancellation = ctx.agent.fullCompaction.cancel();
        throw persistenceError;
      },
    });
    ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;
    await reentrantCancellation;

    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
  });

  it('does not duplicate completion when an async markCompleted wrapper rejects after persistence', async () => {
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({ persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    const originalMarkCompleted = ctx.agent.fullCompaction.markCompleted.bind(
      ctx.agent.fullCompaction,
    );
    vi.spyOn(ctx.agent.fullCompaction, 'markCompleted').mockImplementationOnce(
      (() => {
        originalMarkCompleted();
        return Promise.reject(new Error('async completion observer failed after persistence'));
      }) as () => void,
    );
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await completed;

    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(1);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
  });

  it('keeps deferred work blocked until a transient post-commit injection failure recovers', async () => {
    const injectionStarted = createControlledPromise<void>();
    const releaseFailure = createControlledPromise<void>();
    let generateCalls = 0;
    const ctx = testAgent({
      generate: async () => {
        generateCalls += 1;
        return generateCalls === 1
          ? textResult('Compacted summary.')
          : textResult('Deferred prompt completed.');
      },
    });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    const originalInjection = ctx.agent.injection.injectAfterCompaction.bind(ctx.agent.injection);
    let injectionAttempts = 0;
    vi.spyOn(ctx.agent.injection, 'injectAfterCompaction').mockImplementation(async () => {
      injectionAttempts += 1;
      if (injectionAttempts === 1) {
        injectionStarted.resolve();
        await releaseFailure;
      }
      if (injectionAttempts <= 2) throw new Error('transient injection failure');
      await originalInjection();
    });
    const completed = ctx.once('compaction.completed');

    await ctx.rpc.beginCompaction({});
    await injectionStarted;
    const admission = await ctx.rpc.prompt({
      input: [{ type: 'text', text: 'PROMPT-WAITING-FOR-FINALIZATION' }],
    });
    expect(admission).toMatchObject({ kind: 'deferred' });
    expect(ctx.agent.turn.hasActiveTurn).toBe(false);
    expect(generateCalls).toBe(1);

    const ended = ctx.once('turn.ended');
    releaseFailure.resolve();
    await completed;
    await ended;

    // The two post-commit attempts are exhausted first. Completion may release
    // the TUI reservation, but the deferred turn's beforeStep must repair the
    // pending injection before its first model request is built.
    expect(injectionAttempts).toBe(3);
    expect(generateCalls).toBe(2);
    expect(historyTexts(ctx).join('\n')).toContain('PROMPT-WAITING-FOR-FINALIZATION');
    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
  });

  it('reports a provider AbortError as failure when the compaction signal was not aborted', async () => {
    const persistence = new InMemoryAgentRecordPersistence();
    const ctx = testAgent({
      persistence,
      generate: async () => {
        throw abortError('provider aborted its own request');
      },
    });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    const failed = ctx.once('error');

    await ctx.rpc.beginCompaction({});
    await failed;

    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
    expect(ctx.allEvents.filter((event) => event.event === 'compaction.cancelled')).toHaveLength(0);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.cancel'),
    ).toHaveLength(0);
    expect(
      persistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(0);
    expect(ctx.allEvents).toContainEqual(
      expect.objectContaining({
        type: '[rpc]',
        event: 'error',
        args: expect.objectContaining({ code: 'compaction.failed' }),
      }),
    );
  });

  it('patches the pending compaction across interleaved replay updates for crash and complete prefixes', async () => {
    const summaryRequested = createControlledPromise<void>();
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    const originalPersistence = new InMemoryAgentRecordPersistence();
    const original = testAgent({
      persistence: originalPersistence,
      generate: () => {
        summaryRequested.resolve();
        return summary;
      },
    });
    original.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    original.appendExchange(1, 'older user context', 'older assistant context', 40);

    await original.rpc.beginCompaction({});
    await summaryRequested;
    // These replay entries intentionally land between full_compaction.begin and
    // context.apply_compaction. The apply record must patch the open compaction,
    // not assume that compaction is still the replay array tail.
    original.agent.config.update({ systemPrompt: 'INTERLEAVED-SYSTEM-PROMPT' });
    original.agent.permission.setMode('auto');
    const originalCompleted = original.once('compaction.completed');
    summary.resolve(textResult('Compacted summary with interleaved updates.'));
    await originalCompleted;

    for (const terminalIncluded of [false, true]) {
      const prefix = recordsThrough(
        originalPersistence.records,
        (record) =>
          record.type ===
          (terminalIncluded ? 'full_compaction.complete' : 'context.apply_compaction'),
      );
      const resumedPersistence = new InMemoryAgentRecordPersistence(prefix);
      const resumed = testAgent({ persistence: resumedPersistence });

      await resumed.agent.resume();

      const replay = resumed.agent.replayBuilder.buildResult();
      const compactionIndex = replay.findIndex((record) => record.type === 'compaction');
      const compaction = replay[compactionIndex];
      expect(compaction).toMatchObject({
        type: 'compaction',
        result: expect.objectContaining({
          summary: 'Compacted summary with interleaved updates.',
        }),
      });
      expect(
        replay.slice(compactionIndex + 1).some((record) => record.type === 'config_updated'),
      ).toBe(true);
      expect(
        replay.slice(compactionIndex + 1).some((record) => record.type === 'permission_updated'),
      ).toBe(true);
      expect(
        resumedPersistence.records.filter(
          (record) => record.type === 'full_compaction.complete',
        ),
      ).toHaveLength(1);
      expect(
        resumed.allEvents.filter((event) => event.event === 'compaction.completed'),
      ).toHaveLength(0);
    }
  });

  it('recovers missing after-compaction reminders from a crash immediately after context commit', async () => {
    const originalPersistence = new InMemoryAgentRecordPersistence();
    const original = testAgent({ persistence: originalPersistence });
    original.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    original.appendExchange(1, 'older user context', 'older assistant context', 40);
    original.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    vi.spyOn(original.agent.background, 'list').mockReturnValue([RUNNING_BACKGROUND_TASK]);
    const originalCompleted = original.once('compaction.completed');
    await original.rpc.beginCompaction({});
    await originalCompleted;

    const prefix = recordsThrough(
      originalPersistence.records,
      (record) => record.type === 'context.apply_compaction',
    );
    const recoveryCompleted = createControlledPromise<void>();
    const resumedPersistence = new InMemoryAgentRecordPersistence(prefix, {
      onRecord: (record) => {
        if (record.type === 'full_compaction.complete') recoveryCompleted.resolve();
      },
    });
    const loadStarted = createControlledPromise<void>();
    const releaseLoad = createControlledPromise<void>();
    let backgroundLoaded = false;
    let firstModelRequestText = '';
    const resumed = testAgent({
      persistence: resumedPersistence,
      generate: async (_provider, _system, _tools, messages) => {
        firstModelRequestText = messages
          .flatMap((message) => message.content)
          .filter((part): part is Extract<ContentPart, { type: 'text' }> => part.type === 'text')
          .map((part) => part.text)
          .join('\n');
        return textResult('Recovered request completed.');
      },
      initialConfig: {
        providers: {
          'test-provider': { type: 'kimi', apiKey: 'test-key' },
        },
        models: {
          'kimi-code': {
            provider: 'test-provider',
            model: 'kimi-code',
            maxContextSize: CAPS.max_context_tokens,
          },
        },
      },
    });
    vi.spyOn(resumed.agent.background, 'loadFromDisk').mockImplementation(async () => {
      loadStarted.resolve();
      await releaseLoad;
      backgroundLoaded = true;
    });
    vi.spyOn(resumed.agent.background, 'list').mockImplementation(() =>
      backgroundLoaded ? [RUNNING_BACKGROUND_TASK] : [],
    );

    const resume = resumed.agent.resume();
    await loadStarted;
    await Promise.resolve();

    // The replay callback may reserve compaction, but recovery must not read
    // background state or emit the terminal until background restoration has
    // actually completed.
    expect(resumed.agent.fullCompaction.isCompacting).toBe(true);
    expect(
      resumedPersistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(0);

    releaseLoad.resolve();
    await recoveryCompleted;
    await resume;

    expect(resumed.agent.fullCompaction.isCompacting).toBe(false);
    expect(
      resumed.agent.context.history.filter(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === 'background_task_status',
      ),
    ).toHaveLength(1);
    expect(
      resumedPersistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(
      resumedPersistence.records.filter((record) => record.type === 'full_compaction.cancel'),
    ).toHaveLength(0);
    expect(resumed.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(0);

    const admission = await resumed.rpc.prompt({
      input: [{ type: 'text', text: 'FIRST-REQUEST-AFTER-RECOVERY' }],
    });
    expect(admission).toMatchObject({ kind: 'started' });
    const end = await resumed.agent.turn.waitForCurrentTurn();
    expect(end.event.reason).toBe('completed');
    expect(firstModelRequestText).toContain(RUNNING_BACKGROUND_TASK.description);
  });

  it('does not duplicate an after-compaction reminder that was durable before a crash', async () => {
    const originalPersistence = new InMemoryAgentRecordPersistence();
    const original = testAgent({ persistence: originalPersistence });
    original.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    original.appendExchange(1, 'older user context', 'older assistant context', 40);
    original.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    vi.spyOn(original.agent.background, 'list').mockReturnValue([RUNNING_BACKGROUND_TASK]);
    const originalCompleted = original.once('compaction.completed');
    await original.rpc.beginCompaction({});
    await originalCompleted;

    const prefix = recordsThrough(
      originalPersistence.records,
      (record) => record.type === 'full_compaction.complete',
    );
    const resumedPersistence = new InMemoryAgentRecordPersistence(prefix);
    const resumed = testAgent({ persistence: resumedPersistence });
    vi.spyOn(resumed.agent.background, 'list').mockReturnValue([RUNNING_BACKGROUND_TASK]);

    await resumed.agent.resume();
    await resumed.agent.fullCompaction.beforeStep(new AbortController().signal);

    expect(
      resumed.agent.context.history.filter(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === 'background_task_status',
      ),
    ).toHaveLength(1);
    expect(
      resumedPersistence.records.filter(
        (record) =>
          record.type === 'context.append_message' &&
          record.message.origin?.kind === 'injection' &&
          record.message.origin.variant === 'background_task_status',
      ),
    ).toHaveLength(1);
    expect(
      resumedPersistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(resumed.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(0);
  });

  it('repairs missing reminders after a crash following the completed terminal', async () => {
    const originalPersistence = new InMemoryAgentRecordPersistence();
    const original = testAgent({ persistence: originalPersistence });
    original.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    original.appendExchange(1, 'older user context', 'older assistant context', 40);
    original.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    vi.spyOn(original.agent.background, 'list').mockReturnValue([RUNNING_BACKGROUND_TASK]);
    vi.spyOn(original.agent.injection, 'injectAfterCompaction').mockRejectedValue(
      new Error('post-compaction injection remained unavailable'),
    );
    const originalCompleted = original.once('compaction.completed');

    await original.rpc.beginCompaction({});
    await originalCompleted;

    expect(
      original.agent.context.history.filter(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === 'background_task_status',
      ),
    ).toHaveLength(0);
    const prefix = recordsThrough(
      originalPersistence.records,
      (record) => record.type === 'full_compaction.complete',
    );

    let firstModelRequestText = '';
    const resumedPersistence = new InMemoryAgentRecordPersistence(prefix);
    const resumed = testAgent({
      persistence: resumedPersistence,
      generate: async (_provider, _system, _tools, messages) => {
        firstModelRequestText = messages
          .flatMap((message) => message.content)
          .filter((part): part is Extract<ContentPart, { type: 'text' }> => part.type === 'text')
          .map((part) => part.text)
          .join('\n');
        return textResult('Recovered request completed.');
      },
      initialConfig: {
        providers: {
          'test-provider': { type: 'kimi', apiKey: 'test-key' },
        },
        models: {
          'kimi-code': {
            provider: 'test-provider',
            model: 'kimi-code',
            maxContextSize: CAPS.max_context_tokens,
          },
        },
      },
    });
    vi.spyOn(resumed.agent.background, 'list').mockReturnValue([RUNNING_BACKGROUND_TASK]);

    await resumed.agent.resume();
    expect(resumed.agent.fullCompaction.isCompacting).toBe(false);
    expect(
      resumed.agent.context.history.filter(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === 'background_task_status',
      ),
    ).toHaveLength(0);

    await resumed.rpc.prompt({ input: [{ type: 'text', text: 'FIRST-REQUEST-AFTER-RESTART' }] });
    await resumed.agent.turn.waitForCurrentTurn();

    expect(firstModelRequestText).toContain(RUNNING_BACKGROUND_TASK.description);
    expect(
      resumed.agent.context.history.filter(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === 'background_task_status',
      ),
    ).toHaveLength(1);
    expect(
      resumedPersistence.records.filter((record) => record.type === 'full_compaction.complete'),
    ).toHaveLength(1);
    expect(resumed.allEvents.filter((event) => event.event === 'compaction.completed')).toHaveLength(0);
  });

  it('restores runtime profile handles before recovered compaction releases deferred work', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    const compactionApplied = createControlledPromise<void>();
    const originalPersistence = new InMemoryAgentRecordPersistence([], {
      onRecord: (record) => {
        if (record.type === 'context.apply_compaction') compactionApplied.resolve();
      },
    });
    const original = testAgent({ persistence: originalPersistence, generate: () => summary });
    original.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    original.agent.config.update({
      profileName: 'recovered-runtime-profile',
      systemPrompt: 'STALE-PERSISTED-SYSTEM-PROMPT',
    });
    original.appendExchange(1, 'older user context', 'older assistant context', 40);

    await original.rpc.beginCompaction({});
    await expect(
      original.rpc.prompt({
        input: [{ type: 'text', text: 'DEFERRED-BEHIND-RECOVERY' }],
      }),
    ).resolves.toMatchObject({ kind: 'deferred' });
    const originalCompleted = original.once('compaction.completed');
    const originalTurnEnded = original.once('turn.ended');
    summary.resolve(textResult('Compacted summary.'));
    await compactionApplied;
    const prefix = recordsThrough(
      originalPersistence.records,
      (record) => record.type === 'context.apply_compaction',
    );
    await originalCompleted;
    await originalTurnEnded;

    let runtimeProfileInstalled = false;
    let refreshedRuntimeContext: Parameters<ResolvedAgentProfile['systemPrompt']>[0] | undefined;
    let runtimeProfileInstalledAtGenerate = false;
    let generatedSystemPrompt = '';
    const resumed = testAgent({
      persistence: new InMemoryAgentRecordPersistence(prefix),
      generate: async (_provider, systemPrompt) => {
        runtimeProfileInstalledAtGenerate = runtimeProfileInstalled;
        generatedSystemPrompt = systemPrompt;
        return textResult('Recovered deferred request completed.');
      },
      initialConfig: {
        providers: {
          'test-provider': { type: 'kimi', apiKey: 'test-key' },
        },
        models: {
          'kimi-code': {
            provider: 'test-provider',
            model: 'kimi-code',
            maxContextSize: CAPS.max_context_tokens,
          },
        },
      },
    });
    const runtimeProfile: ResolvedAgentProfile = {
      name: 'recovered-runtime-profile',
      tools: [],
      systemPrompt: (context) => {
        refreshedRuntimeContext = context;
        return `REFRESHED-RUNTIME-SYSTEM-PROMPT\n${context.cwdListing ?? ''}`;
      },
    };
    const resumedTurnEnded = resumed.once('turn.ended');

    await resumed.agent.resume({
      beforePendingWorkResume: () => {
        runtimeProfileInstalled = true;
        resumed.agent.setActiveProfile(runtimeProfile);
      },
    });
    await resumedTurnEnded;

    expect(runtimeProfileInstalledAtGenerate).toBe(true);
    expect(refreshedRuntimeContext).toBeDefined();
    expect(generatedSystemPrompt).toContain('REFRESHED-RUNTIME-SYSTEM-PROMPT');
    expect(generatedSystemPrompt).not.toContain('STALE-PERSISTED-SYSTEM-PROMPT');
    expect(resumed.agent.config.systemPrompt).toContain('REFRESHED-RUNTIME-SYSTEM-PROMPT');
    expect(historyTexts(resumed).join('\n')).toContain('DEFERRED-BEHIND-RECOVERY');
  });

  it('repeated compaction folds the prior summary into the new one, never stacking two summaries', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'user one', 'assistant one', 40);

    ctx.mockNextResponse({ type: 'text', text: 'First summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    ctx.agent.context.appendUserMessage([{ type: 'text', text: 'user two' }]);
    ctx.mockNextResponse({ type: 'text', text: 'Second summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    const summaries = ctx.agent.context.history.filter(
      (message) => message.origin?.kind === 'compaction_summary',
    );
    // Exactly one summary survives; the first was re-summarized, not carried.
    expect(summaries).toHaveLength(1);
    expect(summaryMessageText(ctx)).toContain('Second summary.');
    expect(historyTexts(ctx).join('\n')).not.toContain('First summary.');
  });

  it('closes a dangling tool_use in the compaction summary request via synthesizeMissing', async () => {
    // Full compaction projects its summarizer input with { synthesizeMissing: true }
    // so an unresolved tool_use (whose result is sliced out / not yet recorded)
    // is answered by a synthetic tool_result — keeping the summary request
    // well-formed for strict providers instead of 400-ing on a dangling call.
    let summarizerMessages: Message[] | undefined;
    const capture: GenerateFn = async (_provider, _system, _tools, messages) => {
      summarizerMessages = messages;
      return textResult('Compacted summary.');
    };
    const ctx = testAgent({ generate: capture });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendUnresolvedToolExchange(0); // assistant with 2 tool calls, no results

    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    const msgs = summarizerMessages ?? [];
    const assistantIndex = msgs.findIndex(
      (message) => message.role === 'assistant' && message.toolCalls.length > 0,
    );
    expect(assistantIndex).toBeGreaterThanOrEqual(0);
    for (const toolCall of msgs[assistantIndex]!.toolCalls) {
      const answered = msgs
        .slice(assistantIndex + 1)
        .some((message) => message.role === 'tool' && message.toolCallId === toolCall.id);
      expect(answered).toBe(true);
    }
  });

  // Mutual exclusion: compaction and turn processing must not run concurrently,
  // or a turn mutating the context mid-summary loses output. Auto compaction is
  // structurally safe (it runs while the turn blocks at a step boundary); the
  // manual/SDK path is guarded explicitly here.
  it('rejects a manual compaction while a turn is active', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.agent.context.appendUserMessage([{ type: 'text', text: 'seed' }], { kind: 'user' });
    ctx.mockNextResponse({ type: 'text', text: 'turn done' });

    // launch() sets the active turn synchronously, so a turn is active before the
    // worker yields — exactly the window an SDK beginCompaction could land in.
    ctx.agent.turn.prompt([{ type: 'text', text: 'go' }]);
    expect(ctx.agent.turn.hasActiveTurn).toBe(true);

    await expect(ctx.rpc.beginCompaction({})).rejects.toThrow(/turn/i);

    await ctx.agent.turn.waitForCurrentTurn();
  });

  it('defers a prompt submitted during compaction and runs it afterward', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'user one', 'assistant one', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    ctx.mockNextResponse({ type: 'text', text: 'answer to the deferred prompt' });

    // begin() sets the compacting flag synchronously before the summarizer yields.
    void ctx.rpc.beginCompaction({});
    expect(ctx.agent.fullCompaction.isCompacting).toBe(true);

    // A prompt arriving mid-compaction is buffered (deferred), not rejected: null
    // means "not launched now", and it must run once compaction finishes.
    const turnId = ctx.agent.turn.prompt([{ type: 'text', text: 'DEFERRED-PROMPT' }]);
    expect(turnId).toBeNull();

    await ctx.once('compaction.completed');
    await ctx.agent.turn.waitForCurrentTurn();

    // Ran after compaction — neither lost nor stuck.
    expect(historyTexts(ctx).join('\n')).toContain('DEFERRED-PROMPT');
  });

  it('clears a deferred prompt when the user cancels before compaction finishes', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    let generateCalls = 0;
    const generate: GenerateFn = () => {
      generateCalls += 1;
      return summary;
    };
    const ctx = testAgent({ generate });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'user one', 'assistant one', 40);

    const compaction = ctx.rpc.beginCompaction({});
    expect(ctx.agent.fullCompaction.isCompacting).toBe(true);
    await ctx.rpc.prompt({ input: [{ type: 'text', text: 'CANCELLED-DEFERRED-PROMPT' }] });
    await ctx.rpc.steer({
      input: [{ type: 'text', text: 'CANCELLED-DEFERRED-STEER' }],
    });

    await ctx.rpc.cancel({});
    summary.resolve(textResult('Compacted summary.'));
    await compaction;

    expect(ctx.agent.turn.hasActiveTurn).toBe(false);
    expect(generateCalls).toBe(1);
    const texts = historyTexts(ctx).join('\n');
    expect(texts).not.toContain('CANCELLED-DEFERRED-PROMPT');
    expect(texts).not.toContain('CANCELLED-DEFERRED-STEER');
  });

  it('does not let a stale prompt-owned cancel kill its deferred replacement', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    let generateCalls = 0;
    const ctx = testAgent({
      generate: () => {
        generateCalls += 1;
        return generateCalls === 1
          ? summary
          : Promise.resolve(textResult('Replacement completed.'));
      },
    });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    await ctx.rpc.beginCompaction({});
    await ctx.rpc.prompt({
      input: [{ type: 'text', text: 'DEFERRED-A' }],
      promptId: 'prompt-deferred-a',
    });

    await ctx.rpc.cancel({
      expectedPromptId: 'prompt-deferred-a',
      requireActive: true,
    });
    await ctx.rpc.prompt({
      input: [{ type: 'text', text: 'DEFERRED-REPLACEMENT-X' }],
      promptId: 'prompt-replacement-x',
    });
    await expect(
      ctx.rpc.cancel({
        expectedPromptId: 'prompt-deferred-a',
        requireActive: true,
      }),
    ).rejects.toMatchObject({ code: 'turn.agent_busy' });

    const ended = ctx.once('turn.ended');
    summary.resolve(textResult('Compacted summary.'));
    await ended;

    expect(generateCalls).toBe(2);
    const texts = historyTexts(ctx).join('\n');
    expect(texts).not.toContain('DEFERRED-A');
    expect(texts).toContain('DEFERRED-REPLACEMENT-X');
  });

  it('keeps a deferred prompt when a stale turn-scoped cancel arrives during compaction', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    let generateCalls = 0;
    const generate: GenerateFn = () => {
      generateCalls += 1;
      if (generateCalls === 1) return Promise.resolve(textResult('Initial turn completed.'));
      if (generateCalls === 2) return summary;
      return Promise.resolve(textResult('Deferred turn completed.'));
    };
    const ctx = testAgent({ generate });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });

    await ctx.rpc.prompt({ input: [{ type: 'text', text: 'INITIAL-PROMPT' }] });
    await ctx.agent.turn.waitForCurrentTurn();
    const staleTurnId = ctx.agent.turn.currentId;
    expect(staleTurnId).toBe(0);
    ctx.appendExchange(2, 'older user context', 'older assistant context', 40);

    const compaction = ctx.rpc.beginCompaction({});
    expect(ctx.agent.fullCompaction.isCompacting).toBe(true);
    expect(
      ctx.agent.turn.submitPrompt([{ type: 'text', text: 'PRESERVED-DEFERRED-PROMPT' }]),
    ).toMatchObject({ kind: 'deferred', deferredPromptId: expect.any(String) });
    await ctx.agent.turn.cancel(staleTurnId);

    const deferredTurnEnded = ctx.once('turn.ended');
    summary.resolve(textResult('Compacted summary.'));
    await compaction;
    await deferredTurnEnded;

    expect(generateCalls).toBe(3);
    expect(historyTexts(ctx).join('\n')).toContain('PRESERVED-DEFERRED-PROMPT');
  });

  it('admits only one direct deferred prompt and echoes both correlations when it starts', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    const persistence = new InMemoryAgentRecordPersistence();
    let generateCalls = 0;
    const generate: GenerateFn = () => {
      generateCalls += 1;
      return generateCalls === 1
        ? summary
        : Promise.resolve(textResult('Deferred turn completed.'));
    };
    const ctx = testAgent({ generate, persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);

    await ctx.rpc.beginCompaction({});
    const accepted = await ctx.rpc.prompt({
      input: [{ type: 'text', text: 'FIRST-DEFERRED-PROMPT' }],
      promptId: 'prompt-first',
    });
    expect(accepted).toMatchObject({
      kind: 'deferred',
      deferredPromptId: expect.any(String),
    });
    if (accepted.kind !== 'deferred') throw new Error('expected deferred prompt admission');

    await expect(
      ctx.rpc.prompt({
        input: [{ type: 'text', text: 'SECOND-DEFERRED-PROMPT' }],
        promptId: 'prompt-second',
      }),
    ).rejects.toMatchObject({ code: 'turn.agent_busy' });

    const ended = ctx.once('turn.ended');
    summary.resolve(textResult('Compacted summary.'));
    await ended;

    expect(
      persistence.records
        .filter((record) => record.type === 'turn.prompt')
        .flatMap((record) => record.input)
        .filter((part) => part.type === 'text')
        .map((part) => part.text),
    ).toEqual(['FIRST-DEFERRED-PROMPT']);
    expect(ctx.allEvents).toContainEqual(
      expect.objectContaining({
        type: '[rpc]',
        event: 'turn.started',
        args: expect.objectContaining({
          promptId: 'prompt-first',
          deferredPromptId: accepted.deferredPromptId,
        }),
      }),
    );
    expect(historyTexts(ctx).join('\n')).not.toContain('SECOND-DEFERRED-PROMPT');
  });

  it('cancels only the deferred user prompt while preserving a background steer', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    let generateCalls = 0;
    const generate: GenerateFn = () => {
      generateCalls += 1;
      return generateCalls === 1
        ? summary
        : Promise.resolve(textResult('Background steer handled.'));
    };
    const ctx = testAgent({ generate });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);

    await ctx.rpc.beginCompaction({});
    await ctx.rpc.prompt({ input: [{ type: 'text', text: 'CANCEL-ME' }] });
    ctx.agent.turn.steer([{ type: 'text', text: 'KEEP-BACKGROUND-STEER' }], {
      kind: 'background_task',
      taskId: 'task-1',
      status: 'completed',
      notificationId: 'notification-1',
    });
    await ctx.rpc.cancel({});

    const ended = ctx.once('turn.ended');
    summary.resolve(textResult('Compacted summary.'));
    await ended;

    expect(generateCalls).toBe(2);
    expect(historyTexts(ctx).join('\n')).toContain('KEEP-BACKGROUND-STEER');
    expect(historyTexts(ctx).join('\n')).not.toContain('CANCEL-ME');
  });

  it('steers a compaction-deferred prompt by its stable prompt correlation', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    let generateCalls = 0;
    const ctx = testAgent({
      generate: () => {
        generateCalls += 1;
        return generateCalls === 1
          ? summary
          : Promise.resolve(textResult('Deferred prompt and steer handled.'));
      },
    });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);

    await ctx.rpc.beginCompaction({});
    await expect(
      ctx.rpc.prompt({
        input: [{ type: 'text', text: 'DEFERRED-OWNER' }],
        promptId: 'prompt-deferred-owner',
      }),
    ).resolves.toMatchObject({ kind: 'deferred' });
    await expect(
      ctx.rpc.steer({
        input: [{ type: 'text', text: 'STEER-FOR-DEFERRED-OWNER' }],
        expectedPromptId: 'prompt-deferred-owner',
        requireActive: true,
      }),
    ).resolves.toBeUndefined();

    const ended = ctx.once('turn.ended');
    summary.resolve(textResult('Compacted summary.'));
    await ended;

    const texts = historyTexts(ctx).join('\n');
    expect(texts).toContain('DEFERRED-OWNER');
    expect(texts).toContain('STEER-FOR-DEFERRED-OWNER');
  });

  it('replays a still-pending steer with its deferred prompt after a crash', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    const persistence = new InMemoryAgentRecordPersistence();
    const original = testAgent({ persistence, generate: () => summary });
    original.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    original.appendExchange(1, 'older user context', 'older assistant context', 40);

    await original.rpc.beginCompaction({});
    await original.rpc.prompt({
      input: [{ type: 'text', text: 'CRASH-DEFERRED-OWNER' }],
      promptId: 'prompt-crash-owner',
    });
    await original.rpc.steer({
      input: [{ type: 'text', text: 'CRASH-PENDING-STEER' }],
      expectedPromptId: 'prompt-crash-owner',
      requireActive: true,
    });

    const resumed = testAgent({
      persistence: new InMemoryAgentRecordPersistence(structuredClone(persistence.records)),
      generate: async () => textResult('Recovered deferred work.'),
      initialConfig: {
        providers: {
          'test-provider': { type: 'kimi', apiKey: 'test-key' },
        },
        models: {
          'kimi-code': {
            provider: 'test-provider',
            model: 'kimi-code',
            maxContextSize: CAPS.max_context_tokens,
          },
        },
      },
    });
    await resumed.agent.resume();
    await resumed.agent.turn.waitForCurrentTurn();

    const texts = historyTexts(resumed).join('\n');
    expect(texts).toContain('CRASH-DEFERRED-OWNER');
    expect(texts).toContain('CRASH-PENDING-STEER');
  });

  it('does not replay an activated empty deferred prompt after restart', async () => {
    const summary = createControlledPromise<Awaited<ReturnType<GenerateFn>>>();
    const persistence = new InMemoryAgentRecordPersistence();
    let originalGenerateCalls = 0;
    const original = testAgent({
      persistence,
      generate: () => {
        originalGenerateCalls += 1;
        return originalGenerateCalls === 1
          ? summary
          : Promise.resolve(textResult('Empty retry completed.'));
      },
    });
    original.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    original.appendExchange(1, 'older user context', 'older assistant context', 40);

    await original.rpc.beginCompaction({});
    await original.rpc.prompt({ input: [], promptId: 'empty-retry' });
    const ended = original.once('turn.ended');
    summary.resolve(textResult('Compacted summary.'));
    await ended;

    expect(
      persistence.records.some((record) => record.type === 'turn.deferred_prompt_started'),
    ).toBe(true);
    let replayGenerateCalls = 0;
    const resumed = testAgent({
      persistence: new InMemoryAgentRecordPersistence(structuredClone(persistence.records)),
      generate: async () => {
        replayGenerateCalls += 1;
        return textResult('Unexpected duplicate execution.');
      },
    });
    await resumed.agent.resume();
    await Promise.resolve();

    expect(replayGenerateCalls).toBe(0);
    expect(resumed.agent.turn.hasActiveTurn).toBe(false);
  });

  it('keeps cancelCompaction pending until the aborted worker has settled', async () => {
    const summarizerStarted = createControlledPromise<void>();
    const abortObserved = createControlledPromise<void>();
    const releaseCleanup = createControlledPromise<void>();
    let generateCalls = 0;
    const generate: GenerateFn = async (
      _provider,
      _systemPrompt,
      _tools,
      _history,
      _callbacks,
      options,
    ) => {
      generateCalls += 1;
      if (generateCalls !== 1) return textResult('Follow-up completed.');
      summarizerStarted.resolve();
      await new Promise<void>((resolve) => {
        const onAbort = (): void => {
          abortObserved.resolve();
          resolve();
        };
        if (options?.signal?.aborted === true) onAbort();
        else options?.signal?.addEventListener('abort', onAbort, { once: true });
      });
      await releaseCleanup;
      throw abortError();
    };
    const ctx = testAgent({ generate });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);

    await ctx.rpc.beginCompaction({});
    await summarizerStarted;
    let cancelSettled = false;
    const cancel = ctx.rpc.cancelCompaction({}).then(() => {
      cancelSettled = true;
    });
    await abortObserved;
    await Promise.resolve();
    expect(cancelSettled).toBe(false);

    releaseCleanup.resolve();
    await cancel;
    expect(
      await ctx.rpc.prompt({ input: [{ type: 'text', text: 'FOLLOW-UP-AFTER-CANCEL' }] }),
    ).toMatchObject({ kind: 'started' });
    await ctx.agent.turn.waitForCurrentTurn();
  });

  it('aborts compaction before reporting a cancellation-record persistence failure', async () => {
    const persistenceError = new Error('compaction cancel persistence failed');
    const persistence = new InMemoryAgentRecordPersistence([], {
      onRecord: (record) => {
        if (record.type === 'full_compaction.cancel') throw persistenceError;
      },
    });
    const summarizerStarted = createControlledPromise<void>();
    const generate: GenerateFn = async (
      _provider,
      _systemPrompt,
      _tools,
      _history,
      _callbacks,
      options,
    ) => {
      summarizerStarted.resolve();
      await new Promise<void>((_resolve, reject) => {
        const onAbort = (): void => {
          reject(options?.signal?.reason ?? abortError());
        };
        if (options?.signal?.aborted === true) onAbort();
        else options?.signal?.addEventListener('abort', onAbort, { once: true });
      });
      return textResult('Unexpected summary.');
    };
    const ctx = testAgent({ generate, persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);

    await ctx.rpc.beginCompaction({});
    await summarizerStarted;
    await expect(ctx.rpc.cancelCompaction({})).rejects.toBe(persistenceError);

    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
    expect(historyTexts(ctx).join('\n')).not.toContain('Unexpected summary.');
  });

  it('deduplicates compaction cancellation re-entered by its persistence observer', async () => {
    const summarizerStarted = createControlledPromise<void>();
    let ctx!: TestAgentContext;
    let nestedCancellation: Promise<void> | undefined;
    let cancelRecords = 0;
    const persistence = new InMemoryAgentRecordPersistence([], {
      onRecord: (record) => {
        if (record.type !== 'full_compaction.cancel') return;
        cancelRecords += 1;
        nestedCancellation ??= ctx.agent.fullCompaction.cancel();
      },
    });
    const generate: GenerateFn = async (
      _provider,
      _systemPrompt,
      _tools,
      _history,
      _callbacks,
      options,
    ) => {
      summarizerStarted.resolve();
      await new Promise<void>((_resolve, reject) => {
        const onAbort = (): void => {
          reject(options?.signal?.reason ?? abortError());
        };
        if (options?.signal?.aborted === true) onAbort();
        else options?.signal?.addEventListener('abort', onAbort, { once: true });
      });
      return textResult('Unexpected summary.');
    };
    ctx = testAgent({ generate, persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);
    await ctx.rpc.beginCompaction({});
    await summarizerStarted;

    const cancellation = ctx.agent.fullCompaction.cancel();
    expect(nestedCancellation).toBe(cancellation);
    await cancellation;

    expect(cancelRecords).toBe(1);
    expect(ctx.agent.fullCompaction.isCompacting).toBe(false);
  });

  it('keeps a concurrent prompt deferred until cancelled compaction cleanup finishes', async () => {
    const summarizerStarted = createControlledPromise<void>();
    const abortObserved = createControlledPromise<void>();
    const releaseLateSummary = createControlledPromise<void>();
    let generateCalls = 0;
    const generate: GenerateFn = async (
      _provider,
      _systemPrompt,
      _tools,
      _history,
      _callbacks,
      options,
    ) => {
      generateCalls += 1;
      if (generateCalls !== 1) return textResult('Deferred prompt completed.');
      summarizerStarted.resolve();
      const onAbort = (): void => {
        abortObserved.resolve();
      };
      if (options?.signal?.aborted === true) onAbort();
      else options?.signal?.addEventListener('abort', onAbort, { once: true });
      await releaseLateSummary;
      // Deliberately ignore the abort and return a stale summary. The runtime
      // must discard it before applyCompaction.
      return textResult('LATE-CANCELLED-SUMMARY');
    };
    const ctx = testAgent({ generate });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'ORIGINAL-USER', 'ORIGINAL-ASSISTANT', 40);

    await ctx.rpc.beginCompaction({});
    await summarizerStarted;
    const cancel = ctx.rpc.cancelCompaction({});
    await abortObserved;
    const admitted = await ctx.rpc.prompt({
      input: [{ type: 'text', text: 'PROMPT-DURING-CANCEL-CLEANUP' }],
    });
    expect(admitted).toMatchObject({ kind: 'deferred' });
    expect(ctx.agent.turn.hasActiveTurn).toBe(false);

    const ended = ctx.once('turn.ended');
    releaseLateSummary.resolve();
    await cancel;
    await ended;

    const texts = historyTexts(ctx).join('\n');
    expect(texts).toContain('ORIGINAL-ASSISTANT');
    expect(texts).toContain('PROMPT-DURING-CANCEL-CLEANUP');
    expect(texts).not.toContain('LATE-CANCELLED-SUMMARY');
  });

  it('drops deferred work and awaits compaction settlement during shutdown', async () => {
    const abortObserved = createControlledPromise<void>();
    const releaseCleanup = createControlledPromise<void>();
    const persistence = new InMemoryAgentRecordPersistence();
    let generateCalls = 0;
    const generate: GenerateFn = async (
      _provider,
      _systemPrompt,
      _tools,
      _history,
      _callbacks,
      options,
    ) => {
      generateCalls += 1;
      await new Promise<void>((resolve) => {
        const onAbort = (): void => {
          abortObserved.resolve();
          resolve();
        };
        if (options?.signal?.aborted === true) onAbort();
        else options?.signal?.addEventListener('abort', onAbort, { once: true });
      });
      await releaseCleanup;
      throw abortError();
    };
    const ctx = testAgent({ generate, persistence });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'older user context', 'older assistant context', 40);

    await ctx.rpc.beginCompaction({});
    await ctx.rpc.prompt({
      input: [{ type: 'text', text: 'MUST-NOT-LAUNCH-AFTER-CLOSE' }],
      promptId: 'prompt-closed-before-start',
    });
    let shutdownSettled = false;
    const shutdown = ctx.agent.turn.shutdown(abortError('Session closed')).then(() => {
      shutdownSettled = true;
    });
    await abortObserved;
    await Promise.resolve();
    expect(shutdownSettled).toBe(false);

    releaseCleanup.resolve();
    await shutdown;
    expect(generateCalls).toBe(1);
    expect(ctx.agent.turn.hasActiveTurn).toBe(false);
    expect(historyTexts(ctx).join('\n')).not.toContain('MUST-NOT-LAUNCH-AFTER-CLOSE');
    expect(
      persistence.records.find((record) => record.type === 'turn.cancel'),
    ).toMatchObject({ promptId: 'prompt-closed-before-start' });

    const resumed = testAgent({
      persistence: new InMemoryAgentRecordPersistence(structuredClone(persistence.records)),
      generate: async () => textResult('Unexpected resurrected prompt.'),
    });
    await resumed.agent.resume();
    expect(resumed.agent.turn.hasActiveTurn).toBe(false);
    expect(historyTexts(resumed).join('\n')).not.toContain('MUST-NOT-LAUNCH-AFTER-CLOSE');
  });

  it('defers a steer arriving during compaction and delivers it afterward', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'user one', 'assistant one', 40);
    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    ctx.mockNextResponse({ type: 'text', text: 'handled the steer' });

    void ctx.rpc.beginCompaction({});
    expect(ctx.agent.fullCompaction.isCompacting).toBe(true);

    // A background-task/cron steer mid-compaction must be buffered (null = buffered,
    // which is exactly what those fire-and-forget callers assume), not dropped.
    const turnId = ctx.agent.turn.steer([{ type: 'text', text: 'DEFERRED-STEER' }], {
      kind: 'background_task',
      taskId: 't',
      status: 'completed',
      notificationId: 'n',
    });
    expect(turnId).toBeNull();

    await ctx.once('compaction.completed');
    await ctx.agent.turn.waitForCurrentTurn();

    expect(historyTexts(ctx).join('\n')).toContain('DEFERRED-STEER');
  });
});

describe('compaction — probe tests (high-risk scenarios)', () => {
  // PROBE #1 / CMP-02 — messages appended while the summarizer request is in
  // flight (a live step racing a manual/SDK compaction). The summary only covers
  // the pre-compaction snapshot, and the all-user rebuild would drop the appended
  // assistant/tool tail — so compaction detects the changed history and cancels,
  // leaving the appended turn intact for a later clean-boundary compaction.
  it('preserves an assistant turn appended while the summarizer call is in flight', async () => {
    let ctx!: TestAgentContext;
    const appendDuringGenerate: GenerateFn = async () => {
      // Simulate the turn loop completing a step while compaction awaits.
      ctx.agent.context.appendLoopEvent({
        type: 'step.begin',
        uuid: 'race-step',
        turnId: '',
        step: 9,
      });
      ctx.agent.context.appendLoopEvent({
        type: 'content.part',
        uuid: 'race-part',
        turnId: '',
        step: 9,
        stepUuid: 'race-step',
        part: { type: 'text', text: 'RACE-ASSISTANT-OUTPUT' },
      });
      ctx.agent.context.appendLoopEvent({
        type: 'step.end',
        uuid: 'race-step',
        turnId: '',
        step: 9,
        finishReason: 'end_turn',
      });
      return textResult('Compacted summary.');
    };
    ctx = testAgent({ generate: appendDuringGenerate });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'user one', 'assistant one', 40);

    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.cancelled');

    expect(historyTexts(ctx).join('\n')).toContain('RACE-ASSISTANT-OUTPUT');
  });

  // PROBE #1b — a user-ROLE message that compaction would drop (background-task
  // notification, hook/cron reminder, shell output) appended mid-summary. It is
  // neither summarized (added after the snapshot) nor kept (applyCompaction keeps
  // only real user input), so it would silently vanish; the race guard must cancel
  // on any tail compaction would drop, not just non-user roles.
  it('cancels compaction when a droppable user-role tail is appended mid-summary', async () => {
    let ctx!: TestAgentContext;
    const appendDuringGenerate: GenerateFn = async () => {
      ctx.agent.context.appendUserMessage([{ type: 'text', text: 'BG-NOTIFY-OUTPUT' }], {
        kind: 'background_task',
        taskId: 't',
        status: 'completed',
        notificationId: 'n',
      });
      return textResult('Compacted summary.');
    };
    ctx = testAgent({ generate: appendDuringGenerate });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'user one', 'assistant one', 40);

    await ctx.rpc.beginCompaction({});
    await Promise.race([ctx.once('compaction.completed'), ctx.once('compaction.cancelled')]);

    // Cancelled, so the notification survives in history rather than being dropped.
    expect(historyTexts(ctx).join('\n')).toContain('BG-NOTIFY-OUTPUT');
  });

  // PROBE #2 — empty/truncated summarizer responses drop one oldest message and
  // retry. A dedicated shrink counter, bounded by MAX_COMPACTION_RETRY_ATTEMPTS,
  // keeps a model that always returns empty from issuing ~one call per message.
  it('bounds summarizer calls by the retry limit when the model keeps returning empty', async () => {
    let calls = 0;
    // Empty 7 times, then a valid summary. The bounded shrink counter gives up by
    // ~call 6, so compaction errors out before ever reaching the 8th (valid)
    // response; an unbounded impl would tolerate all 7 and complete on the 8th.
    const flakyEmpty: GenerateFn = async () => {
      calls += 1;
      return calls <= 7 ? textResult('') : textResult('Compacted summary.');
    };
    const ctx = testAgent({ generate: flakyEmpty });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    for (let i = 1; i <= 5; i++) {
      ctx.appendExchange(i, `user ${String(i)}`, `assistant ${String(i)}`, 40);
    }

    await ctx.rpc.beginCompaction({});
    await Promise.race([ctx.once('compaction.completed'), ctx.once('error')]);

    // A retry budget of MAX_COMPACTION_RETRY_ATTEMPTS(5) should bound calls.
    expect(calls).toBeLessThanOrEqual(6);
  });

  // PROBE #3 / CMP-08 — the kept-user budget is a fixed 20k and ignores the
  // model window, so on a small-window model the post-compaction context can
  // still exceed the trigger, re-compacting every turn without converging.
  it.fails('keeps the post-compaction context below the auto-compaction trigger on a small window', async () => {
    const SMALL_WINDOW = 16_000;
    const ctx = testAgent();
    ctx.configure({
      provider: PROVIDER,
      modelCapabilities: { ...CAPS, max_context_tokens: SMALL_WINDOW },
    });
    // ~7.5k tokens of user text per message (30k ascii chars / 4).
    for (let i = 1; i <= 3; i++) {
      ctx.appendExchange(i, 'u'.repeat(30_000), `assistant ${String(i)}`, 40);
    }

    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    // tokenCount after compaction should leave headroom below the 85% trigger,
    // otherwise the next turn immediately re-compacts and never converges.
    expect(ctx.agent.context.tokenCount).toBeLessThan(SMALL_WINDOW * 0.85);
  });

  // PROBE #4 / CMP-01 — compaction started while a tool exchange is still open
  // (SDK/REST caller mid-tool) clears pendingToolResultIds, so the tool.result
  // that arrives afterwards is treated as an orphan and silently dropped.
  it.fails('does not drop a tool result that arrives after a compaction started mid-exchange', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendUnresolvedToolExchange(0); // assistant with 2 tool calls, no results yet

    ctx.mockNextResponse({ type: 'text', text: 'Compacted summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    // The tool finishes after compaction; its result must not vanish.
    ctx.agent.context.appendLoopEvent({
      type: 'tool.result',
      parentUuid: 'call_unresolved_one',
      toolCallId: 'call_unresolved_one',
      result: { output: 'LATE-TOOL-RESULT' },
    });

    expect(historyTexts(ctx).join('\n')).toContain('LATE-TOOL-RESULT');
  });

  // CMP-12 fix — restoring a legacy `context.apply_compaction` record (pre-rework:
  // no keptUserMessageCount; the old `[summary, ...history.slice(compactedCount)]`
  // semantics kept a verbatim recent tail). On restore we reproduce that shape so
  // an upgraded session does not lose its recent assistant/tool tail.
  it('preserves the verbatim tail when restoring a legacy compaction record', () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.appendExchange(1, 'summarized user', 'TAIL-ASSISTANT', 40);

    // Goes through the real restore path so `records.restoring` gates the legacy
    // reconstruction. No keptUserMessageCount + compactedCount < length marks the
    // pre-rework record that kept history.slice(compactedCount) as a tail.
    ctx.agent.records.restore({
      type: 'context.apply_compaction',
      summary: 'Legacy summary.',
      compactedCount: 1,
      tokensBefore: 100,
      tokensAfter: 50,
    });

    expect(historyTexts(ctx).join('\n')).toContain('TAIL-ASSISTANT');
  });

  // PROBE #6 — when the summarizer request overflows, historyForModel is shrunk
  // to a recent suffix but still projected through MicroCompaction.compact()
  // with the cutoff computed for the FULL history. The absolute cutoff applied
  // to the shifted suffix can clear recent tool results the summary needs.
  // SKIPPED: micro-compaction has been disabled and its flag removed, so this
  // defect no longer exists.
  it.skip('does not clear recent tool results when projecting a shrunk suffix under an active micro-compaction cutoff', () => {
    // This defect only exists when micro-compaction is active, so enable the
    // flag explicitly rather than inheriting the ambient KIMI_CODE_EXPERIMENTAL
    // master switch — otherwise the probe's pass/fail flips with the runner's
    // environment (on locally with the master switch, off in CI by default).
    const ctx = testAgent({
      experimentalFlags: new FlagResolver(
        { KIMI_CODE_EXPERIMENTAL_MICRO_COMPACTION: '1' },
        FLAG_DEFINITIONS,
      ),
    });
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });

    const bigToolOutput = 'TOOL-OUTPUT-CONTENT '.repeat(60); // > minContentTokens(100)
    const full: ContextMessage[] = [];
    for (let i = 0; i < 20; i++) {
      if (i === 15) {
        full.push({
          role: 'tool',
          content: [{ type: 'text', text: bigToolOutput } satisfies ContentPart],
          toolCalls: [],
          toolCallId: `tool-${String(i)}`,
        });
      } else {
        full.push({
          role: i % 2 === 0 ? 'user' : 'assistant',
          content: [{ type: 'text', text: `m${String(i)}` }],
          toolCalls: [],
          origin: i % 2 === 0 ? { kind: 'user' } : undefined,
        });
      }
    }

    // Cutoff computed for the full history: keep the recent 10 (indices >= 10).
    ctx.agent.microCompaction.apply(10);

    // In the full history the tool result is at index 15 (>= cutoff) -> kept.
    const projectedFull = ctx.agent.context.project(full);
    const fullToolText = projectedFull
      .map((m) => m.content.map((p) => (p.type === 'text' ? p.text : '')).join(''))
      .join('\n');
    expect(fullToolText).toContain('TOOL-OUTPUT-CONTENT');

    // After an overflow shrink drops the oldest 10, the SAME tool result sits at
    // suffix index 5; the unchanged cutoff(10) now covers it. It must still be
    // preserved (it is a recent result the summary depends on).
    const shrunkSuffix = full.slice(10);
    const projectedSuffix = ctx.agent.context.project(shrunkSuffix);
    const suffixToolText = projectedSuffix
      .map((m) => m.content.map((p) => (p.type === 'text' ? p.text : '')).join(''))
      .join('\n');
    expect(suffixToolText).toContain('TOOL-OUTPUT-CONTENT');
  });

  // PROBE #7 / CMP-07 — when the oldest kept user message overflows the budget it
  // is truncated to text only, dropping any image/audio/video it carried: media
  // can't be partially truncated, and keeping it whole would overshoot the
  // budget. Recent messages that fit keep their media; only this boundary message
  // loses its attachments. Documented as an accepted limitation rather than fixed.
  it.fails('keeps media on the oldest kept user message instead of dropping it on truncation', () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    // Oldest user message: an image + long text that will overflow the budget.
    ctx.agent.context.appendUserMessage(
      [
        { type: 'image_url', imageUrl: { url: 'data:image/png;base64,AAAA' } },
        { type: 'text', text: 'x'.repeat(120_000) }, // ~30k tokens of text
      ],
      { kind: 'user' },
    );
    ctx.agent.context.appendUserMessage([{ type: 'text', text: 'recent user' }], { kind: 'user' });

    ctx.agent.context.applyCompaction({
      summary: 'Summary.',
      compactedCount: 2,
      tokensBefore: 100,
    });

    const keptParts = ctx.agent.context.history.flatMap((message) => message.content);
    expect(keptParts.some((part) => part.type === 'image_url')).toBe(true);
  });
});

describe('compaction — summarizer request media handling', () => {
  // GUARD — the first summarizer attempt sends media as-is: a multimodal
  // summarizer can still read an image nobody narrated. Media is only
  // replaced with text markers when the provider rejects the request body
  // as too large (see full.test.ts "request too large" cases).
  it('keeps media parts in the summarizer request when it is not rejected', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.agent.context.appendUserMessage(
      [
        { type: 'text', text: '<image path="/workspace/shot.png">' },
        { type: 'image_url', imageUrl: { url: 'data:image/png;base64,AAAA' } },
        { type: 'text', text: '</image>' },
      ],
      { kind: 'user' },
    );

    ctx.mockNextResponse({ type: 'text', text: 'Summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    const request = ctx.llmCalls.at(-1)!;
    const parts = request.history.flatMap((message) => message.content);
    expect(parts.some((part) => part.type === 'image_url')).toBe(true);
  });
});

describe('compaction — head/tail user-message retention', () => {
  const FIRST = `FIRST ${'a'.repeat(4_000)}`; // ~1k tokens
  const MIDDLE = 'b'.repeat(88_000); // ~22k tokens, over the 20k budget on its own
  const LAST = `LAST ${'c'.repeat(4_000)}`; // ~1k tokens

  async function compactedOversizedPool() {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    for (const text of [FIRST, MIDDLE, LAST]) {
      ctx.agent.context.appendUserMessage([{ type: 'text', text }]);
    }
    ctx.mockNextResponse({ type: 'text', text: 'Summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');
    return ctx;
  }

  it('splits an oversized user pool into head + elision marker + tail', async () => {
    const ctx = await compactedOversizedPool();

    const history = ctx.agent.context.history;
    const texts = historyTexts(ctx);
    // [FIRST, head slice of MIDDLE, marker, tail slice of MIDDLE, LAST, summary]
    expect(history).toHaveLength(6);
    expect(texts[0]).toBe(FIRST);
    expect(/^b+$/.test(texts[1]!)).toBe(true);
    expect(MIDDLE.startsWith(texts[1]!)).toBe(true);
    expect(history[2]!.origin).toEqual({ kind: 'injection', variant: COMPACTION_ELISION_VARIANT });
    expect(texts[2]).toContain('<system-reminder>');
    expect(texts[2]).toContain('omitted');
    expect(/^b+$/.test(texts[3]!)).toBe(true);
    expect(MIDDLE.endsWith(texts[3]!)).toBe(true);
    expect(texts[4]).toBe(LAST);
    expect(history[5]!.origin?.kind).toBe('compaction_summary');

    const completedEvent = ctx.allEvents.find((entry) => entry.event === 'compaction.completed');
    expect(completedEvent?.args).toEqual({
      result: expect.objectContaining({
        keptUserMessageCount: 4,
        keptHeadUserMessageCount: 2,
      }),
    });

    await ctx.expectResumeMatches();
  });

  it('does not stack elision markers or re-summarize them across repeated compactions', async () => {
    const ctx = await compactedOversizedPool();

    ctx.agent.context.appendUserMessage([{ type: 'text', text: 'd'.repeat(8_000) }]);
    ctx.mockNextResponse({ type: 'text', text: 'Second summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    const markers = ctx.agent.context.history.filter(
      (message) =>
        message.origin?.kind === 'injection' && message.origin.variant === COMPACTION_ELISION_VARIANT,
    );
    expect(markers).toHaveLength(1);
    const summaries = ctx.agent.context.history.filter(
      (message) => message.origin?.kind === 'compaction_summary',
    );
    expect(summaries).toHaveLength(1);
  });

  it('keeps everything verbatim (no marker) when the user pool fits the budget', async () => {
    const ctx = testAgent();
    ctx.configure({ provider: PROVIDER, modelCapabilities: CAPS });
    ctx.agent.context.appendUserMessage([{ type: 'text', text: 'small question' }]);
    ctx.mockNextResponse({ type: 'text', text: 'Summary.' });
    await ctx.rpc.beginCompaction({});
    await ctx.once('compaction.completed');

    expect(historyTexts(ctx)[0]).toBe('small question');
    expect(
      ctx.agent.context.history.some(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === COMPACTION_ELISION_VARIANT,
      ),
    ).toBe(false);

    const completedEvent = ctx.allEvents.find((entry) => entry.event === 'compaction.completed');
    expect(completedEvent?.args).toEqual({
      result: expect.not.objectContaining({ keptHeadUserMessageCount: expect.anything() }),
    });
  });

  it('restores a pre-split wire record with the tail-only selection and no marker', async () => {
    // A record written before the head/tail split (no `keptHeadUserMessageCount`)
    // must restore with the original tail-only selection, or the rebuilt live
    // history would diverge from the persisted keptUserMessageCount that the
    // wire-transcript reducer uses for its folded length.
    const big = 'x'.repeat(88_000); // ~22k tokens: over budget under the old algorithm too
    const records = [
      { type: 'metadata', protocol_version: AGENT_WIRE_PROTOCOL_VERSION, created_at: 1 },
      {
        type: 'context.append_message',
        message: {
          role: 'user',
          content: [{ type: 'text', text: big }],
          toolCalls: [],
          origin: { kind: 'user' },
        },
      },
      {
        type: 'context.append_message',
        message: {
          role: 'user',
          content: [{ type: 'text', text: 'recent question' }],
          toolCalls: [],
          origin: { kind: 'user' },
        },
      },
      {
        type: 'context.apply_compaction',
        summary: 'OLD SUMMARY',
        contextSummary: 'OLD SUMMARY',
        compactedCount: 2,
        tokensBefore: 22_007,
        tokensAfter: 20_005,
        keptUserMessageCount: 2,
      },
    ] as unknown as AgentRecord[];
    const ctx = testAgent({ persistence: new InMemoryAgentRecordPersistence(records) });
    await ctx.agent.resume();

    const history = ctx.agent.context.history;
    const texts = historyTexts(ctx);
    // Old tail-only shape: [truncated big message, recent question, summary].
    expect(history).toHaveLength(3);
    expect(
      history.some(
        (message) =>
          message.origin?.kind === 'injection' &&
          message.origin.variant === COMPACTION_ELISION_VARIANT,
      ),
    ).toBe(false);
    // The legacy truncation keeps the boundary message's beginning.
    expect(texts[0]!.length).toBeGreaterThan(0);
    expect(big.startsWith(texts[0]!)).toBe(true);
    expect(texts[1]).toBe('recent question');
    expect(history.at(-1)!.origin?.kind).toBe('compaction_summary');
  });
});
