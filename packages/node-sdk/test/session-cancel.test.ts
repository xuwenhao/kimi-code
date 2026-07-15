// Scenario: public session cancellation across the SDK, core turn worker, and provider stream.
// Responsibilities: cancellation settlement, model-visible outcome context, and clean follow-up history.
// Wiring: real SDK/core/generation stack with only the remote model provider boundary stubbed.
// Run: pnpm --dir packages/node-sdk exec vitest run test/session-cancel.test.ts

import { writeFile } from 'node:fs/promises';
import { join } from 'node:path';

import type {
  ChatProvider,
  GenerateOptions,
  Message,
  StreamedMessage,
  StreamedMessagePart,
  Tool,
} from '@moonshot-ai/kosong';
import type * as KosongModule from '@moonshot-ai/kosong';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createKimiHarness, type KimiError, type Event } from '#/index';

import { makeTempDir, removeTempDirs, waitForSDKEvent } from './session-runtime-helpers';
import { TEST_IDENTITY } from './test-identity';

const fakeProviderState = vi.hoisted(() => ({
  generate: undefined as ChatProvider['generate'] | undefined,
}));

vi.mock('@moonshot-ai/kosong', async (importOriginal) => {
  const actual = await importOriginal<typeof KosongModule>();
  return {
    ...actual,
    createProvider: () => ({
      name: 'fake',
      modelName: 'fake-model',
      thinkingEffort: null,
      async generate(
        systemPrompt: string,
        tools: Tool[],
        history: Message[],
        options?: GenerateOptions,
      ) {
        if (fakeProviderState.generate !== undefined) {
          return fakeProviderState.generate(systemPrompt, tools, history, options);
        }
        await waitForAbort(options?.signal);
        throwAbortError();
      },
      withThinking() {
        return this;
      },
    }),
  };
});

const tempDirs: string[] = [];

beforeEach(() => {
  fakeProviderState.generate = undefined;
});

afterEach(async () => {
  fakeProviderState.generate = undefined;
  await removeTempDirs(tempDirs);
});

describe('Session.cancel', () => {
  it('cancels an active streaming turn and emits turn_ended(cancelled)', async () => {
    const homeDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-home-');
    const workDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-work-');
    await writeFakeModelConfig(homeDir);
    const harness = createKimiHarness({ homeDir, identity: TEST_IDENTITY });

    try {
      const session = await harness.createSession({ id: 'ses_cancel_active_turn', workDir });
      const events: Event[] = [];
      const unsubscribe = session.onEvent((event) => {
        events.push(event);
      });
      const started = waitForSDKEvent(session, (event) => event.type === 'turn.started');
      const ended = waitForSDKEvent(session, (event) => event.type === 'turn.ended');

      await session.prompt('start a turn that will be cancelled');
      const startedEvent = await started;
      await session.cancel();
      const endedEvent = await ended;
      unsubscribe();

      expect(startedEvent).toMatchObject({
        type: 'turn.started',
        sessionId: session.id,
      });
      expect(endedEvent).toMatchObject({
        type: 'turn.ended',
        sessionId: session.id,
        turnId: startedEvent.type === 'turn.started' ? startedEvent.turnId : undefined,
        reason: 'cancelled',
      });
      expect(events).toContainEqual(expect.objectContaining({ type: 'turn.started' }));
      expect(events).toContainEqual(expect.objectContaining({ type: 'turn.ended' }));
    } finally {
      await harness.close();
    }
  });

  it('establishes a clean follow-up boundary when provider stream cleanup never settles', async () => {
    const homeDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-boundary-home-');
    const workDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-boundary-work-');
    await writeFakeModelConfig(homeDir);
    const provider = installUncooperativeStreamProvider();
    const harness = createKimiHarness({ homeDir, identity: TEST_IDENTITY });
    let cancel: Promise<void> | undefined;
    let unsubscribe: (() => void) | undefined;
    let unsubscribeFollowUp: (() => void) | undefined;

    try {
      const session = await harness.createSession({ id: 'ses_cancel_clean_boundary', workDir });
      const events: Event[] = [];
      unsubscribe = session.onEvent((event) => {
        events.push(event);
      });

      await session.prompt('Start work, then wait.');
      await provider.firstNextStarted;
      cancel = session.cancel();
      await provider.abortObserved;

      expect(provider.cleanupRequested).toBe(true);
      expect(provider.cleanupRequestCount).toBe(1);
      await cancel;
      expect(provider.cleanupReleased).toBe(false);
      expect(events).toContainEqual(
        expect.objectContaining({
          type: 'turn.ended',
          sessionId: session.id,
          reason: 'cancelled',
        }),
      );
      expect((await session.getContext()).history.at(-1)).toMatchObject({
        role: 'user',
        content: [
          {
            type: 'text',
            text: expect.stringContaining('The user interrupted the previous turn'),
          },
        ],
        origin: { kind: 'injection', variant: 'turn_outcome' },
      });

      const followUpEnded = createDeferred<void>();
      unsubscribeFollowUp = session.onEvent((event) => {
        if (event.type === 'turn.ended' && event.reason === 'completed') {
          followUpEnded.resolve(undefined);
        }
      });
      await session.prompt('Continue.');
      await provider.followUpNextStarted;
      await provider.releaseLateOutput();
      provider.completeFollowUp();
      await followUpEnded.promise;
      unsubscribeFollowUp();
      unsubscribeFollowUp = undefined;

      expect(provider.histories[1]?.slice(-3)).toEqual([
        {
          role: 'user',
          content: [{ type: 'text', text: 'Start work, then wait.' }],
          toolCalls: [],
        },
        {
          role: 'user',
          content: [
            {
              type: 'text',
              text: [
                '<system-reminder>',
                'The user interrupted the previous turn before it finished.',
                '',
                'Some operations may already have taken effect. Treat the next user message as a follow-up, and check existing state before repeating operations.',
                '</system-reminder>',
              ].join('\n'),
            },
          ],
          toolCalls: [],
        },
        {
          role: 'user',
          content: [{ type: 'text', text: 'Continue.' }],
          toolCalls: [],
        },
      ]);
      expect(JSON.stringify(provider.histories[1])).not.toContain('late output');
      expect(JSON.stringify((await session.getContext()).history)).not.toContain('late output');
    } finally {
      unsubscribe?.();
      unsubscribeFollowUp?.();
      provider.completeFollowUp();
      await provider.releaseLateOutput();
      provider.releaseCleanup();
      await cancel?.catch(() => undefined);
      await harness.close();
    }
  });

  it('rejects manual compaction on an empty session with compaction.unable', async () => {
    const homeDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-compact-home-');
    const workDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-compact-work-');
    await writeFakeModelConfig(homeDir);
    const harness = createKimiHarness({ homeDir, identity: TEST_IDENTITY });

    try {
      const session = await harness.createSession({ id: 'ses_cancel_compaction', workDir });

      await expect(session.compact({ instruction: 'Keep the compact test pending.' })).rejects.toMatchObject({
        name: 'KimiError',
        code: 'compaction.unable',
      } satisfies Partial<KimiError>);
    } finally {
      await harness.close();
    }
  });

  it('rejects after the session is closed', async () => {
    const homeDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-home-');
    const workDir = await makeTempDir(tempDirs, 'kimi-sdk-cancel-work-');
    const harness = createKimiHarness({ homeDir, identity: TEST_IDENTITY });

    try {
      const session = await harness.createSession({ id: 'ses_cancel_closed', workDir });
      await session.close();

      await expect(session.cancel()).rejects.toMatchObject({
        name: 'KimiError',
        code: 'session.closed',
      } satisfies Partial<KimiError>);
      await expect(session.cancelCompaction()).rejects.toMatchObject({
        name: 'KimiError',
        code: 'session.closed',
      } satisfies Partial<KimiError>);
    } finally {
      await harness.close();
    }
  });
});

describe('KimiHarness.forkSession', () => {
  it('rejects while the source session has an active turn', async () => {
    const homeDir = await makeTempDir(tempDirs, 'kimi-sdk-fork-active-home-');
    const workDir = await makeTempDir(tempDirs, 'kimi-sdk-fork-active-work-');
    await writeFakeModelConfig(homeDir);
    const harness = createKimiHarness({ homeDir, identity: TEST_IDENTITY });

    try {
      const session = await harness.createSession({ id: 'ses_fork_active_turn', workDir });
      const started = waitForSDKEvent(session, (event) => event.type === 'turn.started');
      const ended = waitForSDKEvent(session, (event) => event.type === 'turn.ended');

      await session.prompt('keep this turn active');
      await started;
      try {
        await expect(
          harness.forkSession({
            id: session.id,
            forkId: 'ses_fork_active_child',
          }),
        ).rejects.toMatchObject({
          name: 'KimiError',
          code: 'session.fork_active_turn',
        } satisfies Partial<KimiError>);
      } finally {
        await session.cancel().catch(() => undefined);
        await ended.catch(() => undefined);
      }
    } finally {
      await harness.close();
    }
  });
});

async function writeFakeModelConfig(homeDir: string): Promise<void> {
  await writeFile(
    join(homeDir, 'config.toml'),
    `
default_model = "fake-model"

[providers.local]
type = "kimi"
base_url = "https://example.test/v1"
api_key = "sk-test"

[models.fake-model]
provider = "local"
model = "fake-model"
max_context_size = 1000
`,
    'utf-8',
  );
}

function waitForAbort(signal: AbortSignal | undefined): Promise<void> {
  if (signal?.aborted === true) {
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    signal?.addEventListener(
      'abort',
      () => {
        resolve();
      },
      { once: true },
    );
  });
}

function throwAbortError(): never {
  throw new DOMException('The operation was aborted.', 'AbortError');
}

interface Deferred<T> {
  readonly promise: Promise<T>;
  readonly resolve: (value: T | PromiseLike<T>) => void;
}

interface UncooperativeStreamProvider {
  readonly firstNextStarted: Promise<void>;
  readonly abortObserved: Promise<void>;
  readonly followUpNextStarted: Promise<void>;
  readonly histories: readonly Message[][];
  readonly cleanupRequested: boolean;
  readonly cleanupRequestCount: number;
  readonly cleanupReleased: boolean;
  readonly completeFollowUp: () => void;
  readonly releaseLateOutput: () => Promise<void>;
  readonly releaseCleanup: () => void;
}

function installUncooperativeStreamProvider(): UncooperativeStreamProvider {
  const firstNextStarted = createDeferred<void>();
  const abortObserved = createDeferred<void>();
  const firstNext = createDeferred<IteratorResult<StreamedMessagePart>>();
  const followUpNextStarted = createDeferred<void>();
  const followUpNext = createDeferred<IteratorResult<StreamedMessagePart>>();
  const cleanupRelease = createDeferred<void>();
  const histories: Message[][] = [];
  let cleanupRequests = 0;
  let cleanupReleased = false;
  let calls = 0;
  let followUpIterations = 0;

  const requestCleanup = async (): Promise<IteratorResult<StreamedMessagePart>> => {
    cleanupRequests += 1;
    await cleanupRelease.promise;
    return { done: true, value: undefined };
  };
  const firstIterator: AsyncIterator<StreamedMessagePart> = {
    next: () => {
      firstNextStarted.resolve(undefined);
      return firstNext.promise;
    },
    return: requestCleanup,
  };
  const firstStream: StreamedMessage & {
    return: () => Promise<IteratorResult<StreamedMessagePart>>;
  } = {
    ...streamFromIterator(firstIterator),
    return: requestCleanup,
  };
  const followUpIterator: AsyncIterator<StreamedMessagePart> = {
    next: () => {
      followUpIterations += 1;
      if (followUpIterations === 1) {
        followUpNextStarted.resolve(undefined);
        return followUpNext.promise;
      }
      return Promise.resolve({ done: true, value: undefined });
    },
  };

  fakeProviderState.generate = async (_systemPrompt, _tools, history, options) => {
    histories.push(structuredClone(history));
    calls += 1;
    if (calls === 1) {
      if (options?.signal?.aborted === true) {
        abortObserved.resolve(undefined);
      } else {
        options?.signal?.addEventListener(
          'abort',
          () => {
            abortObserved.resolve(undefined);
          },
          { once: true },
        );
      }
      return firstStream;
    }
    return streamFromIterator(followUpIterator);
  };

  return {
    firstNextStarted: firstNextStarted.promise,
    abortObserved: abortObserved.promise,
    followUpNextStarted: followUpNextStarted.promise,
    histories,
    get cleanupRequested() {
      return cleanupRequests > 0;
    },
    get cleanupRequestCount() {
      return cleanupRequests;
    },
    get cleanupReleased() {
      return cleanupReleased;
    },
    completeFollowUp: () => {
      followUpNext.resolve({ done: false, value: { type: 'text', text: 'Continued.' } });
    },
    releaseLateOutput: async () => {
      const observed = firstNext.promise.then(() => undefined);
      firstNext.resolve({ done: false, value: { type: 'text', text: 'late output' } });
      await observed;
    },
    releaseCleanup: () => {
      cleanupReleased = true;
      cleanupRelease.resolve(undefined);
    },
  };
}

function streamFromIterator(iterator: AsyncIterator<StreamedMessagePart>): StreamedMessage {
  return {
    id: null,
    usage: null,
    finishReason: null,
    rawFinishReason: null,
    [Symbol.asyncIterator]: () => iterator,
  };
}

function createDeferred<T>(): Deferred<T> {
  let resolve!: Deferred<T>['resolve'];
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}
