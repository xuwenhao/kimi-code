/**
 * `llmProtocol` generation cancellation scenarios — verifies abort authority
 * across the provider request, streamed iteration, and streamed-part callback
 * while keeping tool-call dispatch after stream completion. The provider is
 * the only stubbed external boundary.
 *
 * Run: pnpm --filter @moonshot-ai/agent-core-v2 exec vitest run test/app/llmProtocol/generate.test.ts
 */

import { generate } from '#/app/llmProtocol/generate';
import type { StreamedMessagePart } from '#/app/llmProtocol/message';
import type {
  ChatProvider,
  GenerateOptions,
  StreamedMessage,
  ThinkingEffort,
} from '#/app/llmProtocol/provider';
import { describe, expect, it, vi } from 'vitest';

interface Deferred<T> {
  readonly promise: Promise<T>;
  readonly resolve: (value: T) => void;
}

function deferred<T>(): Deferred<T> {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

function createProvider(
  request: (options?: GenerateOptions) => Promise<StreamedMessage>,
): ChatProvider {
  return {
    name: 'test-provider',
    modelName: 'test-model',
    thinkingEffort: null,
    generate: (_systemPrompt, _tools, _history, options) => request(options),
    withThinking(_effort: ThinkingEffort): ChatProvider {
      return this;
    },
  };
}

function createStream(iterator: AsyncIterator<StreamedMessagePart>): StreamedMessage {
  return {
    id: 'response-1',
    usage: null,
    finishReason: 'completed',
    rawFinishReason: 'stop',
    [Symbol.asyncIterator]: () => iterator,
  };
}

describe('generate cancellation boundaries', () => {
  it('rejects with AbortError when the provider request never settles', async () => {
    const controller = new AbortController();
    const request = deferred<StreamedMessage>();
    const result = generate(
      createProvider(() => request.promise),
      '',
      [],
      [],
      undefined,
      { signal: controller.signal },
    );

    controller.abort();

    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('retires a stream that the provider resolves after cancellation', async () => {
    const controller = new AbortController();
    const request = deferred<StreamedMessage>();
    const cancel = vi.fn<() => void>();
    const streamIterator = vi.fn<() => AsyncIterator<StreamedMessagePart>>();
    const lateStream: StreamedMessage & { cancel(): void } = {
      id: 'late-response',
      usage: null,
      finishReason: null,
      rawFinishReason: null,
      cancel,
      [Symbol.asyncIterator]: streamIterator,
    };
    const result = generate(
      createProvider(() => request.promise),
      '',
      [],
      [],
      undefined,
      { signal: controller.signal },
    );

    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    request.resolve(lateStream);
    await request.promise;

    expect(cancel).toHaveBeenCalledOnce();
    expect(streamIterator).not.toHaveBeenCalled();
  });

  it('rejects without waiting when iterator progress and cleanup never settle', async () => {
    const controller = new AbortController();
    const nextStarted = deferred<void>();
    const nextResult = deferred<IteratorResult<StreamedMessagePart>>();
    const cancel = vi.fn(() => Promise.reject(new Error('cleanup failed')));
    const returnIterator = vi.fn(() => new Promise<IteratorResult<StreamedMessagePart>>(() => {}));
    const iterator: AsyncIterator<StreamedMessagePart> = {
      next: () => {
        nextStarted.resolve(undefined);
        return nextResult.promise;
      },
      return: returnIterator,
    };
    const stream = Object.assign(createStream(iterator), { cancel });
    const result = generate(
      createProvider(() => Promise.resolve(stream)),
      '',
      [],
      [],
      undefined,
      { signal: controller.signal },
    );
    await nextStarted.promise;

    controller.abort();

    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancel).toHaveBeenCalledOnce();
    expect(returnIterator).toHaveBeenCalledOnce();
  });

  it('does not deliver an iterator result that arrives after cancellation', async () => {
    const controller = new AbortController();
    const nextStarted = deferred<void>();
    const nextResult = deferred<IteratorResult<StreamedMessagePart>>();
    const iterator: AsyncIterator<StreamedMessagePart> = {
      next: () => {
        nextStarted.resolve(undefined);
        return nextResult.promise;
      },
      return: () => Promise.resolve({ done: true, value: undefined }),
    };
    const onMessagePart = vi.fn<(part: StreamedMessagePart) => void>();
    const result = generate(
      createProvider(() => Promise.resolve(createStream(iterator))),
      '',
      [],
      [],
      { onMessagePart },
      { signal: controller.signal },
    );
    await nextStarted.promise;

    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    nextResult.resolve({ done: false, value: { type: 'text', text: 'late output' } });
    await nextResult.promise;

    expect(onMessagePart).not.toHaveBeenCalled();
  });

  it('does not deliver a chunk when cancellation lands between next resolution and yield', async () => {
    const controller = new AbortController();
    const nextStarted = deferred<void>();
    const nextResult = deferred<IteratorResult<StreamedMessagePart>>();
    const returnIterator = vi.fn<
      () => Promise<IteratorResult<StreamedMessagePart>>
    >(() => Promise.resolve({ done: true, value: undefined }));
    const iterator: AsyncIterator<StreamedMessagePart> = {
      next: () => {
        nextStarted.resolve(undefined);
        return nextResult.promise;
      },
      return: returnIterator,
    };
    const onMessagePart = vi.fn<(part: StreamedMessagePart) => void>();
    const result = generate(
      createProvider(() => Promise.resolve(createStream(iterator))),
      '',
      [],
      [],
      { onMessagePart },
      { signal: controller.signal },
    );
    await nextStarted.promise;
    void nextResult.promise.then(() => {
      controller.abort();
    });

    nextResult.resolve({ done: false, value: { type: 'text', text: 'raced output' } });

    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(onMessagePart).not.toHaveBeenCalled();
    expect(returnIterator).toHaveBeenCalledOnce();
  });

  it('rejects with AbortError when the message callback never settles', async () => {
    const controller = new AbortController();
    const callbackStarted = deferred<void>();
    const callbackResult = deferred<void>();
    let nextCount = 0;
    const iterator: AsyncIterator<StreamedMessagePart> = {
      next: () => {
        nextCount += 1;
        return Promise.resolve(
          nextCount === 1
            ? { done: false, value: { type: 'text', text: 'partial output' } }
            : { done: true, value: undefined },
        );
      },
      return: () => Promise.resolve({ done: true, value: undefined }),
    };
    const result = generate(
      createProvider(() => Promise.resolve(createStream(iterator))),
      '',
      [],
      [],
      {
        onMessagePart: () => {
          callbackStarted.resolve(undefined);
          return callbackResult.promise;
        },
      },
      { signal: controller.signal },
    );
    await callbackStarted.promise;

    controller.abort();

    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
  });

  it('requests each iteration cleanup operation once when callback settlement races cancellation', async () => {
    const controller = new AbortController();
    const callbackStarted = deferred<void>();
    const callbackResult = deferred<void>();
    const cancel = vi.fn<() => void>();
    const returnIterator = vi.fn<
      () => Promise<IteratorResult<StreamedMessagePart>>
    >(() => Promise.resolve({ done: true, value: undefined }));
    let nextCount = 0;
    const iterator: AsyncIterator<StreamedMessagePart> = {
      next: () => {
        nextCount += 1;
        return Promise.resolve(
          nextCount === 1
            ? { done: false, value: { type: 'text', text: 'partial output' } }
            : { done: true, value: undefined },
        );
      },
      return: returnIterator,
    };
    const stream = Object.assign(createStream(iterator), { cancel });
    const result = generate(
      createProvider(() => Promise.resolve(stream)),
      '',
      [],
      [],
      {
        onMessagePart: () => {
          callbackStarted.resolve(undefined);
          return callbackResult.promise;
        },
      },
      { signal: controller.signal },
    );
    await callbackStarted.promise;
    void callbackResult.promise.then(() => {
      controller.abort();
    });

    callbackResult.resolve(undefined);

    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(cancel).toHaveBeenCalledOnce();
    expect(returnIterator).toHaveBeenCalledOnce();
  });

  it('dispatches assembled tool calls only after the stream completes', async () => {
    const events: string[] = [];
    let nextCount = 0;
    const iterator: AsyncIterator<StreamedMessagePart> = {
      next: () => {
        nextCount += 1;
        if (nextCount === 1) {
          return Promise.resolve({
            done: false,
            value: { type: 'function', id: 'tool-1', name: 'read_file', arguments: '{}' },
          });
        }
        events.push('stream completed');
        return Promise.resolve({ done: true, value: undefined });
      },
    };

    await generate(createProvider(() => Promise.resolve(createStream(iterator))), '', [], [], {
      onToolCall: () => {
        events.push('tool dispatched');
      },
    });

    expect(events).toEqual(['stream completed', 'tool dispatched']);
  });

  it('dispatches every completed tool call when the first post-stream callback aborts', async () => {
    const controller = new AbortController();
    const dispatched: string[] = [];
    let nextCount = 0;
    const iterator: AsyncIterator<StreamedMessagePart> = {
      next: () => {
        nextCount += 1;
        if (nextCount === 1) {
          return Promise.resolve({
            done: false,
            value: { type: 'function', id: 'tool-1', name: 'read_file', arguments: '{}' },
          });
        }
        if (nextCount === 2) {
          return Promise.resolve({
            done: false,
            value: { type: 'function', id: 'tool-2', name: 'write_file', arguments: '{}' },
          });
        }
        return Promise.resolve({ done: true, value: undefined });
      },
    };

    const result = await generate(
      createProvider(() => Promise.resolve(createStream(iterator))),
      '',
      [],
      [],
      {
        onToolCall: (toolCall) => {
          dispatched.push(toolCall.id);
          if (toolCall.id === 'tool-1') {
            controller.abort();
          }
        },
      },
      { signal: controller.signal },
    );

    expect(dispatched).toEqual(['tool-1', 'tool-2']);
    expect(result.message.toolCalls).toEqual([
      { type: 'function', id: 'tool-1', name: 'read_file', arguments: '{}' },
      { type: 'function', id: 'tool-2', name: 'write_file', arguments: '{}' },
    ]);
  });
});
