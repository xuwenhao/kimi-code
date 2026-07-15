/**
 * `llmProtocol` provider stream cancellation scenarios — verifies that the
 * public `StreamedMessage.cancel()` contract reaches every built-in SDK
 * transport, that Google composes request/config signals, and that late native
 * streams are retired. Each SDK client is the only stubbed external boundary.
 *
 * Run: pnpm --filter @moonshot-ai/agent-core-v2 exec vitest run test/app/llmProtocol/providers/stream-cancellation.test.ts
 */

import { generate } from '#/app/llmProtocol/generate';
import { AnthropicChatProvider } from '#/app/llmProtocol/providers/anthropic';
import { GoogleGenAIChatProvider } from '#/app/llmProtocol/providers/google-genai';
import { KimiChatProvider } from '#/app/llmProtocol/providers/kimi';
import { OpenAILegacyChatProvider } from '#/app/llmProtocol/providers/openai-legacy';
import { OpenAIResponsesChatProvider } from '#/app/llmProtocol/providers/openai-responses';
import type { StreamedMessage } from '#/app/llmProtocol/provider';
import { describe, expect, it, vi } from 'vitest';

const HISTORY = [
  { role: 'user' as const, content: [{ type: 'text' as const, text: 'Hello' }], toolCalls: [] },
];

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

function nativeStream(abort: () => void): AsyncIterable<never> & {
  readonly controller: { abort(): void };
} {
  return {
    controller: { abort },
    [Symbol.asyncIterator]: () => ({
      next: () => new Promise<IteratorResult<never>>(() => {}),
    }),
  };
}

function cancel(stream: StreamedMessage): void {
  stream.cancel?.();
}

describe('built-in provider stream cancellation', () => {
  it('aborts the Kimi SDK transport when a returned stream is cancelled', async () => {
    const abort = vi.fn<() => void>();
    const create = vi.fn().mockResolvedValue(nativeStream(abort));
    const provider = new KimiChatProvider({
      model: 'test-model',
      stream: true,
      clientFactory: () => ({ chat: { completions: { create } } }) as never,
    });
    const stream = await provider.generate('', [], HISTORY);

    cancel(stream);

    expect(abort).toHaveBeenCalledOnce();
  });

  it('aborts the OpenAI chat SDK transport when a returned stream is cancelled', async () => {
    const abort = vi.fn<() => void>();
    const create = vi.fn().mockResolvedValue(nativeStream(abort));
    const provider = new OpenAILegacyChatProvider({
      model: 'test-model',
      stream: true,
      clientFactory: () => ({ chat: { completions: { create } } }) as never,
    });
    const stream = await provider.generate('', [], HISTORY);

    cancel(stream);

    expect(abort).toHaveBeenCalledOnce();
  });

  it('aborts the OpenAI responses SDK transport when a returned stream is cancelled', async () => {
    const abort = vi.fn<() => void>();
    const create = vi.fn().mockResolvedValue(nativeStream(abort));
    const provider = new OpenAIResponsesChatProvider({
      model: 'test-model',
      clientFactory: () => ({ responses: { create } }) as never,
    });
    const stream = await provider.generate('', [], HISTORY);

    cancel(stream);

    expect(abort).toHaveBeenCalledOnce();
  });

  it('aborts the Anthropic SDK transport when a returned stream is cancelled', async () => {
    const abort = vi.fn<() => void>();
    const create = vi.fn().mockResolvedValue(nativeStream(abort));
    const provider = new AnthropicChatProvider({
      model: 'test-model',
      stream: true,
      defaultMaxTokens: 1024,
      clientFactory: () => ({ messages: { create } }) as never,
    });
    const stream = await provider.generate('', [], HISTORY);

    cancel(stream);

    expect(abort).toHaveBeenCalledOnce();
  });

  it('aborts the Google SDK request signal when a returned stream is cancelled', async () => {
    let transportSignal: AbortSignal | undefined;
    const generateContentStream = vi.fn((params: Record<string, unknown>) => {
      transportSignal = (params['config'] as Record<string, unknown>)['abortSignal'] as AbortSignal;
      return Promise.resolve(nativeStream(() => {}));
    });
    const provider = new GoogleGenAIChatProvider({
      model: 'test-model',
      stream: true,
      clientFactory: () => ({ models: { generateContentStream } }) as never,
    });
    const stream = await provider.generate('', [], HISTORY);

    cancel(stream);

    expect(transportSignal?.aborted).toBe(true);
  });

  it('aborts the Google SDK request signal when the caller signal aborts', async () => {
    let transportSignal: AbortSignal | undefined;
    const generateContentStream = vi.fn((params: Record<string, unknown>) => {
      transportSignal = (params['config'] as Record<string, unknown>)['abortSignal'] as AbortSignal;
      return Promise.resolve(nativeStream(() => {}));
    });
    const provider = new GoogleGenAIChatProvider({
      model: 'test-model',
      stream: true,
      clientFactory: () => ({ models: { generateContentStream } }) as never,
    });
    const controller = new AbortController();
    const stream = await provider.generate('', [], HISTORY, { signal: controller.signal });

    controller.abort();

    expect(transportSignal?.aborted).toBe(true);
    cancel(stream);
  });

  it('includes the configured signal when a per-request Google signal is also present', async () => {
    let transportSignal: AbortSignal | undefined;
    const generateContentStream = vi.fn((params: Record<string, unknown>) => {
      transportSignal = (params['config'] as Record<string, unknown>)['abortSignal'] as AbortSignal;
      return Promise.resolve(nativeStream(() => {}));
    });
    const configuredController = new AbortController();
    const requestController = new AbortController();
    const provider = new GoogleGenAIChatProvider({
      model: 'test-model',
      stream: true,
      clientFactory: () => ({ models: { generateContentStream } }) as never,
    }).withGenerationKwargs({ abortSignal: configuredController.signal });
    const stream = await provider.generate('', [], HISTORY, { signal: requestController.signal });

    configuredController.abort();

    expect(transportSignal?.aborted).toBe(true);
    expect(requestController.signal.aborted).toBe(false);
    cancel(stream);
  });

  it('cancels a Google native stream that resolves after generate cancellation', async () => {
    const request = deferred<ReturnType<typeof nativeStream>>();
    const transportCancelled = deferred<void>();
    const abort = vi.fn(() => {
      transportCancelled.resolve(undefined);
    });
    const generateContentStream = vi.fn(() => request.promise);
    const provider = new GoogleGenAIChatProvider({
      model: 'test-model',
      stream: true,
      clientFactory: () => ({ models: { generateContentStream } }) as never,
    });
    const controller = new AbortController();
    const result = generate(provider, '', [], HISTORY, undefined, { signal: controller.signal });

    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    request.resolve(nativeStream(abort));
    await transportCancelled.promise;

    expect(abort).toHaveBeenCalledOnce();
  });
});
