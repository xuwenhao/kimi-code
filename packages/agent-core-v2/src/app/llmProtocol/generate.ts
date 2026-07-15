/**
 * `llmProtocol` domain (L0) — streams and assembles one provider response.
 *
 * Owns abort-aware provider, iterator, and streamed-part callback boundaries
 * with best-effort cleanup while preserving post-stream tool-call dispatch.
 */

import { APIEmptyResponseError } from './errors';
import {
  isContentPart,
  isToolCall,
  isToolCallPart,
  mergeInPlace,
  type Message,
  type StreamedMessagePart,
  type ToolCall,
} from './message';
import type { ChatProvider, FinishReason, GenerateOptions, StreamedMessage } from './provider';
import type { Tool } from './tool';
import type { TokenUsage } from './usage';

type StoredToolCall = Omit<ToolCall, '_streamIndex'>;

export interface GenerateResult {
  readonly id: string | null;
  readonly message: Message;
  readonly usage: TokenUsage | null;
  readonly finishReason: FinishReason | null;
  readonly rawFinishReason: string | null;
}

export interface GenerateCallbacks {
  onMessagePart?: (part: StreamedMessagePart) => void | Promise<void>;
  onToolCall?: (toolCall: ToolCall) => void | Promise<void>;
}

export async function generate(
  provider: ChatProvider,
  systemPrompt: string,
  tools: Tool[],
  history: Message[],
  callbacks?: GenerateCallbacks,
  options?: GenerateOptions,
): Promise<GenerateResult> {
  const message: Message = { role: 'assistant', content: [], toolCalls: [] };
  let pendingPart: StreamedMessagePart | null = null;

  const toolCallIndexMap = new Map<number | string, number>();

  if (options?.signal?.aborted) {
    throwAbortError();
  }

  const wireTools = tools.some((tool) => tool.deferred === true)
    ? tools.filter((tool) => tool.deferred !== true)
    : tools;

  options?.onRequestStart?.();
  const stream = await waitForAbort(
    provider.generate(systemPrompt, wireTools, history, options),
    options?.signal,
    { onLateResolve: requestCancellation },
  );

  throwIfAborted(options?.signal, stream);

  let serverDecodeMs = 0;
  let clientConsumeMs = 0;
  let firstPartAt: number | undefined;
  let lastResumeAt = 0;

  for await (const part of iterateWithAbort(stream, options?.signal)) {
    const arrivedAt = Date.now();
    if (firstPartAt === undefined) {
      firstPartAt = arrivedAt;
    } else {
      serverDecodeMs += arrivedAt - lastResumeAt;
    }

    try {
      throwIfAborted(options?.signal);

      if (callbacks?.onMessagePart !== undefined) {
        await waitForAbort(callbacks.onMessagePart(deepCopyPart(part)), options?.signal);
        throwIfAborted(options?.signal);
      }

      if (
        isToolCallPart(part) &&
        part.index !== undefined &&
        !isPendingToolCallAtIndex(pendingPart, part.index)
      ) {
        const arrayIdx = toolCallIndexMap.get(part.index);
        if (arrayIdx !== undefined) {
          const target = message.toolCalls[arrayIdx];
          if (target !== undefined && part.argumentsPart !== null) {
            target.arguments =
              target.arguments === null
                ? part.argumentsPart
                : target.arguments + part.argumentsPart;
          }
          continue;
        }
      }

      if (pendingPart === null) {
        pendingPart = part;
      } else if (!mergeInPlace(pendingPart, part)) {
        flushPart(message, pendingPart, toolCallIndexMap);
        pendingPart = part;
      }
    } finally {
      lastResumeAt = Date.now();
      clientConsumeMs += lastResumeAt - arrivedAt;
    }
  }

  throwIfAborted(options?.signal, stream);
  if (firstPartAt !== undefined) {
    serverDecodeMs += Date.now() - lastResumeAt;
  }
  options?.onStreamEnd?.(
    firstPartAt === undefined ? undefined : { serverDecodeMs, clientConsumeMs },
  );

  if (pendingPart !== null) {
    flushPart(message, pendingPart, toolCallIndexMap);
  }
  if (message.content.length === 0 && message.toolCalls.length === 0) {
    throw new APIEmptyResponseError(
      'The API returned an empty response (no content, no tool calls).' +
        formatFinishReasonHint(stream) +
        ` Provider: ${provider.name}, model: ${provider.modelName}`,
      {
        finishReason: stream.finishReason,
        rawFinishReason: stream.rawFinishReason,
      },
    );
  }

  const hasThink = message.content.some((p) => p.type === 'think');
  const hasText = message.content.some((p) => p.type === 'text' && p.text.trim().length > 0);
  const hasToolCalls = message.toolCalls.length > 0;

  if (hasThink && !hasText && !hasToolCalls) {
    throw new APIEmptyResponseError(
      'The API returned a response containing only thinking content ' +
        'without any text or tool calls. This usually indicates the ' +
        'stream was interrupted or the output token budget was exhausted ' +
        'during reasoning.' +
        formatFinishReasonHint(stream) +
        ` Provider: ${provider.name}, model: ${provider.modelName}`,
      {
        finishReason: stream.finishReason,
        rawFinishReason: stream.rawFinishReason,
      },
    );
  }

  if (callbacks?.onToolCall !== undefined && message.toolCalls.length > 0) {
    throwIfAborted(options?.signal, stream);
    for (const toolCall of message.toolCalls) {
      await callbacks.onToolCall(toolCall);
    }
  }

  return {
    id: stream.id,
    message,
    usage: stream.usage,
    finishReason: stream.finishReason,
    rawFinishReason: stream.rawFinishReason,
  };
}

type CancelableTarget = {
  cancel?: () => unknown;
  return?: () => unknown;
};

function throwAbortError(): never {
  throw createAbortError();
}

function requestCancellation(target: unknown): void {
  if (target === null || (typeof target !== 'object' && typeof target !== 'function')) {
    return;
  }
  const cancelable = target as CancelableTarget;
  invokeCancellation(cancelable, 'cancel');
  invokeCancellation(cancelable, 'return');
}

function invokeCancellation(target: CancelableTarget, method: keyof CancelableTarget): void {
  try {
    const cleanup = target[method]?.();
    void Promise.resolve(cleanup).catch(() => {});
  } catch {}
}

interface AbortWaitOptions<T> {
  readonly onAbort?: () => void;
  readonly onLateResolve?: (value: T) => void;
}

function waitForAbort<T>(
  value: T | PromiseLike<T>,
  signal?: AbortSignal,
  options?: AbortWaitOptions<T>,
): Promise<T> {
  const promise = Promise.resolve(value);
  if (signal === undefined) {
    return promise;
  }

  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const removeAbortListener = (): void => {
      signal.removeEventListener('abort', onAbort);
    };
    const onAbort = (): void => {
      if (settled) {
        return;
      }
      settled = true;
      removeAbortListener();
      try {
        options?.onAbort?.();
      } catch {}
      reject(createAbortError());
    };

    if (signal.aborted) {
      onAbort();
    } else {
      signal.addEventListener('abort', onAbort, { once: true });
    }

    void promise.then(
      (result) => {
        if (settled) {
          try {
            options?.onLateResolve?.(result);
          } catch {}
          return;
        }
        settled = true;
        removeAbortListener();
        resolve(result);
      },
      (error: unknown) => {
        if (settled) {
          return;
        }
        settled = true;
        removeAbortListener();
        reject(error);
      },
    );
  });
}

async function* iterateWithAbort(
  stream: StreamedMessage,
  signal?: AbortSignal,
): AsyncGenerator<StreamedMessagePart> {
  const iterator = stream[Symbol.asyncIterator]();
  let completed = false;
  let cancellationRequested = false;
  const cancel = (): void => {
    if (cancellationRequested) {
      return;
    }
    cancellationRequested = true;
    invokeCancellation(stream as CancelableTarget, 'cancel');
    invokeCancellation(iterator as CancelableTarget, 'return');
  };

  try {
    while (true) {
      if (signal?.aborted) {
        cancel();
        throwAbortError();
      }
      const next = await waitForAbort(iterator.next(), signal, { onAbort: cancel });
      if (signal?.aborted) {
        cancel();
        throwAbortError();
      }
      if (next.done === true) {
        completed = true;
        return;
      }
      yield next.value;
    }
  } finally {
    if (!completed) {
      cancel();
    }
  }
}

function createAbortError(): DOMException {
  return new DOMException('The operation was aborted.', 'AbortError');
}

function throwIfAborted(signal?: AbortSignal, stream?: StreamedMessage): void {
  if (!signal?.aborted) {
    return;
  }

  if (stream !== undefined) {
    requestCancellation(stream);
  }

  throwAbortError();
}

function isPendingToolCallAtIndex(
  pending: StreamedMessagePart | null,
  index: number | string,
): pending is ToolCall {
  return pending !== null && isToolCall(pending) && pending._streamIndex === index;
}

function flushPart(
  message: Message,
  part: StreamedMessagePart,
  toolCallIndexMap: Map<number | string, number>,
): void {
  if (isContentPart(part)) {
    message.content.push(part);
    return;
  }
  if (isToolCall(part)) {
    const streamIndex = part._streamIndex;
    const stored: StoredToolCall = {
      type: 'function',
      id: part.id,
      name: part.name,
      arguments: part.arguments,
      extras: part.extras,
    };
    const ordinal = message.toolCalls.length;
    message.toolCalls.push(stored as ToolCall);
    if (streamIndex !== undefined) {
      toolCallIndexMap.set(streamIndex, ordinal);
    }
  }
}

function formatFinishReasonHint(stream: StreamedMessage): string {
  if (stream.finishReason === null && stream.rawFinishReason === null) return '';

  const raw =
    stream.rawFinishReason === null ? '' : `, rawFinishReason=${stream.rawFinishReason}`;
  const filteredHint =
    stream.finishReason === 'filtered'
      ? ' The provider filtered the response before visible output was emitted.'
      : '';

  return ` Provider stop details: finishReason=${stream.finishReason ?? 'unknown'}${raw}.${filteredHint}`;
}

function deepCopyPart(part: StreamedMessagePart): StreamedMessagePart {
  return structuredClone(part);
}
