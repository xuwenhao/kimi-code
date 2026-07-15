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

/** Snapshot of a ToolCall excluding the internal `_streamIndex` routing field. */
type StoredToolCall = Omit<ToolCall, '_streamIndex'>;

/**
 * The result of a single {@link generate} call.
 *
 * Contains the fully-assembled assistant {@link message}, an optional
 * provider-assigned {@link id}, and token {@link usage} statistics.
 */
export interface GenerateResult {
  /** Provider-assigned response identifier, or `null` if unavailable. */
  readonly id: string | null;
  /** The fully-assembled assistant message with merged content parts and tool calls. */
  readonly message: Message;
  /** Token usage for this generation, or `null` if not reported. */
  readonly usage: TokenUsage | null;
  /**
   * Normalized finish reason reported by the provider, or `null` if no
   * finish_reason was emitted (for example, the stream was interrupted
   * before the final event).
   */
  readonly finishReason: FinishReason | null;
  /**
   * Raw provider-specific finish_reason string preserved verbatim.
   * `null` if the provider did not emit one.
   */
  readonly rawFinishReason: string | null;
}

export interface GenerateCallbacks {
  onMessagePart?: (part: StreamedMessagePart) => void | Promise<void>;
  /**
   * Fires once per fully-assembled tool call after the stream drains, in the
   * order tool calls appear in the final assistant message.
   *
   * Tool calls are deliberately deferred until after the stream completes:
   * parallel-tool-call streams may interleave argument deltas across calls
   * (e.g. tc0-header → tc1-header → tc0-args → tc1-args), so firing mid-stream
   * would dispatch a tool with half-parsed arguments and trigger toolParseError.
   */
  onToolCall?: (toolCall: ToolCall) => void | Promise<void>;
}

/**
 * Generate one assistant message by streaming from the given provider.
 *
 * Parts of the message are streamed and merged: consecutive compatible parts
 * (e.g. TextPart + TextPart, ToolCall + ToolCallPart) are merged in-place so
 * the returned message always contains fully-assembled parts.
 *
 * **Tool call completion** is inferred from merge boundaries (a non-merging
 * next part flushes the pending tool call into `message.toolCalls`) and from
 * stream end. Provider adapters translate native "done" signals into this
 * unified form; the generate loop never sees a separate done event.
 *
 * @param provider - The chat provider to generate from.
 * @param systemPrompt - System-level instruction prepended to the request.
 * @param tools - Tool definitions the model may invoke.
 * @param history - The conversation history sent as context.
 * @param callbacks - Optional streaming callbacks.
 * @param options - Optional per-call settings (e.g. an {@link AbortSignal}).
 *
 * @throws {DOMException} with name `"AbortError"` when `options.signal` is
 *   aborted before or during streaming.
 * @throws {APIEmptyResponseError} when the response contains no content and
 *   no tool calls, or only thinking content without any text or tool calls.
 */
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

  // Map from provider streaming index (e.g. OpenAI Chat `index`, Responses
  // `item_id`) to the position inside `message.toolCalls`. Used to route
  // interleaved argument deltas from parallel tool calls to the correct call.
  const toolCallIndexMap = new Map<number | string, number>();

  // Pre-flight abort check: if the caller's signal is already aborted, we
  // must not issue the provider request at all. Providers that do not
  // themselves honor `signal` would otherwise emit a network call that the
  // caller has explicitly cancelled.
  if (options?.signal?.aborted) {
    throwAbortError();
  }

  // Deferred tools are executable client-side but must not appear in the
  // request's top-level `tools[]` (their schemas travel via message-level
  // `tools` declarations; the top-level list stays byte-stable for prompt
  // caching). This is the single strip point for every provider call.
  const wireTools = tools.some((tool) => tool.deferred === true)
    ? tools.filter((tool) => tool.deferred !== true)
    : tools;

  options?.onRequestStart?.();
  const providerRequest = provider.generate(systemPrompt, wireTools, history, options);
  const stream = await waitForAbort(providerRequest, options?.signal, {
    // A provider may ignore AbortSignal and resolve after the caller has
    // already moved on. Retire that late stream without ever iterating it.
    onLateResolve: (lateStream) => {
      requestCancellation(lateStream);
    },
  });

  // Post-await abort check: `provider.generate()` may have resolved before
  // noticing a mid-flight abort. Reject immediately rather than draining
  // the stream.
  throwIfAborted(options?.signal, stream);

  // Decode-phase accounting. We split the window from the first streamed part
  // to stream end into time spent awaiting the next part (server + network) vs.
  // time spent processing each part in-process (deep copy, host callback, part
  // merge). `lastResumeAt` marks the end of the previous part's processing, so
  // the gap until the next part arrives is attributed to the server. The
  // per-part processing is wrapped in try/finally so the accounting stays
  // correct across `continue` and thrown aborts.
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
      // The abort-aware iterator owns stream cleanup while its yielded part is
      // being consumed. Let closing the iterator request cancellation once.
      throwIfAborted(options?.signal);

      // Notify raw part callback (deep copy to avoid aliasing mutations).
      if (callbacks?.onMessagePart !== undefined) {
        await waitForAbort(callbacks.onMessagePart(deepCopyPart(part)), options?.signal);
        throwIfAborted(options?.signal);
      }

      // Index-based routing for parallel tool call argument deltas.
      // When a ToolCallPart arrives with an index referring to a tool call
      // that is NOT the currently-pending one, append it directly to the
      // correct ToolCall in message.toolCalls instead of relying on sequential
      // merging. This prevents argument cross-contamination across parallel calls.
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
        // Unknown index — fall through to the sequential logic as a safety net.
      }

      if (pendingPart === null) {
        pendingPart = part;
      } else if (!mergeInPlace(pendingPart, part)) {
        // Could not merge — flush the pending part and start a new one.
        // For parallel tool calls this happens when a new ToolCall header arrives
        // while a previous ToolCall is still pending; the flush finalizes the
        // previous tool call into `message.toolCalls`.
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
    // Tail wait: from the last processed part to the stream's done signal.
    serverDecodeMs += Date.now() - lastResumeAt;
  }
  options?.onStreamEnd?.(
    firstPartAt === undefined ? undefined : { serverDecodeMs, clientConsumeMs },
  );

  // Flush the last pending part.
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

  // Think-only response (no real text, no tool calls) is treated as incomplete.
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

  // Fire onToolCall for every fully-assembled tool call, in final order.
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

/**
 * Ask provider-owned resources to stop without making caller cancellation
 * depend on provider cooperation. Both synchronous throws and eventual
 * rejections are contained: cancellation is best-effort cleanup, never a new
 * reason for the turn worker to remain pending.
 */
function requestCancellation(...targets: unknown[]): void {
  const seen = new Set<unknown>();
  for (const target of targets) {
    if (
      target === null ||
      (typeof target !== 'object' && typeof target !== 'function') ||
      seen.has(target)
    ) {
      continue;
    }
    seen.add(target);
    const cancelable = target as CancelableTarget;
    invokeCancellation(cancelable, 'cancel');
    invokeCancellation(cancelable, 'return');
  }
}

function invokeCancellation(target: CancelableTarget, method: keyof CancelableTarget): void {
  try {
    const cleanup = target[method]?.();
    // Attaching a rejection handler also prevents a provider that rejects its
    // cleanup promise later from producing an unhandled rejection.
    void Promise.resolve(cleanup).catch(() => {});
  } catch {}
}

interface AbortWaitOptions<T> {
  readonly onAbort?: () => void;
  readonly onLateResolve?: (value: T) => void;
}

/**
 * Wait for an external operation while making AbortSignal authoritative even
 * when that operation ignores it. The original promise always keeps handlers
 * attached, so a late resolution/rejection is observed but cannot re-enter
 * the caller after cancellation.
 */
function waitForAbort<T>(
  value: T | PromiseLike<T>,
  signal?: AbortSignal,
  options?: AbortWaitOptions<T>,
): Promise<T> {
  const promise = Promise.resolve(value);
  if (signal === undefined) return promise;

  return new Promise<T>((resolve, reject) => {
    let settled = false;
    const removeAbortListener = (): void => {
      signal.removeEventListener('abort', onAbort);
    };
    const onAbort = (): void => {
      if (settled) return;
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
        if (settled) return;
        settled = true;
        removeAbortListener();
        reject(error);
      },
    );
  });
}

/**
 * Abort-aware adapter around an arbitrary provider iterator. In particular,
 * `next()` is raced independently for every chunk: a provider that stops
 * answering after an abort cannot pin the generate call forever. Late iterator
 * results remain consumed by `waitForAbort` and are never yielded to callbacks.
 */
async function* iterateWithAbort(
  stream: StreamedMessage,
  signal?: AbortSignal,
): AsyncGenerator<StreamedMessagePart> {
  const iterator = stream[Symbol.asyncIterator]();
  let completed = false;
  let cancellationRequested = false;
  const cancel = (): void => {
    if (cancellationRequested) return;
    cancellationRequested = true;
    // The stream owns transport cancellation; the iterator owns iteration
    // teardown. Avoid also calling stream.return / iterator.cancel, which can
    // close the same provider resource twice through wrapper aliases.
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
    if (!completed) cancel();
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

/** True when `pending` is a ToolCall whose _streamIndex equals `index`. */
function isPendingToolCallAtIndex(
  pending: StreamedMessagePart | null,
  index: number | string,
): pending is ToolCall {
  return pending !== null && isToolCall(pending) && pending._streamIndex === index;
}

/**
 * Append a fully-merged part to the message.
 *
 * - ContentPart -> message.content
 * - ToolCall    -> message.toolCalls (the `_streamIndex` routing key is
 *                  registered in the map and stripped before storage).
 * - ToolCallPart -> ignored (orphaned delta without a matching pending call)
 */
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
  // ToolCallPart: orphaned delta — silently ignore.
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

/**
 * Produce a shallow-ish copy of a StreamedMessagePart.
 *
 * This is intentionally minimal: we only need isolation for the mutable
 * string fields that `mergeInPlace` mutates (text, think, arguments).
 */
function deepCopyPart(part: StreamedMessagePart): StreamedMessagePart {
  return structuredClone(part);
}
