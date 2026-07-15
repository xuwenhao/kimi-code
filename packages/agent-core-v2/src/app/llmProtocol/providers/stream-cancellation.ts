/**
 * `llmProtocol` domain (L0) — best-effort native stream transport cancellation.
 *
 * Bridges provider SDK stream handles to the protocol-level cancellation
 * contract. Owns no scoped state.
 */

type NativeCancelableStream = {
  abort?: () => unknown;
  controller?: {
    abort(): unknown;
  };
};

export function cancelNativeStream(stream: unknown): void {
  if (stream === null || (typeof stream !== 'object' && typeof stream !== 'function')) {
    return;
  }

  const cancelable = stream as NativeCancelableStream;
  try {
    if (typeof cancelable.abort === 'function') {
      void Promise.resolve(cancelable.abort()).catch(() => {});
      return;
    }
    void Promise.resolve(cancelable.controller?.abort()).catch(() => {});
  } catch {}
}
