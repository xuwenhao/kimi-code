const appendAcceptance = new WeakMap<Error, boolean>();

/**
 * Preserve the original error identity while recording whether the requested
 * record crossed the persistence append boundary. Callers can then retry an
 * explicitly rejected append without duplicating one whose observer failed
 * after acceptance.
 */
export function markAgentRecordAppendError(error: unknown, accepted: boolean): Error {
  const normalized =
    error instanceof Error ? error : new Error(String(error), { cause: error });
  appendAcceptance.set(normalized, accepted);
  return normalized;
}

/** `undefined` means the persistence implementation supplied no acceptance contract. */
export function agentRecordAppendAccepted(error: unknown): boolean | undefined {
  return error instanceof Error ? appendAcceptance.get(error) : undefined;
}
