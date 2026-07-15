/**
 * `fullCompaction` domain (L4) — wire Model (`CompactionModel`) and the
 * `full_compaction.begin` (`fullCompactionBegin`) / `full_compaction.cancel`
 * (`fullCompactionCancel`) / `full_compaction.complete`
 * (`fullCompactionComplete`) Ops that mirror the full-compaction lifecycle
 * into a persisted, replayable phase, plus the `compaction.*` edge events
 * (`started` / `blocked` / `cancelled` / `completed`) declared on `DomainEventMap`
 * (`compaction.started` is derived from the `full_compaction.begin` Op's
 * `toEvent`; the rest publish directly from the service).
 *
 * The Model is intentionally phase-only — `{ phase }` (initial `idle`). A
 * `context.apply_compaction` cross-reducer advances a running compaction to
 * `committed`, making the context replacement the durable commit point. The
 * richer per-compaction data is not resume state: `instruction` is only needed
 * by the live worker and telemetry, while the committed summary and accounting
 * already live in `contextMemory`. A pre-commit failure uses the existing
 * `full_compaction.complete` record type with `outcome: 'failed'`, preserving
 * the v1 wire vocabulary while distinguishing it from a successful terminal;
 * legacy complete records omit that field and remain successful. Older records
 * may also carry result numbers, which `apply` ignores. Each reducer returns the
 * same reference on a no-op so the wire's reference-equality gate stays quiet
 * and carries no non-determinism.
 *
 * The runtime orchestration — `ActiveCompaction`, its `AbortController`, and
 * the in-flight worker promise — stays OUT of the Model (live-only service
 * members): none of it can be resumed. The service's `wire.hooks.onDidRestore`
 * hook fails a stranded pre-commit run and converges a committed run before
 * returning the Agent to `idle`.
 *
 * The `compaction.*` events publish to `IEventBus` (`compaction.started` via the
 * `begin` Op's `toEvent`; the rest directly from the service); they are
 * declared here via interface-merge (`error` is already declared by `mcp`, so
 * it is not re-declared). The `full_compaction.*` record shapes are registered in
 * `PersistedOpMap` (`#/wire/types`, below) because the records still
 * ride the per-agent `wire.jsonl` journal restored by `IWireService`.
 * Consumed by the Agent-scope `fullCompactionService`.
 */

import { z } from 'zod';

import { defineModel } from '#/wire/model';

import type { CompactionBeginData, CompactionResult } from './types';

export interface CompactionStartedEvent {
  readonly type: 'compaction.started';
  readonly trigger: 'manual' | 'auto';
  readonly instruction?: string;
}

export interface CompactionBlockedEvent {
  readonly type: 'compaction.blocked';
  readonly turnId?: number;
}

export interface CompactionCancelledEvent {
  readonly type: 'compaction.cancelled';
}

export interface CompactionCompletedEvent {
  readonly type: 'compaction.completed';
  readonly result: CompactionResult;
}

export type CompactionPhase = 'idle' | 'running' | 'committed';

export interface CompactionState {
  readonly phase: CompactionPhase;
}

export const CompactionModel = defineModel<CompactionState>(
  'fullCompaction',
  () => ({ phase: 'idle' }),
  {
    reducers: {
      'context.apply_compaction': (state) =>
        state.phase === 'running' ? { phase: 'committed' } : state,
    },
  },
);

declare module '#/app/event/eventBus' {
  interface DomainEventMap {
    'compaction.started': CompactionStartedEvent;
    'compaction.blocked': CompactionBlockedEvent;
    'compaction.cancelled': CompactionCancelledEvent;
    'compaction.completed': CompactionCompletedEvent;
  }
}

declare module '#/wire/types' {
  interface PersistedOpMap {
    'full_compaction.begin': typeof fullCompactionBegin;
    'full_compaction.cancel': typeof fullCompactionCancel;
    'full_compaction.complete': typeof fullCompactionComplete;
  }
}

export const fullCompactionBegin = CompactionModel.defineOp('full_compaction.begin', {
  schema: z.custom<CompactionBeginData>(),
  apply: (s) => (s.phase === 'running' ? s : { phase: 'running' }),
  toEvent: (p) => ({
    type: 'compaction.started' as const,
    trigger: p.source,
    instruction: p.instruction,
  }),
});

export const fullCompactionCancel = CompactionModel.defineOp('full_compaction.cancel', {
  schema: z.object({}),
  apply: (s) => (s.phase === 'idle' ? s : { phase: 'idle' }),
});

export const fullCompactionComplete = CompactionModel.defineOp('full_compaction.complete', {
  schema: z.object({ outcome: z.literal('failed').optional() }),
  apply: (s) => (s.phase === 'idle' ? s : { phase: 'idle' }),
});

export function fullCompactionFail(): ReturnType<typeof fullCompactionComplete> {
  return fullCompactionComplete({ outcome: 'failed' });
}
