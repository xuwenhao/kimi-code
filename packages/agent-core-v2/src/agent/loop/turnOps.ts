/**
 * `loop` domain (L4) — wire Model (`TurnModel`) and the Ops that bookkeep the
 * agent's turn lifecycle and abnormal outcome handoffs on the wire.
 *
 * Declares the next turn id as a wire Model (initial `0`). The persisted
 * `turn.prompt` record carries exactly v1's field set (`{ input, origin }` —
 * no `turnId`), and `apply` mirrors v1's `restorePrompt()`: every record
 * advances the counter by one, so the counter is restored by counting
 * turn starts. Every turn is started by `loopService.enqueue` admitting a
 * request that creates a new Turn, which dispatches one
 * `turn.prompt` per start. `turn.cancel` carries cancellation reminder intent
 * atomically, while `turn.outcome` records failure intent; a
 * `context.append_message` acknowledgement consumes the intent after the
 * reminder reaches model context. As a belt-and-suspenders for v1-written logs
 * whose internally-driven turns (goal continuations) have no `turn.prompt` record,
 * `TurnModel` also registers a cross-model reducer on
 * `context.append_loop_event` that raises the counter past any `turnId`
 * observed in a replayed loop event — the v1 `observeRestoredTurnId`
 * semantics. The `turn.started` / `turn.ended` / `error` signals are not part
 * of this Op set and remain on their existing path (published by the loop
 * service around a run). Consumed by the Agent-scope `loopService` and by the
 * `activity` kernel (which reads the next turn id on admission).
 */

import { z } from 'zod';

import { defineModel } from '#/wire/model';
import type { ContentPart } from '#/app/llmProtocol/message';
import type { PromptOrigin } from '#/agent/contextMemory/types';

export interface TurnModelState {
  readonly nextTurnId: number;
  readonly pendingOutcomes: readonly TurnOutcomeIntent[];
}

export interface TurnOutcomeIntent {
  readonly outcomeId: string;
  readonly turnId: number;
  readonly content: string;
}

export const TurnModel = defineModel<TurnModelState>(
  'turn',
  () => ({ nextTurnId: 0, pendingOutcomes: [] }),
  {
    reducers: {
      'context.append_loop_event': (state, { event }) => {
        if (event.type === 'tool.result' || event.turnId === undefined) {
          return state;
        }

        const turnId = Number.parseInt(event.turnId, 10);
        return Number.isInteger(turnId) && turnId >= state.nextTurnId
          ? { ...state, nextTurnId: turnId + 1 }
          : state;
      },
      'context.append_message': (state, payload) => {
        const outcomeId = payload.materializedTurnOutcomeId;
        if (outcomeId === undefined) return state;
        const pendingOutcomes = state.pendingOutcomes.filter(
          (outcome) => outcome.outcomeId !== outcomeId,
        );
        return pendingOutcomes.length === state.pendingOutcomes.length
          ? state
          : { ...state, pendingOutcomes };
      },
      'context.clear': (state) => clearPendingOutcomes(state),
      'context.undo': (state, payload) =>
        payload.count > 0 ? clearPendingOutcomes(state) : state,
    },
  },
);

const turnInputShape = {
  input: z.custom<readonly ContentPart[]>(),
  origin: z.custom<PromptOrigin>(),
};

declare module '#/wire/types' {
  interface PersistedOpMap {
    'turn.prompt': typeof promptTurn;
    'turn.steer': typeof steerTurn;
    'turn.cancel': typeof cancelTurn;
    'turn.outcome': typeof turnOutcome;
  }
}

export const promptTurn = TurnModel.defineOp('turn.prompt', {
  schema: z.object(turnInputShape),
  apply: (s) => ({ ...s, nextTurnId: s.nextTurnId + 1 }),
});

export const steerTurn = TurnModel.defineOp('turn.steer', {
  schema: z.object(turnInputShape),
  apply: (s) => s,
});

export const cancelTurn = TurnModel.defineOp('turn.cancel', {
  schema: z.object({
    turnId: z.number().optional(),
    outcomeId: z.string().optional(),
    outcomeTurnId: z.number().optional(),
    outcomeContent: z.string().optional(),
  }),
  apply: (s, payload) => addOutcomeFromCancel(s, payload),
});

export const turnOutcome = TurnModel.defineOp('turn.outcome', {
  schema: z.object({
    outcomeId: z.string(),
    turnId: z.number(),
    content: z.string(),
  }),
  apply: (s, payload) => addPendingOutcome(s, payload),
});

function addOutcomeFromCancel(
  state: TurnModelState,
  payload: {
    readonly outcomeId?: string;
    readonly outcomeTurnId?: number;
    readonly outcomeContent?: string;
  },
): TurnModelState {
  if (
    payload.outcomeId === undefined ||
    payload.outcomeTurnId === undefined ||
    payload.outcomeContent === undefined
  ) {
    return state;
  }
  return addPendingOutcome(state, {
    outcomeId: payload.outcomeId,
    turnId: payload.outcomeTurnId,
    content: payload.outcomeContent,
  });
}

function addPendingOutcome(
  state: TurnModelState,
  outcome: TurnOutcomeIntent,
): TurnModelState {
  if (state.pendingOutcomes.some((pending) => pending.turnId === outcome.turnId)) return state;
  return { ...state, pendingOutcomes: [...state.pendingOutcomes, outcome] };
}

function clearPendingOutcomes(state: TurnModelState): TurnModelState {
  return state.pendingOutcomes.length === 0 ? state : { ...state, pendingOutcomes: [] };
}
