/**
 * `contextMemory` domain helper — derives the v1-compatible full-compaction
 * handoff shape for live rewrites, wire replay, and snapshot reducers.
 */

import { estimateTokens, estimateTokensForMessage, estimateTokensForMessages } from '#/_base/utils/tokens';
import type { ContentPart } from '#/app/llmProtocol/message';
import summaryPrefixTemplate from './compaction-summary-prefix.md?raw';
import type { ContextMessage, PromptOrigin } from './types';

export const COMPACTION_SUMMARY_PREFIX = summaryPrefixTemplate.trimEnd();
export const COMPACT_USER_MESSAGE_MAX_TOKENS = 20_000;
export const COMPACT_USER_MESSAGE_HEAD_TOKENS = 2_000;
export const COMPACTION_ELISION_VARIANT = 'compaction_elision';
export const TURN_OUTCOME_VARIANT = 'turn_outcome';

const COMPACTION_SOURCE_INDEX = Symbol('compactionSourceIndex');

type MessageLike = ContextMessage;
type IndexedMessage = ContextMessage & { readonly [COMPACTION_SOURCE_INDEX]: number };

export interface CompactionUserSelection<T> {
  readonly head: T[];
  readonly tail: T[];
  readonly elided: boolean;
  readonly omittedTokens: number;
}

export interface CompactionTurnOutcome {
  readonly index: number;
  readonly message: ContextMessage;
}

export interface ContextCompactionShapeInput {
  readonly summary: string;
  readonly legacySummaryMessage?: ContextMessage;
  readonly contextSummary?: string;
  readonly compactedCount: number;
  readonly tokensBefore: number;
  readonly tokensAfter?: number;
  readonly keptUserMessageCount?: number;
  readonly keptHeadUserMessageCount?: number;
  readonly keptTurnOutcomeCount?: number;
  readonly droppedCount?: number;
  readonly legacyTail?: boolean;
}

export interface ContextCompactionShape {
  readonly summary: string;
  readonly contextSummary: string;
  readonly compactedCount: number;
  readonly tokensBefore: number;
  readonly tokensAfter: number;
  readonly keptUserMessageCount: number;
  readonly keptHeadUserMessageCount?: number;
  readonly keptTurnOutcomeCount?: number;
  readonly droppedCount?: number;
  readonly messages: readonly ContextMessage[];
}

export function buildContextCompactionShape(
  history: readonly ContextMessage[],
  input: ContextCompactionShapeInput,
): ContextCompactionShape {
  if (usesLegacyTailShape(input)) {
    const contextSummary = input.contextSummary ?? input.summary;
    const messages = [
      input.legacySummaryMessage ?? createCompactionSummaryMessage(contextSummary),
      ...history.slice(input.compactedCount),
    ];
    return {
      summary: input.summary,
      contextSummary,
      compactedCount: input.compactedCount,
      tokensBefore: input.tokensBefore,
      tokensAfter: input.tokensAfter ?? estimateTokensForMessages(messages),
      keptUserMessageCount: 0,
      keptTurnOutcomeCount: input.keptTurnOutcomeCount,
      droppedCount: input.droppedCount,
      messages,
    };
  }

  const compactableUserMessages = collectIndexedCompactableUserMessages(history);
  const selection = selectCompactionUserMessages(compactableUserMessages);
  const elisionMessage = selection.elided
    ? createCompactionElisionMessage(selection.omittedTokens)
    : undefined;
  const turnOutcome = latestTurnOutcomeAfterLastAssistant(history);
  const keptMessages = mergeCompactionHandoff(selection, elisionMessage, turnOutcome);
  const contextSummary = input.contextSummary ?? input.summary;
  const tokensAfter =
    input.tokensAfter ?? estimateTokens(contextSummary) + estimateTokensForMessages(keptMessages);
  const keptUserMessageCount =
    input.keptUserMessageCount ?? selection.head.length + selection.tail.length;
  const keptHeadUserMessageCount =
    input.keptHeadUserMessageCount ?? (selection.elided ? selection.head.length : undefined);

  return {
    summary: input.summary,
    contextSummary,
    compactedCount: input.compactedCount,
    tokensBefore: input.tokensBefore,
    tokensAfter,
    keptUserMessageCount,
    keptHeadUserMessageCount,
    keptTurnOutcomeCount: turnOutcome === undefined ? undefined : 1,
    droppedCount: input.droppedCount,
    messages: [...keptMessages, createCompactionSummaryMessage(contextSummary)],
  };
}

export function buildCompactionSummaryText(summary: string): string {
  const suffix = summary.trim();
  return `${COMPACTION_SUMMARY_PREFIX}\n${suffix.length > 0 ? suffix : '(no summary available)'}`;
}

export function createCompactionSummaryMessage(text: string): ContextMessage {
  return {
    role: 'user',
    content: [{ type: 'text', text }],
    toolCalls: [],
    origin: { kind: 'compaction_summary' },
  };
}

export function createCompactionElisionMessage(omittedTokens: number): ContextMessage {
  return {
    role: 'user',
    content: [{ type: 'text', text: buildCompactionElisionText(omittedTokens) }],
    toolCalls: [],
    origin: { kind: 'injection', variant: COMPACTION_ELISION_VARIANT },
  };
}

export function buildCompactionElisionText(omittedTokens: number): string {
  return [
    '<system-reminder>',
    `Some of this conversation's user messages were omitted here during compaction: the messages above this note are the oldest user input, the messages below are the most recent, and roughly ${String(omittedTokens)} tokens in between were dropped. The omitted content is covered by the compaction summary at the end of the conversation.`,
    '</system-reminder>',
  ].join('\n');
}

export function collectCompactableUserMessages<T extends MessageLike>(messages: readonly T[]): T[] {
  return messages.filter(
    (message) => isRealUserInput(message) && !isCompactionSummaryMessage(message),
  );
}

export function latestTurnOutcomeAfterLastAssistant(
  messages: readonly ContextMessage[],
): CompactionTurnOutcome | undefined {
  let lastAssistantIndex = -1;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index];
    if (message?.role === 'assistant' && message.origin === undefined) {
      lastAssistantIndex = index;
      break;
    }
  }
  for (let index = messages.length - 1; index > lastAssistantIndex; index--) {
    const message = messages[index];
    if (
      message?.role === 'user' &&
      message.origin?.kind === 'injection' &&
      message.origin.variant === TURN_OUTCOME_VARIANT
    ) {
      return { index, message };
    }
  }
  return undefined;
}

export function isCompactionSummaryMessage(message: MessageLike): boolean {
  return message.origin?.kind === 'compaction_summary';
}

export function isRealUserInput(message: MessageLike): boolean {
  return message.role === 'user' && compactionUserMessageDisposition(message.origin) === 'keep';
}

export function compactionUserMessageDisposition(
  origin: PromptOrigin | undefined,
): 'keep' | 'drop' {
  if (origin === undefined) return 'keep';
  switch (origin.kind) {
    case 'user':
      return 'keep';
    case 'skill_activation':
    case 'plugin_command':
      return origin.trigger === 'user-slash' ? 'keep' : 'drop';
    case 'injection':
    case 'shell_command':
    case 'compaction_summary':
    case 'system_trigger':
    case 'task':
    case 'cron_job':
    case 'cron_missed':
    case 'hook_result':
    case 'retry':
      return 'drop';
    default: {
      const exhaustive: never = origin;
      void exhaustive;
      return 'drop';
    }
  }
}

export function selectRecentUserMessages<T extends MessageLike>(
  messages: readonly T[],
  maxTokens: number = COMPACT_USER_MESSAGE_MAX_TOKENS,
): T[] {
  const selected: T[] = [];
  let remaining = maxTokens;
  for (let i = messages.length - 1; i >= 0 && remaining > 0; i--) {
    const message = messages[i]!;
    const tokens = estimateTokensForMessage(message);
    if (tokens <= remaining) {
      selected.push(message);
      remaining -= tokens;
    } else {
      selected.push(truncateUserMessage(message, remaining));
      break;
    }
  }
  selected.reverse();
  return selected;
}

export function selectCompactionUserMessages<T extends MessageLike>(
  messages: readonly T[],
  maxTokens: number = COMPACT_USER_MESSAGE_MAX_TOKENS,
  headTokens: number = COMPACT_USER_MESSAGE_HEAD_TOKENS,
): CompactionUserSelection<T> {
  let totalTokens = 0;
  for (const message of messages) {
    totalTokens += estimateTokensForMessage(message);
  }
  if (totalTokens <= maxTokens) {
    return { head: [], tail: [...messages], elided: false, omittedTokens: 0 };
  }

  const headBudget = Math.min(Math.max(headTokens, 0), maxTokens);
  const tailBudget = maxTokens - headBudget;
  const tail: T[] = [];
  let tailRemaining = tailBudget;
  let headEndExclusive = messages.length;
  let tailBoundaryDroppedPrefix: T | null = null;
  for (let i = messages.length - 1; i >= 0 && tailRemaining > 0; i--) {
    const message = messages[i]!;
    const tokens = estimateTokensForMessage(message);
    if (tokens <= tailRemaining) {
      tail.push(message);
      tailRemaining -= tokens;
      headEndExclusive = i;
      continue;
    }
    const fullText = extractText(message.content);
    const keptSuffix = truncateTextToTokensFromEnd(fullText, tailRemaining);
    tail.push(replaceMessageText(message, keptSuffix));
    headEndExclusive = i;
    const droppedPrefix = fullText.slice(0, fullText.length - keptSuffix.length);
    if (droppedPrefix.length > 0) {
      tailBoundaryDroppedPrefix = replaceMessageText(message, droppedPrefix);
    }
    break;
  }
  tail.reverse();

  const headCandidates = messages.slice(0, headEndExclusive);
  if (tailBoundaryDroppedPrefix !== null) {
    headCandidates.push(tailBoundaryDroppedPrefix);
  }
  const head: T[] = [];
  let headRemaining = headBudget;
  for (const message of headCandidates) {
    if (headRemaining <= 0) break;
    const tokens = estimateTokensForMessage(message);
    if (tokens <= headRemaining) {
      head.push(message);
      headRemaining -= tokens;
      continue;
    }
    head.push(truncateUserMessage(message, headRemaining));
    break;
  }

  let keptTokens = 0;
  for (const message of head) keptTokens += estimateTokensForMessage(message);
  for (const message of tail) keptTokens += estimateTokensForMessage(message);
  return { head, tail, elided: true, omittedTokens: Math.max(0, totalTokens - keptTokens) };
}

function usesLegacyTailShape(input: ContextCompactionShapeInput): boolean {
  return input.legacyTail === true;
}

function collectIndexedCompactableUserMessages(
  history: readonly ContextMessage[],
): IndexedMessage[] {
  const messages: IndexedMessage[] = [];
  history.forEach((message, index) => {
    if (!isRealUserInput(message) || isCompactionSummaryMessage(message)) return;
    messages.push({ ...message, [COMPACTION_SOURCE_INDEX]: index });
  });
  return messages;
}

function mergeCompactionHandoff(
  selection: CompactionUserSelection<IndexedMessage>,
  elisionMessage: ContextMessage | undefined,
  turnOutcome: CompactionTurnOutcome | undefined,
): ContextMessage[] {
  const head = [...selection.head];
  const tail = [...selection.tail];
  if (turnOutcome !== undefined) {
    const headInsertAt = head.findIndex(
      (message) => message[COMPACTION_SOURCE_INDEX] > turnOutcome.index,
    );
    if (headInsertAt >= 0) {
      head.splice(headInsertAt, 0, withSourceIndex(turnOutcome.message, turnOutcome.index));
    } else {
      const tailInsertAt = tail.findIndex(
        (message) => message[COMPACTION_SOURCE_INDEX] > turnOutcome.index,
      );
      tail.splice(
        tailInsertAt < 0 ? tail.length : tailInsertAt,
        0,
        withSourceIndex(turnOutcome.message, turnOutcome.index),
      );
    }
  }
  const indexed = elisionMessage === undefined
    ? [...head, ...tail]
    : [...head, elisionMessage, ...tail];
  return indexed.map(stripSourceIndex);
}

function withSourceIndex(message: ContextMessage, index: number): IndexedMessage {
  return { ...message, [COMPACTION_SOURCE_INDEX]: index };
}

function stripSourceIndex(message: ContextMessage | IndexedMessage): ContextMessage {
  const { [COMPACTION_SOURCE_INDEX]: _index, ...plain } = message as IndexedMessage;
  void _index;
  return plain;
}

function extractText(content: readonly ContentPart[]): string {
  let text = '';
  for (const part of content) {
    if (part.type === 'text') {
      text += part.text;
    }
  }
  return text;
}

function truncateTextToTokens(text: string, maxTokens: number): string {
  if (maxTokens <= 0) return '';
  let asciiCount = 0;
  let nonAsciiCount = 0;
  let end = 0;
  for (const char of text) {
    if (char.codePointAt(0)! <= 127) {
      asciiCount++;
    } else {
      nonAsciiCount++;
    }
    if (Math.ceil(asciiCount / 4) + nonAsciiCount > maxTokens) break;
    end += char.length;
  }
  return text.slice(0, end);
}

function truncateTextToTokensFromEnd(text: string, maxTokens: number): string {
  if (maxTokens <= 0) return '';
  let asciiCount = 0;
  let nonAsciiCount = 0;
  let start = text.length;
  for (let i = text.length - 1; i >= 0; i--) {
    let isAscii = false;
    const code = text.charCodeAt(i);
    if (code >= 0xdc00 && code <= 0xdfff && i > 0) {
      const high = text.charCodeAt(i - 1);
      if (high >= 0xd800 && high <= 0xdbff) {
        i--;
      }
    } else {
      isAscii = code <= 127;
    }
    if (isAscii) {
      asciiCount++;
    } else {
      nonAsciiCount++;
    }
    if (Math.ceil(asciiCount / 4) + nonAsciiCount > maxTokens) break;
    start = i;
  }
  return text.slice(start);
}

function replaceMessageText<T extends MessageLike>(message: T, text: string): T {
  return {
    ...message,
    content: [{ type: 'text', text }],
    toolCalls: [],
  } as unknown as T;
}

function truncateUserMessage<T extends MessageLike>(message: T, maxTokens: number): T {
  return replaceMessageText(message, truncateTextToTokens(extractText(message.content), maxTokens));
}
