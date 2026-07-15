/**
 * End-to-end smoke for image-compression caption rerouting.
 *
 * Prompt ingestion (server route, TUI paste, ACP) annotates a compressed
 * image with an inline `<system>` caption inside the user's own message.
 * The context layer must split that caption out into a hidden
 * system-reminder injection so no raw `<system>` markup is ever rendered in
 * a user bubble, while the model still receives the note.
 *
 * These tests drive the REAL pipeline — rpc.prompt → turn → context →
 * scripted provider — and assert at every seam:
 *   - the wire request the model receives (reminder present, user text clean)
 *   - the stored context history (origins drive UI hiding)
 *   - the recorded wire log (what session resume replays)
 *   - resume parity via expectResumeMatches (the TUI replay data source)
 */

import { expect, it, vi } from 'vitest';

import type { AgentRecord } from '../../src/agent';
import { InMemoryAgentRecordPersistence } from '../../src/agent/records';
import { buildImageCompressionCaption } from '../../src/tools/support/image-compress';
import { testAgent } from './harness/agent';

const CAPTION = buildImageCompressionCaption({
  original: { width: 3264, height: 666, byteLength: 344 * 1024, mimeType: 'image/png' },
  final: { width: 2000, height: 408, byteLength: 282 * 1024, mimeType: 'image/png' },
  originalPath: '/tmp/originals/shot.png',
});

const SECOND_CAPTION = buildImageCompressionCaption({
  original: { width: 4000, height: 3000, byteLength: 9 * 1024 * 1024, mimeType: 'image/jpeg' },
  final: { width: 2000, height: 1500, byteLength: 1024 * 1024, mimeType: 'image/jpeg' },
  originalPath: '/tmp/originals/photo.jpg',
});

const IMAGE_URL = 'data:image/png;base64,AAAA';

it('smoke: a compressed-image prompt reaches the model with the caption as a system reminder', async () => {
  const ctx = testAgent();
  ctx.configure();

  ctx.mockNextResponse({ type: 'text', text: 'I can see the screenshot.' });
  // The TUI merges the caption into the preceding text segment — the exact
  // shape from the bug report.
  await ctx.rpc.prompt({
    input: [
      { type: 'text', text: `能展示但是没有快捷键提示${CAPTION}` },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
    ],
  });
  await ctx.untilTurnEnd();

  // What the model actually received on the wire.
  const llmInput = JSON.stringify(ctx.lastLlmInput().input);
  expect(llmInput).toContain('<system-reminder>');
  expect(llmInput).toContain('Image compressed to fit model limits');
  expect(llmInput).toContain('/tmp/originals/shot.png');
  expect(llmInput).not.toContain('<system>');
  expect(llmInput).toContain('能展示但是没有快捷键提示');

  // What the UI renders from: the reminder is a separate injection-origin
  // message; the user message holds only what the user actually submitted.
  const history = ctx.agent.context.history;
  expect(history.map(({ role, origin }) => ({ role, origin }))).toEqual([
    { role: 'user', origin: { kind: 'injection', variant: 'image_compression' } },
    { role: 'user', origin: { kind: 'user' } },
    { role: 'assistant', origin: undefined },
  ]);
  const userText = history[1]!.content
    .map((part) => (part.type === 'text' ? part.text : ''))
    .join('');
  expect(userText).toBe('能展示但是没有快捷键提示');
  expect(history[1]!.content.some((part) => part.type === 'image_url')).toBe(true);

  // The recorded wire log is what session resume replays into the TUI: the
  // user append_message record must already be caption-free.
  const appendRecords = ctx.allEvents.filter(
    (event) => event.type === '[wire]' && event.event === 'context.append_message',
  );
  expect(appendRecords).toHaveLength(2);
  expect(JSON.stringify(appendRecords[0]!.args)).toContain('system-reminder');
  expect(JSON.stringify(appendRecords[1]!.args)).not.toContain('<system>');

  // Resume the session from the records and require identical state.
  await ctx.expectResumeMatches();
});

it('smoke: multiple compressed images in one prompt each announce via their own reminder', async () => {
  const ctx = testAgent();
  ctx.configure();

  ctx.mockNextResponse({ type: 'text', text: 'Two images noted.' });
  // Server-route shape: standalone caption part directly before each image.
  await ctx.rpc.prompt({
    input: [
      { type: 'text', text: CAPTION },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
      { type: 'text', text: SECOND_CAPTION },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
      { type: 'text', text: '对比一下这两张截图' },
    ],
  });
  await ctx.untilTurnEnd();

  const llmInput = JSON.stringify(ctx.lastLlmInput().input);
  expect(llmInput).toContain('/tmp/originals/shot.png');
  expect(llmInput).toContain('/tmp/originals/photo.jpg');
  expect(llmInput).not.toContain('<system>');

  const history = ctx.agent.context.history;
  expect(history.map(({ role, origin }) => ({ role, origin }))).toEqual([
    { role: 'user', origin: { kind: 'injection', variant: 'image_compression' } },
    { role: 'user', origin: { kind: 'injection', variant: 'image_compression' } },
    { role: 'user', origin: { kind: 'user' } },
    { role: 'assistant', origin: undefined },
  ]);
  const userMessage = history[2]!;
  expect(userMessage.content.filter((part) => part.type === 'image_url')).toHaveLength(2);
  const userText = userMessage.content
    .map((part) => (part.type === 'text' ? part.text : ''))
    .join('');
  expect(userText).toBe('对比一下这两张截图');

  await ctx.expectResumeMatches();
});

it('smoke: a steered prompt with a caption is split the same way', async () => {
  const ctx = testAgent();
  ctx.configure();

  ctx.mockNextResponse({ type: 'text', text: 'Steered.' });
  await ctx.rpc.steer({
    input: [
      { type: 'text', text: `看这张${CAPTION}` },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
    ],
  });
  await ctx.untilTurnEnd();

  const history = ctx.agent.context.history;
  expect(history.map(({ role, origin }) => ({ role, origin }))).toEqual([
    { role: 'user', origin: { kind: 'injection', variant: 'image_compression' } },
    { role: 'user', origin: { kind: 'user' } },
    { role: 'assistant', origin: undefined },
  ]);
  const userText = history[1]!.content
    .map((part) => (part.type === 'text' ? part.text : ''))
    .join('');
  expect(userText).toBe('看这张');

  await ctx.expectResumeMatches();
});

it('smoke: a prompt without images is completely untouched', async () => {
  const ctx = testAgent();
  ctx.configure();

  ctx.mockNextResponse({ type: 'text', text: 'Hi.' });
  await ctx.rpc.prompt({ input: [{ type: 'text', text: 'plain hello' }] });
  await ctx.untilTurnEnd();

  const history = ctx.agent.context.history;
  expect(history.map(({ role, origin }) => ({ role, origin }))).toEqual([
    { role: 'user', origin: { kind: 'user' } },
    { role: 'assistant', origin: undefined },
  ]);
  const userText = history[0]!.content
    .map((part) => (part.type === 'text' ? part.text : ''))
    .join('');
  expect(userText).toBe('plain hello');

  await ctx.expectResumeMatches();
});

it.each(['none', 'clear', 'compaction'] as const)(
  'recovers a multi-caption prompt exactly once after its first child record (%s)',
  async (destructiveRecord) => {
    const pendingProvider = new Promise<never>(() => undefined);
    const persistence = new InMemoryAgentRecordPersistence();
    const original = testAgent({ persistence, generate: () => pendingProvider });
    original.configure();

    await original.rpc.prompt({
      input: [
        { type: 'text', text: CAPTION },
        { type: 'image_url', imageUrl: { url: IMAGE_URL } },
        { type: 'text', text: SECOND_CAPTION },
        { type: 'image_url', imageUrl: { url: IMAGE_URL } },
        { type: 'text', text: 'CRASH-PREFIX-IMAGE-PROMPT' },
      ],
    });

    const firstChildIndex = persistence.records.findIndex(
      (record) =>
        record.type === 'context.append_message' &&
        record.turnInputPart?.index === 0,
    );
    expect(firstChildIndex).toBeGreaterThanOrEqual(0);
    const prefix = structuredClone(persistence.records.slice(0, firstChildIndex + 1));
    const crashed = new InMemoryAgentRecordPersistence(prefix);
    if (destructiveRecord === 'clear') {
      crashed.append({ type: 'context.clear', time: Date.now() });
    } else if (destructiveRecord === 'compaction') {
      crashed.append({
        type: 'context.apply_compaction',
        summary: 'CRASH-PREFIX-SUMMARY',
        contextSummary: 'CRASH-PREFIX-SUMMARY',
        compactedCount: 1,
        tokensBefore: 10,
        tokensAfter: 5,
        keptUserMessageCount: 0,
        droppedCount: 1,
        time: Date.now(),
      });
    }

    const resumed = testAgent({ persistence: crashed });
    resumed.configure();
    resumed.mockNextResponse({ type: 'text', text: 'Recovered image prompt.' });
    await resumed.agent.resume();
    await resumed.agent.turn.waitForCurrentTurn();

    const texts = resumed.agent.context.history
      .flatMap((message) => message.content)
      .filter((part) => part.type === 'text')
      .map((part) => part.text)
      .join('\n');
    expect(texts.match(/\/tmp\/originals\/shot\.png/g)).toHaveLength(1);
    expect(texts.match(/\/tmp\/originals\/photo\.jpg/g)).toHaveLength(1);
    expect(texts.match(/CRASH-PREFIX-IMAGE-PROMPT/g)).toHaveLength(1);
  },
);

it('rolls back an orphan caption when cancellation is accepted from its record observer', async () => {
  let original!: ReturnType<typeof testAgent>;
  let cancellation: Promise<void> | undefined;
  const persistence = new InMemoryAgentRecordPersistence([], {
    onRecord: (record) => {
      if (
        cancellation === undefined &&
        record.type === 'context.append_message' &&
        record.turnInputPart?.index === 0
      ) {
        cancellation = original.agent.turn.cancel(0);
      }
    },
  });
  original = testAgent({ persistence });
  original.configure();

  await original.rpc.prompt({
    input: [
      { type: 'text', text: CAPTION },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
      { type: 'text', text: SECOND_CAPTION },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
      { type: 'text', text: 'CANCELLED-IMAGE-PROMPT' },
    ],
  });
  await original.untilTurnEnd();
  await cancellation;

  const cancel = persistence.records.find(
    (record) => record.type === 'turn.cancel' && record.cancelledTurnInputs !== undefined,
  );
  expect(cancel).toBeDefined();
  const cancelIndex = persistence.records.indexOf(cancel!);
  const generate = vi.fn();
  const resumed = testAgent({
    persistence: new InMemoryAgentRecordPersistence(
      structuredClone(persistence.records.slice(0, cancelIndex + 1)),
    ),
    generate,
  });
  await resumed.agent.resume();

  const texts = resumed.agent.context.history
    .flatMap((message) => message.content)
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n');
  expect(texts).not.toContain('/tmp/originals/shot.png');
  expect(texts).not.toContain('/tmp/originals/photo.jpg');
  expect(texts).not.toContain('CANCELLED-IMAGE-PROMPT');
  expect(texts.match(/The user interrupted the previous turn/g)).toHaveLength(1);
  expect(generate).not.toHaveBeenCalled();
});

it('keeps the whole expansion when cancellation observes its accepted final record', async () => {
  let ctx!: ReturnType<typeof testAgent>;
  let cancellation: Promise<void> | undefined;
  const persistence = new InMemoryAgentRecordPersistence([], {
    onRecord: (record) => {
      if (
        cancellation === undefined &&
        record.type === 'context.append_message' &&
        record.turnInputPart?.index === 2
      ) {
        cancellation = ctx.agent.turn.cancel(0);
      }
    },
  });
  ctx = testAgent({ persistence });
  ctx.configure();

  await ctx.rpc.prompt({
    input: [
      { type: 'text', text: CAPTION },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
      { type: 'text', text: SECOND_CAPTION },
      { type: 'image_url', imageUrl: { url: IMAGE_URL } },
      { type: 'text', text: 'FINAL-RECORD-IMAGE-PROMPT' },
    ],
  });
  await ctx.untilTurnEnd();
  await cancellation;

  const cancelRecord = persistence.records.find((record) => record.type === 'turn.cancel');
  expect(cancelRecord?.type === 'turn.cancel' ? cancelRecord.cancelledTurnInputs : undefined).toBeUndefined();
  const texts = ctx.agent.context.history
    .flatMap((message) => message.content)
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('\n');
  expect(texts.match(/\/tmp\/originals\/shot\.png/g)).toHaveLength(1);
  expect(texts.match(/\/tmp\/originals\/photo\.jpg/g)).toHaveLength(1);
  expect(texts.match(/FINAL-RECORD-IMAGE-PROMPT/g)).toHaveLength(1);
  await ctx.expectResumeMatches();
});

it('undo removes the hidden caption children of every removed image prompt', async () => {
  const ctx = testAgent();
  ctx.configure();
  for (const [prompt, response] of [
    ['FIRST-IMAGE-PROMPT', 'first response'],
    ['SECOND-IMAGE-PROMPT', 'second response'],
  ] as const) {
    ctx.mockNextResponse({ type: 'text', text: response });
    await ctx.rpc.prompt({
      input: [
        { type: 'text', text: CAPTION },
        { type: 'image_url', imageUrl: { url: IMAGE_URL } },
        { type: 'text', text: prompt },
      ],
    });
    await ctx.untilTurnEnd();
  }

  ctx.agent.context.undo(2);

  expect(ctx.agent.context.history).toEqual([]);
  await ctx.expectResumeMatches();
});
