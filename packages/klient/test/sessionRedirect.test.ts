/**
 * Unit coverage for the multi-instance session-ownership follow/retry loop
 * (`KlientConnection` + `SessionRedirectChannel`): routable redirects rebase
 * the shared origin and re-send, `creating` retries stay on the same
 * instance, terminal phases surface structured handoff errors, and unknown
 * payload shapes pass through untouched (forward compatibility).
 */

import { describe, expect, it, vi } from 'vitest';

import { RPCError } from '../src/core/errors.js';
import type { WsLike, WsLikeCtor } from '../src/transports/ws/wsSocket.js';
import {
  KlientConnection,
  readSessionOwnershipDetails,
  SessionRedirectChannel,
  type SessionRedirectInfo,
  type SessionRedirectOptions,
} from '../src/sessionRedirect.js';

const PEER = 'http://127.0.0.1:60002';
const HOLDER = 'http://127.0.0.1:60001';
const OWNED_MSG = 'session s1 is held by another instance (routable)';

const tick = (ms = 0): Promise<void> => new Promise((resolve) => setTimeout(resolve, ms));

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { 'content-type': 'application/json' },
  });
}

const okEnvelope = (data: unknown): Response =>
  jsonResponse({ code: 0, msg: 'ok', data, request_id: 'r' });
const heldByPeer = (details: unknown, msg = OWNED_MSG): Response =>
  jsonResponse({ code: 40921, msg, data: null, request_id: 'r', details });

function urls(fetchMock: ReturnType<typeof vi.fn<typeof fetch>>): string[] {
  return fetchMock.mock.calls.map((call) => call[0] as string);
}

interface Rig {
  connection: KlientConnection;
  channel: SessionRedirectChannel;
  redirects: SessionRedirectInfo[];
}

function rig(
  fetchMock: typeof fetch,
  policy: SessionRedirectOptions & { token?: string; WebSocketImpl?: WsLikeCtor } = {},
): Rig {
  const { token, WebSocketImpl, ...redirect } = policy;
  const connection = new KlientConnection({ url: PEER, ...redirect });
  const channel = new SessionRedirectChannel({
    connection,
    token,
    fetch: fetchMock,
    WebSocketImpl,
  });
  const redirects: SessionRedirectInfo[] = [];
  connection.onRedirect((info) => redirects.push(info));
  return { connection, channel, redirects };
}

const readS1 = (channel: SessionRedirectChannel): Promise<unknown> =>
  channel.call({ sessionId: 's1' }, 'sessionMetadata', 'read', []);

describe('session ownership redirect (SESSION_HELD_BY_PEER)', () => {
  it('follows a routable redirect: rebases onto the holder, re-sends the call, emits the signal', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'routable', address: `${HOLDER}/` }),
      )
      .mockResolvedValueOnce(okEnvelope({ id: 's1' }));
    const { connection, channel, redirects } = rig(fetchMock);

    const data = await readS1(channel);

    expect(data).toEqual({ id: 's1' });
    expect(urls(fetchMock)).toEqual([
      `${PEER}/api/v2/session/s1/sessionMetadata/read`,
      // address normalizes to its bare origin (trailing slash dropped)
      `${HOLDER}/api/v2/session/s1/sessionMetadata/read`,
    ]);
    expect(connection.currentUrl).toBe(HOLDER);
    expect(redirects).toEqual([{ from: PEER, to: HOLDER, follow: 1 }]);
  });

  it('lands every later request (core, session, and agent scopes) on the holder', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'routable', address: HOLDER }),
      )
      // a fresh Response per call — the body is consumed on every read
      .mockImplementation(() => Promise.resolve(okEnvelope(null)));
    const { channel } = rig(fetchMock);

    await readS1(channel);
    await channel.call({}, 'sessionIndex', 'list', [{}]);
    await readS1(channel);
    await channel.call({ sessionId: 's1', agentId: 'a1' }, 'sessionMetadata', 'read', []);

    expect(urls(fetchMock)).toEqual([
      `${PEER}/api/v2/session/s1/sessionMetadata/read`,
      `${HOLDER}/api/v2/session/s1/sessionMetadata/read`,
      `${HOLDER}/api/v2/sessionIndex/list`,
      `${HOLDER}/api/v2/session/s1/sessionMetadata/read`,
      `${HOLDER}/api/v2/session/s1/agent/a1/sessionMetadata/read`,
    ]);
  });

  it('keeps the bearer token and reuses it against the holder origin', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'routable', address: HOLDER }),
      )
      .mockResolvedValueOnce(okEnvelope(null));
    const { channel } = rig(fetchMock, { token: 'tok' });

    await readS1(channel);

    for (const call of fetchMock.mock.calls) {
      expect((call[1]?.headers as Record<string, string>)['authorization']).toBe('Bearer tok');
    }
  });

  it('stops after the redirect limit (anti-loop) with a structured error', async () => {
    const details = { kind: 'held-by-peer', phase: 'routable', address: HOLDER };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementation(() => Promise.resolve(heldByPeer(details)));
    const { connection, channel } = rig(fetchMock);

    const failure = await readS1(channel).then(
      () => ({ error: undefined as unknown }),
      (error: unknown) => ({ error }),
    );

    // default maxRedirects = 1: one follow, then the loop guard fires.
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(connection.currentUrl).toBe(HOLDER);
    expect(failure.error).toMatchObject({ name: 'RPCError', code: 40921, details });
    expect((failure.error as Error).message).toContain(OWNED_MSG);
    expect((failure.error as Error).message).toMatch(/redirect loop/);
  });

  it('honors a raised maxRedirects before the loop guard fires (ping-pong redirect)', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      // holder ping-pongs the client back and forth: the loop guard, not the
      // self-loop terminal, must fire after maxRedirects follows
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'routable', address: HOLDER }),
      )
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'routable', address: PEER }),
      )
      .mockImplementation(() =>
        Promise.resolve(heldByPeer({ kind: 'held-by-peer', phase: 'routable', address: HOLDER })),
      );
    const { channel } = rig(fetchMock, { maxRedirects: 2 });

    await expect(readS1(channel)).rejects.toMatchObject({
      code: 40921,
      message: expect.stringMatching(/redirect loop/) as unknown as string,
    });
    expect(fetchMock).toHaveBeenCalledTimes(3);
  });

  it('retries creating on the SAME instance per retry_after_ms, honoring the fallback delay', async () => {
    const delays: number[] = [];
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'creating', retry_after_ms: 120 }),
      )
      .mockResolvedValueOnce(heldByPeer({ kind: 'held-by-peer', phase: 'creating' }))
      .mockResolvedValueOnce(okEnvelope({ id: 's1' }));
    const { connection, channel } = rig(fetchMock, {
      sleep: (ms) => {
        delays.push(ms);
        return Promise.resolve();
      },
    });

    await expect(readS1(channel)).resolves.toEqual({ id: 's1' });
    expect(delays).toEqual([120, 500]);
    expect(urls(fetchMock)).toEqual([
      `${PEER}/api/v2/session/s1/sessionMetadata/read`,
      `${PEER}/api/v2/session/s1/sessionMetadata/read`,
      `${PEER}/api/v2/session/s1/sessionMetadata/read`,
    ]);
    expect(connection.currentUrl).toBe(PEER);
  });

  it('gives up after maxCreatingRetries creating answers with a structured error', async () => {
    const details = { kind: 'held-by-peer', phase: 'creating', retry_after_ms: 10 };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementation(() => Promise.resolve(heldByPeer(details)));
    const { channel } = rig(fetchMock, { sleep: () => Promise.resolve() });

    await expect(readS1(channel)).rejects.toMatchObject({
      code: 40921,
      details,
      message: expect.stringMatching(/mid-creation/) as unknown as string,
    });
    expect(fetchMock).toHaveBeenCalledTimes(4); // 1 initial + maxCreatingRetries(3) retries
  });

  it.each([
    {
      name: 'holder-unresponsive',
      details: { kind: 'held-by-peer', phase: 'holder-unresponsive', retry_after_ms: 2000 },
      message: /not responding.*2000ms.*force-unlock/s,
    },
    {
      name: 'held-by-local-instance',
      details: { kind: 'held-by-peer', phase: 'held-by-local-instance' },
      message: /without a network address.*force-unlock/s,
    },
    {
      name: 'unregistered-writer',
      details: { kind: 'unregistered-writer' },
      message: /unregistered writer/,
    },
  ])(
    'throws a terminal structured error for $name (no retry, no follow)',
    async ({ details, message }) => {
      const fetchMock = vi
        .fn<typeof fetch>()
        .mockImplementation(() => Promise.resolve(heldByPeer(details)));
      const { connection, channel } = rig(fetchMock, { sleep: () => Promise.resolve() });

      await expect(readS1(channel)).rejects.toMatchObject({
        name: 'RPCError',
        code: 40921,
        details,
        message: expect.stringMatching(message) as unknown as string,
      });
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(connection.currentUrl).toBe(PEER);
    },
  );

  it('surfaces the raw refusal (plus holder guidance) when follow is disabled', async () => {
    const details = { kind: 'held-by-peer', phase: 'routable', address: HOLDER };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementation(() => Promise.resolve(heldByPeer(details)));
    const { connection, channel, redirects } = rig(fetchMock, { follow: false });

    await expect(readS1(channel)).rejects.toMatchObject({
      code: 40921,
      details,
      message: expect.stringMatching(new RegExp(HOLDER.replace(/[.:]/g, '\\$&'))) as unknown as string,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(redirects).toEqual([]);
    expect(connection.currentUrl).toBe(PEER);
  });

  it('treats a routable address pointing back at this instance as terminal (self-loop)', async () => {
    const details = { kind: 'held-by-peer', phase: 'routable', address: `${PEER}/` };
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockImplementation(() => Promise.resolve(heldByPeer(details)));
    const { channel, redirects } = rig(fetchMock);

    await expect(readS1(channel)).rejects.toMatchObject({
      code: 40921,
      details,
      message: expect.stringMatching(/this very instance/) as unknown as string,
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(redirects).toEqual([]);
  });

  it('rethrows unknown ownership payload shapes untouched (forward compatibility)', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'future-phase', address: HOLDER }),
      );
    const { channel } = rig(fetchMock);

    await expect(readS1(channel)).rejects.toMatchObject({
      code: 40921,
      message: OWNED_MSG, // no guidance appended
      details: { kind: 'held-by-peer', phase: 'future-phase', address: HOLDER },
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('does not touch non-40921 RPC errors', async () => {
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        jsonResponse({ code: 40401, msg: 'session not found', data: null, request_id: 'r' }),
      );
    const { channel } = rig(fetchMock);

    await expect(readS1(channel)).rejects.toMatchObject({
      code: 40401,
      message: 'session not found',
    });
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('closes the stale event bridge on redirect; later listens target the holder origin', async () => {
    const server = new FakeServer();
    const fetchMock = vi
      .fn<typeof fetch>()
      .mockResolvedValueOnce(
        heldByPeer({ kind: 'held-by-peer', phase: 'routable', address: HOLDER }),
      )
      .mockResolvedValueOnce(okEnvelope({ id: 's1' }));
    const { channel } = rig(fetchMock, { WebSocketImpl: fakeCtor(server) });

    channel.listen({}, { kind: 'stream', name: 'events' }, () => undefined);
    await tick(5);
    expect(server.lastUrl).toContain('127.0.0.1:60002');
    expect(server.sockets).toHaveLength(1);

    await readS1(channel);

    // The old bridge is dropped — its streams would keep talking to the wrong
    // instance; the next listen lazily re-creates against the holder origin.
    expect(server.sockets[0]!.readyState).toBe(3);

    const second = channel.listen({}, { kind: 'stream', name: 'events' }, () => undefined);
    await tick(5);
    expect(server.lastUrl).toContain('127.0.0.1:60001');
    expect(server.sockets).toHaveLength(2);
    second.dispose();
    await channel.close();
  });
});

describe('readSessionOwnershipDetails', () => {
  it('parses the wire payload shapes and rejects everything else', () => {
    expect(
      readSessionOwnershipDetails(
        new RPCError(40921, 'x', { kind: 'held-by-peer', phase: 'routable', address: 'h' }),
      ),
    ).toEqual({
      kind: 'held-by-peer',
      phase: 'routable',
      address: 'h',
      retry_after_ms: undefined,
    });
    expect(
      readSessionOwnershipDetails(new RPCError(40921, 'x', { kind: 'unregistered-writer' })),
    ).toEqual({ kind: 'unregistered-writer' });
    // unknown phase → undefined (forward compat: payload passes through untouched)
    expect(
      readSessionOwnershipDetails(
        new RPCError(40921, 'x', { kind: 'held-by-peer', phase: 'future-phase' }),
      ),
    ).toBeUndefined();
    expect(readSessionOwnershipDetails(new RPCError(40401, 'x'))).toBeUndefined();
    expect(readSessionOwnershipDetails(new Error('boom'))).toBeUndefined();
    // garbage fields are dropped rather than trusted
    expect(
      readSessionOwnershipDetails(
        new RPCError(40921, 'x', {
          kind: 'held-by-peer',
          phase: 'creating',
          retry_after_ms: -5,
        }),
      ),
    ).toEqual({
      kind: 'held-by-peer',
      phase: 'creating',
      address: undefined,
      retry_after_ms: undefined,
    });
  });
});

// --- WS fake (event-bridge level, mirrors wsSocket.test.ts helpers) ----------

type Listener = (event: never) => void;

class FakeServer {
  readonly frames: Record<string, unknown>[] = [];
  readonly sockets: FakeClientSocket[] = [];
  lastUrl = '';

  attach(socket: FakeClientSocket): void {
    this.sockets.push(socket);
    queueMicrotask(() => {
      socket.readyState = FakeClientSocket.OPEN;
      socket.fire('open');
      this.deliver(socket, { type: 'ready', heartbeatMs: 30_000 });
    });
  }

  receive(raw: string): void {
    const frame = JSON.parse(raw) as Record<string, unknown>;
    this.frames.push(frame);
    if (frame['type'] === 'listen') {
      const socket = this.sockets.at(-1);
      if (socket !== undefined) this.deliver(socket, { type: 'listen_result', id: frame['id'] });
    }
  }

  private deliver(socket: FakeClientSocket, frame: Record<string, unknown>): void {
    socket.deliver(frame);
  }
}

class FakeClientSocket implements WsLike {
  static readonly OPEN = 1;
  readyState = 0;
  private readonly handlers = new Map<string, Set<Listener>>();

  constructor(
    private readonly server: FakeServer,
    url: string,
  ) {
    server.lastUrl = url;
    server.attach(this);
  }

  addEventListener(type: string, listener: Listener): void {
    const set = this.handlers.get(type) ?? new Set<Listener>();
    set.add(listener);
    this.handlers.set(type, set);
  }

  send(data: string): void {
    this.server.receive(data);
  }

  close(): void {
    this.readyState = 3;
    this.fire('close');
  }

  fire(type: string): void {
    for (const handler of this.handlers.get(type) ?? []) handler(undefined as never);
  }

  deliver(frame: Record<string, unknown>): void {
    queueMicrotask(() => {
      for (const handler of this.handlers.get('message') ?? []) {
        handler({ data: JSON.stringify(frame) } as never);
      }
    });
  }
}

function fakeCtor(server: FakeServer): WsLikeCtor {
  class BoundFakeSocket extends FakeClientSocket {
    constructor(url: string) {
      super(server, url);
    }
  }
  return BoundFakeSocket as unknown as WsLikeCtor;
}
