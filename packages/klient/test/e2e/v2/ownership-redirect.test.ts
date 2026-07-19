/**
 * Klient multi-instance redirect e2e (phase 3, design §2.1): a klient pointed
 * at the WRONG instance follows the `session.held_by_peer` (40921) `routable`
 * answer onto the holder origin — and every later call lands on the holder.
 *
 * In-process `startServerPair` (shared home, `multi_server` flag, port 0).
 * The session is created through `/api/v1` on instance A, so A owns the write
 * lease registered with its own address; klient drives `/api/v2` starting
 * from B. `SessionFacade.restore()` routes to
 * `sessionLifecycleService.restore` → `resume`, the v2 open-session entry
 * that materializes the session and refuses peer-held ones; scoped reads
 * only `get()` and would answer 40401 on the wrong instance (dispatcher
 * materialization is a separate workflow).
 */
import { describe, expect, it } from 'vitest';

import { createKlient } from '../../../src/transports/http/index.js';
import type { SessionRedirectInfo } from '../../../src/sessionRedirect.js';
import { startServerPair } from '../harness/testing/index.js';
import { createCaseLogger } from '../legacy/log.js';

describe('klient: session-ownership redirect onto the holder instance', () => {
  it(
    'opening a peer-held session from the wrong instance follows routable 40921 onto the holder',
    { timeout: 60_000 },
    async () => {
      const log = createCaseLogger('klient-ownership-redirect/follow-routable');
      const pair = await startServerPair();
      try {
        const created = await pair
          .connectClient(pair.a)
          .createSession({ metadata: { cwd: pair.cwd } });
        const sessionId = created.id;
        log('session created on A (A holds the lease)', {
          sessionId,
          urlA: pair.urlA,
          urlB: pair.urlB,
        });

        const redirects: SessionRedirectInfo[] = [];
        const klient = createKlient({
          url: pair.urlB,
          onSessionRedirect: (info) => redirects.push(info),
        });

        // Open from the WRONG instance: B refuses 40921 routable(address=urlA);
        // the transport switches origin and re-sends the same call.
        const restored = await klient.session(sessionId).restore();
        log('restore via klient after redirect', { restored, redirects });
        expect(restored).toBe(true);
        expect(redirects).toEqual([{ from: pair.urlB, to: pair.urlA, follow: 1 }]);
        expect(klient.currentUrl).toBe(pair.urlA);

        // Every later request — session facade, global facade — now lands on
        // the holder (B's dispatcher would answer 40401 for either call).
        const meta = await klient.session(sessionId).get();
        const page = await klient.global.sessions.list({ limit: 20 });
        log('post-redirect reads land on the holder', {
          currentUrl: klient.currentUrl,
          meta: meta.id,
          listed: page.items.map((item) => item.id),
        });
        expect(meta.id).toBe(sessionId);
        expect(page.items.some((item) => item.id === sessionId)).toBe(true);
        await klient.close();
      } finally {
        await pair.dispose();
      }
    },
  );
});
