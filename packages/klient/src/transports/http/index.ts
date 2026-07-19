/**
 * `createKlient` over HTTP(S) against kap-server's `/api/v2` surface. Event
 * subscriptions transparently ride a lazily opened WebSocket — after
 * initialization the klient behaves exactly like its ipc/memory siblings.
 * Browser-safe: only `fetch` + an injectable `WebSocket` are required.
 *
 * Multi-instance shared home: when a call is refused with envelope code 40921
 * (`session.held_by_peer`) because a sibling instance holds the session's
 * write lease, the transport follows the holder address automatically
 * (`SessionRedirectChannel`); `currentUrl` tracks the live origin and
 * `onSessionRedirect` fires once per followed redirect.
 */

import { createKlientFromChannel, type Klient, type KlientOptions } from '../../core/klient.js';
import {
  KlientConnection,
  SessionRedirectChannel,
  type SessionRedirectInfo,
  type SessionRedirectOptions,
} from '../../sessionRedirect.js';
import type { HttpChannelOptions } from './channel.js';

export interface HttpKlientOptions
  extends KlientOptions,
    HttpChannelOptions,
    SessionRedirectOptions {
  /** Fires once per followed session-ownership redirect (re-subscribe events here). */
  readonly onSessionRedirect?: (info: SessionRedirectInfo) => void;
}

export interface HttpKlient extends Klient {
  /** Origin every later call targets; changes after a followed redirect. */
  readonly currentUrl: string;
}

export function createKlient(options: HttpKlientOptions): HttpKlient {
  const connection = new KlientConnection(options);
  if (options.onSessionRedirect !== undefined) {
    connection.onRedirect(options.onSessionRedirect);
  }
  const channel = new SessionRedirectChannel({
    connection,
    token: options.token,
    fetch: options.fetch,
    WebSocketImpl: options.WebSocketImpl,
  });
  const klient = createKlientFromChannel(channel, options);
  return {
    ...klient,
    get currentUrl() {
      return connection.currentUrl;
    },
  };
}
