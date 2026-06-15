/// <reference types="vitest/config" />
import { execFileSync } from 'node:child_process';

import { defineConfig } from 'vite';
import vue from '@vitejs/plugin-vue';
import tailwindcss from '@tailwindcss/vite';
import pkg from './package.json';

const webPort = Number(process.env.WEB_PORT) || 5175;
// Where the dev proxy forwards server traffic. Defaults to the local server
// (or `pnpm dev:stub`). Override to point dev at another server instance.
const serverTarget = process.env.KIMI_SERVER_URL || 'http://127.0.0.1:7878';
const webCommit = resolveWebCommit();

function shortCommit(value: string | undefined): string {
  const commit = value?.trim() ?? '';
  return commit.length > 7 ? commit.slice(0, 7) : commit;
}

function resolveWebCommit(): string {
  const fromEnv = shortCommit(process.env.KIMI_CODE_COMMIT ?? process.env.GITHUB_SHA);
  if (fromEnv.length > 0) return fromEnv;

  try {
    return shortCommit(execFileSync('git', ['rev-parse', 'HEAD'], {
      cwd: import.meta.dirname,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
    }));
  } catch {
    return '';
  }
}

export default defineConfig({
  plugins: [vue(), tailwindcss()],
  // Expose the dev proxy's upstream server target to the client so the UI can
  // show which server it is connected to (the browser otherwise only sees its
  // own same-origin URL). Unused by the same-origin production build.
  define: {
    __KIMI_DEV_PROXY_TARGET__: JSON.stringify(serverTarget),
    __KIMI_WEB_VERSION__: JSON.stringify(`v${pkg.version}`),
    __KIMI_WEB_COMMIT__: JSON.stringify(webCommit),
  },
  server: {
    port: webPort,
    strictPort: false,
    // Same-origin dev: the browser calls Vite, Vite forwards to the server.
    // No CORS anywhere. The real server serves REST + WS all under /api/v1.
    proxy: {
      '/api/v1': { target: serverTarget, changeOrigin: true, ws: true },
    },
  },
  // `vite preview` (the production build served locally) needs the same proxy —
  // bugs that only exist in production chunking (e.g. optional-peer-dep stubs)
  // can't be reproduced without running the built app against a server.
  preview: {
    port: Number(process.env.WEB_PREVIEW_PORT) || 4175,
    proxy: {
      '/api/v1': { target: serverTarget, changeOrigin: true, ws: true },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
    target: 'es2022',
  },
  test: {
    environment: 'jsdom',
    environmentOptions: {
      jsdom: {
        url: 'http://localhost/',
      },
    },
    globals: true,
    setupFiles: ['./test/setup.ts'],
  },
});
