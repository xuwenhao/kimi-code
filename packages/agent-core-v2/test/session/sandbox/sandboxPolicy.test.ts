/**
 * sandboxPolicy tests for the v2 sandbox domain.
 *
 * Exercises the pure `resolveSandboxPolicy`: writable-root sets per mode,
 * `~` expansion against the host home directory, normalization and dedupe,
 * the always-on `DEFAULT_DENY_READ` list, and the literal `.env` masks under
 * every writable root.
 */

import { describe, expect, it } from 'vitest';

import { DEFAULT_DENY_READ, resolveSandboxPolicy } from '#/session/sandbox/sandboxPolicy';
import type { SandboxConfig } from '#/session/sandbox/sandboxTypes';

const ENV = { tmpdir: '/tmp', homeDir: '/home/test' } as const;
const WORKSPACE = { workDir: '/workspace/app', additionalDirs: ['/workspace/extra'] } as const;

const EXPANDED_DEFAULTS = DEFAULT_DENY_READ.map((p) => `/home/test/${p.slice(2)}`);

function expectedDenyRead(writableRoots: readonly string[], userRules: readonly string[] = []) {
  return [...EXPANDED_DEFAULTS, ...writableRoots.map((root) => `${root}/.env`), ...userRules];
}

describe('resolveSandboxPolicy', () => {
  it('defaults to workspace-write with workspace + additionalDirs + tmpdir writable', () => {
    const policy = resolveSandboxPolicy({}, WORKSPACE, ENV);

    expect(policy.mode).toBe('workspace-write');
    expect(policy.writableRoots).toEqual(['/workspace/app', '/workspace/extra', '/tmp']);
    expect(policy.denyWrite).toEqual([]);
    expect(policy.networkEnabled).toBe(false);
  });

  it('always includes the built-in deny_read list and a literal .env under each writable root', () => {
    const policy = resolveSandboxPolicy({}, WORKSPACE, ENV);

    expect(policy.denyRead).toEqual(
      expectedDenyRead(['/workspace/app', '/workspace/extra', '/tmp']),
    );
  });

  it('read-only mode keeps only tmpdir + filesystem.allowWrite writable', () => {
    const config: SandboxConfig = {
      mode: 'read-only',
      filesystem: { allowWrite: ['/data/out'] },
    };
    const policy = resolveSandboxPolicy(config, WORKSPACE, ENV);

    expect(policy.mode).toBe('read-only');
    expect(policy.writableRoots).toEqual(['/tmp', '/data/out']);
    expect(policy.denyRead).toEqual(expectedDenyRead(['/tmp', '/data/out']));
  });

  it('appends filesystem.allowWrite in workspace-write mode', () => {
    const config: SandboxConfig = { filesystem: { allowWrite: ['/data/out', '~/.cache/tool'] } };
    const policy = resolveSandboxPolicy(config, WORKSPACE, ENV);

    expect(policy.writableRoots).toEqual([
      '/workspace/app',
      '/workspace/extra',
      '/tmp',
      '/data/out',
      '/home/test/.cache/tool',
    ]);
  });

  it('expands ~ in denyRead / denyWrite / allowWrite against homeDir', () => {
    const config: SandboxConfig = {
      filesystem: {
        denyRead: ['~/.gnupg', '~/secrets.txt', '/etc/shadow'],
        denyWrite: ['~/.git', '~'],
        allowWrite: ['~'],
      },
    };
    const policy = resolveSandboxPolicy(config, WORKSPACE, ENV);

    // ~/.gnupg is already in the built-in defaults — merged without duplicates.
    expect(policy.denyRead).toEqual(
      expectedDenyRead(['/workspace/app', '/workspace/extra', '/tmp', '/home/test'], [
        '/home/test/secrets.txt',
        '/etc/shadow',
      ]),
    );
    expect(policy.denyWrite).toEqual(['/home/test/.git', '/home/test']);
    expect(policy.writableRoots).toContain('/home/test');
  });

  it('strips a trailing /** from rule entries', () => {
    const config: SandboxConfig = { filesystem: { denyRead: ['/data/secret/**'] } };
    const policy = resolveSandboxPolicy(config, WORKSPACE, ENV);

    expect(policy.denyRead).toContain('/data/secret');
    expect(policy.denyRead).not.toContain('/data/secret/**');
  });

  it('normalizes and dedupes roots', () => {
    const config: SandboxConfig = {
      filesystem: { allowWrite: ['/workspace/app/', '/data//out'] },
    };
    const policy = resolveSandboxPolicy(
      config,
      { workDir: '/workspace/app', additionalDirs: ['/workspace/app', '/workspace/extra'] },
      ENV,
    );

    expect(policy.writableRoots).toEqual(['/workspace/app', '/workspace/extra', '/tmp', '/data/out']);
    expect(policy.denyRead.filter((p) => p === '/workspace/app/.env')).toHaveLength(1);
  });

  it('reads networkEnabled from network.enabled', () => {
    expect(resolveSandboxPolicy({ network: { enabled: true } }, WORKSPACE, ENV).networkEnabled).toBe(
      true,
    );
    expect(resolveSandboxPolicy({ network: {} }, WORKSPACE, ENV).networkEnabled).toBe(false);
  });
});
