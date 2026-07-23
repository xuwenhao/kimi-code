/**
 * `sandbox` domain (L3) — effective sandbox policy resolution.
 *
 * Pure function folding the `[sandbox]` config section, the session workspace
 * roots, and host path facts into a `ResolvedSandboxPolicy`: the writable-root
 * set for the mode (`workspace-write` = workspace + additionalDirs + tmpdir +
 * `filesystem.allowWrite`; `read-only` = tmpdir + `filesystem.allowWrite`),
 * and the deny lists, all as normalized absolute paths with `~` expanded
 * against the host home directory and a trailing `/**` stripped (backends
 * mask literal paths; subtree semantics live in the matchers). `denyRead`
 * always starts from `DEFAULT_DENY_READ` (well-known credential locations,
 * plus a literal `.env` under every writable root) — the built-in list cannot
 * be turned off; `filesystem.denyRead` appends to it.
 */

import { normalize } from 'pathe';

import type { ResolvedSandboxPolicy, SandboxConfig, SandboxMode } from './sandboxTypes';

export interface SandboxWorkspaceRoots {
  readonly workDir: string;
  readonly additionalDirs: readonly string[];
}

export interface SandboxPathEnv {
  readonly tmpdir: string;
  readonly homeDir: string;
}

export const DEFAULT_DENY_READ: readonly string[] = [
  '~/.ssh',
  '~/.aws',
  '~/.gnupg',
  '~/.azure',
  '~/.config/gcloud',
  '~/.kube',
  '~/.docker',
  '~/.netrc',
  '~/.git-credentials',
  '~/.config/gh',
];

const SUBTREE_SUFFIX = '/**';

export function resolveSandboxPolicy(
  config: SandboxConfig,
  workspace: SandboxWorkspaceRoots,
  env: SandboxPathEnv,
): ResolvedSandboxPolicy {
  const mode: SandboxMode = config.mode ?? 'workspace-write';
  const expand = (p: string): string =>
    stripTrailingSlash(normalize(expandHome(stripSubtreeSuffix(p), env.homeDir)));
  const allowWrite = (config.filesystem?.allowWrite ?? []).map(expand);
  const writableRoots = dedupe(
    (mode === 'read-only'
      ? [env.tmpdir, ...allowWrite]
      : [workspace.workDir, ...workspace.additionalDirs, env.tmpdir, ...allowWrite]
    ).map(expand),
  );
  const denyRead = dedupe([
    ...DEFAULT_DENY_READ.map(expand),
    ...writableRoots.map((root) => `${root}/.env`),
    ...(config.filesystem?.denyRead ?? []).map(expand),
  ]);
  return {
    mode,
    writableRoots,
    denyRead,
    denyWrite: dedupe((config.filesystem?.denyWrite ?? []).map(expand)),
    networkEnabled: config.network?.enabled ?? false,
  };
}

function expandHome(p: string, homeDir: string): string {
  if (p === '~') return homeDir;
  if (p.startsWith('~/')) return `${homeDir}/${p.slice(2)}`;
  return p;
}

function stripSubtreeSuffix(p: string): string {
  return p.endsWith(SUBTREE_SUFFIX) ? p.slice(0, -SUBTREE_SUFFIX.length) : p;
}

function stripTrailingSlash(p: string): string {
  return p.length > 1 && p.endsWith('/') ? p.slice(0, -1) : p;
}

function dedupe(paths: readonly string[]): readonly string[] {
  return [...new Set(paths)];
}
