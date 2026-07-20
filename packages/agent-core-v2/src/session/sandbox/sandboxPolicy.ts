/**
 * `sandbox` domain (L3) — effective sandbox policy resolution.
 *
 * Pure function folding the `[sandbox]` config section, the session workspace
 * roots, and host path facts into a `ResolvedSandboxPolicy`: the writable-root
 * set for the mode (`workspace-write` = workspace + additionalDirs + tmpdir +
 * `filesystem.allowWrite`; `read-only` = tmpdir + `filesystem.allowWrite`),
 * and the deny lists, all as normalized absolute paths with `~` expanded
 * against the host home directory. `denyRead` defaults to empty — sensitive
 * files stay guarded by the `isSensitiveFile` permission policy (Phase 2).
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

export function resolveSandboxPolicy(
  config: SandboxConfig,
  workspace: SandboxWorkspaceRoots,
  env: SandboxPathEnv,
): ResolvedSandboxPolicy {
  const mode: SandboxMode = config.mode ?? 'workspace-write';
  const expand = (p: string): string => stripTrailingSlash(normalize(expandHome(p, env.homeDir)));
  const allowWrite = (config.filesystem?.allowWrite ?? []).map(expand);
  const writableRoots =
    mode === 'read-only'
      ? [env.tmpdir, ...allowWrite]
      : [workspace.workDir, ...workspace.additionalDirs, env.tmpdir, ...allowWrite];
  return {
    mode,
    writableRoots: dedupe(writableRoots.map(expand)),
    denyRead: dedupe((config.filesystem?.denyRead ?? []).map(expand)),
    denyWrite: dedupe((config.filesystem?.denyWrite ?? []).map(expand)),
    networkEnabled: config.network?.enabled ?? false,
  };
}

function expandHome(p: string, homeDir: string): string {
  if (p === '~') return homeDir;
  if (p.startsWith('~/')) return `${homeDir}/${p.slice(2)}`;
  return p;
}

function stripTrailingSlash(p: string): string {
  return p.length > 1 && p.endsWith('/') ? p.slice(0, -1) : p;
}

function dedupe(paths: readonly string[]): readonly string[] {
  return [...new Set(paths)];
}
