/**
 * `sandbox` domain (L3) — sandbox configuration and decision types.
 *
 * Pure type vocabulary for the OS-level command sandbox (bubblewrap on Linux,
 * seatbelt on macOS): the `[sandbox]` config-section shape (`SandboxConfig`),
 * the per-command verdict (`SandboxDecision`) produced by `ISandboxService`,
 * and the backend-ready `ResolvedSandboxPolicy` (absolute, `~`-expanded paths)
 * consumed by the sandbox backends. No IO, no DI.
 */

export type SandboxMode = 'workspace-write' | 'read-only';

export interface SandboxFilesystemConfig {
  readonly denyRead?: readonly string[];
  readonly allowWrite?: readonly string[];
  readonly denyWrite?: readonly string[];
}

export interface SandboxNetworkConfig {
  readonly enabled?: boolean;
  readonly allowedDomains?: readonly string[];
  readonly allowUnixSockets?: readonly string[];
}

export interface SandboxConfig {
  readonly enabled?: boolean;
  readonly mode?: SandboxMode;
  readonly require?: boolean;
  readonly autoAllowSandboxedBash?: boolean;
  readonly excludedCommands?: readonly string[];
  readonly filesystem?: SandboxFilesystemConfig;
  readonly network?: SandboxNetworkConfig;
}

export type SandboxBackendId = 'bwrap' | 'seatbelt';

export type SandboxDecision =
  | { readonly kind: 'sandboxed'; readonly argv: readonly string[]; readonly backendId: SandboxBackendId }
  | { readonly kind: 'excluded'; readonly matched: string }
  | {
      readonly kind: 'unsandboxed';
      readonly reason: 'disabled' | 'backend-unavailable' | 'unsupported-platform';
    }
  // `require = true` but no usable backend: fail-closed, the command must not run.
  | { readonly kind: 'blocked'; readonly reason: string };

export interface ResolvedSandboxPolicy {
  readonly mode: SandboxMode;
  // Absolute paths with `~` expanded. `workspace-write`: cwd + additionalDirs +
  // tmpdir + filesystem.allowWrite; `read-only`: tmpdir + filesystem.allowWrite.
  readonly writableRoots: readonly string[];
  readonly denyRead: readonly string[];
  readonly denyWrite: readonly string[];
  readonly networkEnabled: boolean;
}
