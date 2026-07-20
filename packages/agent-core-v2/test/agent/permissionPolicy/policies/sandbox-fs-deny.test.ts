import { tmpdir } from 'node:os';

import type { ToolCall } from '#/kosong/contract/message';
import { describe, expect, it } from 'vitest';

import type { IConfigService } from '#/app/config/config';
import type { ResolvedToolExecutionHookContext } from '#/agent/toolExecutor/toolHooks';
import { SandboxFsDenyPermissionPolicyService } from '#/agent/permissionPolicy/policies/sandbox-fs-deny';
import type { IHostEnvironment } from '#/os/interface/hostEnvironment';
import type { SandboxConfig } from '#/session/sandbox/sandboxTypes';
import { ToolAccesses, type ToolAccesses as ToolAccessList } from '#/tool/toolContract';

import { stubWorkspaceContext } from '../../../session/workspaceContext/stub-workspace-context';

const signal = new AbortController().signal;

function stubConfig(section: SandboxConfig | undefined): IConfigService {
  return {
    _serviceBrand: undefined,
    get: (domain: string) => (domain === 'sandbox' ? section : undefined),
  } as unknown as IConfigService;
}

function stubEnv(): IHostEnvironment {
  return {
    _serviceBrand: undefined,
    osKind: 'Linux',
    osArch: 'x86_64',
    osVersion: 'test',
    shellName: 'bash',
    shellPath: '/bin/bash',
    pathClass: 'posix',
    homeDir: '/home/test',
    ready: Promise.resolve(),
  };
}

function policy(section: SandboxConfig | undefined): SandboxFsDenyPermissionPolicyService {
  return new SandboxFsDenyPermissionPolicyService(
    stubConfig(section),
    stubWorkspaceContext('/workspace/app', ['/workspace/extra']),
    stubEnv(),
  );
}

function policyContext(accesses: ToolAccessList): ResolvedToolExecutionHookContext {
  const call: ToolCall = {
    type: 'function',
    id: 'call_tool',
    name: 'Read',
    arguments: '{}',
  };
  return {
    turnId: 0,
    signal,
    toolCall: call,
    toolCalls: [call],
    args: {},
    execution: {
      approvalRule: 'Read',
      accesses,
      execute: async () => ({ output: '' }),
    },
  };
}

describe('SandboxFsDenyPermissionPolicyService', () => {
  it('denies reads under a deny_read subtree rule, including /foo-evil escape resistance', () => {
    const p = policy({ enabled: true, filesystem: { denyRead: ['~/.ssh/**'] } });

    expect(
      p.evaluate(policyContext(ToolAccesses.readFile('/home/test/.ssh/config'))),
    ).toMatchObject({ kind: 'deny' });
    expect(
      p.evaluate(policyContext(ToolAccesses.readFile('/home/test/.ssh2/config'))),
    ).toBeUndefined();
  });

  it('denies searches under a deny_read rule', () => {
    const p = policy({ enabled: true, filesystem: { denyRead: ['/home/test/.gnupg'] } });

    expect(
      p.evaluate(policyContext(ToolAccesses.searchTree('/home/test/.gnupg/keyring'))),
    ).toMatchObject({ kind: 'deny' });
    expect(
      p.evaluate(policyContext(ToolAccesses.searchTree('/home/test/other'))),
    ).toBeUndefined();
  });

  it('treats a deny_read entry without /** as an exact file (no suffix leakage)', () => {
    const p = policy({ enabled: true, filesystem: { denyRead: ['/home/test/secrets.txt'] } });

    expect(
      p.evaluate(policyContext(ToolAccesses.readFile('/home/test/secrets.txt'))),
    ).toMatchObject({ kind: 'deny' });
    expect(
      p.evaluate(policyContext(ToolAccesses.readFile('/home/test/secrets.txt.bak'))),
    ).toBeUndefined();
  });

  it('denies readwrite accesses against deny_read (readwrite reads too)', () => {
    const p = policy({ enabled: true, filesystem: { denyRead: ['/home/test/secrets.txt'] } });

    expect(
      p.evaluate(policyContext(ToolAccesses.readWriteFile('/home/test/secrets.txt'))),
    ).toMatchObject({ kind: 'deny' });
  });

  it('denies writes under a deny_write rule', () => {
    const p = policy({ enabled: true, filesystem: { denyWrite: ['/workspace/app/.git/**'] } });

    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile('/workspace/app/.git/config'))),
    ).toMatchObject({ kind: 'deny' });
    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile('/workspace/app/.gitx/config'))),
    ).toBeUndefined();
    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile('/workspace/app/src/a.ts'))),
    ).toBeUndefined();
  });

  it('denies writes outside the writable roots in read-only mode', () => {
    const p = policy({ enabled: true, mode: 'read-only' });

    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile('/workspace/app/src/a.ts'))),
    ).toMatchObject({ kind: 'deny', reason: { sandbox_mode: 'read-only' } });
    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile(`${tmpdir()}/scratch.txt`))),
    ).toBeUndefined();
    expect(
      p.evaluate(policyContext(ToolAccesses.readFile('/workspace/app/src/a.ts'))),
    ).toBeUndefined();
  });

  it('allows writes inside the workspace in workspace-write mode', () => {
    const p = policy({ enabled: true });

    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile('/workspace/app/src/a.ts'))),
    ).toBeUndefined();
    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile('/workspace/extra/b.ts'))),
    ).toBeUndefined();
  });

  it('denies sensitive files for any operation (ask upgraded to deny)', () => {
    const p = policy({ enabled: true });

    expect(
      p.evaluate(policyContext(ToolAccesses.readFile('/home/test/.ssh/id_rsa'))),
    ).toMatchObject({ kind: 'deny' });
    expect(
      p.evaluate(policyContext(ToolAccesses.writeFile('/workspace/app/.env'))),
    ).toMatchObject({ kind: 'deny' });
    expect(
      p.evaluate(policyContext(ToolAccesses.readFile('/workspace/app/.env.example'))),
    ).toBeUndefined();
  });

  it('defers when the sandbox is disabled or there are no file accesses', () => {
    expect(
      policy({ enabled: false }).evaluate(
        policyContext(ToolAccesses.readFile('/home/test/.ssh/id_rsa')),
      ),
    ).toBeUndefined();
    expect(
      policy(undefined).evaluate(policyContext(ToolAccesses.readFile('/home/test/.ssh/id_rsa'))),
    ).toBeUndefined();
    expect(policy({ enabled: true }).evaluate(policyContext(ToolAccesses.none()))).toBeUndefined();
  });
});
