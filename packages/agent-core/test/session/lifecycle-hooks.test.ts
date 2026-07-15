/**
 * Session lifecycle and hook contracts through real session/agent workers.
 * Filesystem, process, and provider behavior are controlled test boundaries.
 * Run: pnpm --dir packages/agent-core exec vitest run test/session/lifecycle-hooks.test.ts
 */

import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'pathe';
import { Readable } from 'node:stream';
import type { Writable } from 'node:stream';

import type { KaosProcess } from '@moonshot-ai/kaos';

import { testKaos } from '../fixtures/test-kaos';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { SDKSessionRPC } from '../../src/rpc';
import { Agent } from '../../src/agent';
import { Session } from '../../src/session';
import { SessionAPIImpl } from '../../src/session/rpc';
import { ProcessBackgroundTask } from '../../src/agent/background';
import { agentTask } from '../agent/background/helpers';


const tempDirs: string[] = [];

afterEach(async () => {
  vi.unstubAllEnvs();
  for (const dir of tempDirs.splice(0)) {
    await rm(dir, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
  }
});

describe('Session lifecycle hooks', () => {
  it('fires SessionStart on startup and SessionEnd on close', async () => {
    const { command, logPath, sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-123',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      hooks: [
        { event: 'SessionStart', matcher: 'startup', command, timeout: 5 },
        { event: 'SessionEnd', matcher: 'exit', command, timeout: 5 },
      ],
    });

    await session.createMain();
    await session.close();

    expect(await readHookPayloads(logPath)).toMatchObject([
      {
        hook_event_name: 'SessionStart',
        session_id: 'session-123',
        cwd: workDir,
        source: 'startup',
      },
      {
        hook_event_name: 'SessionEnd',
        session_id: 'session-123',
        cwd: workDir,
        reason: 'exit',
      },
    ]);
  });

  it('fires SessionStart with resume source after loading metadata', async () => {
    const { command, logPath, sessionDir, workDir } = await hookFixture();
    await writeFile(
      join(sessionDir, 'state.json'),
      JSON.stringify({
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-01T00:00:00.000Z',
        title: 'Resumed Session',
        isCustomTitle: false,
        agents: {},
        custom: {},
      }),
      'utf-8',
    );
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-456',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      hooks: [{ event: 'SessionStart', matcher: 'resume', command, timeout: 5 }],
    });

    await session.resume();

    expect(await readHookPayloads(logPath)).toMatchObject([
      {
        hook_event_name: 'SessionStart',
        session_id: 'session-456',
        cwd: workDir,
        source: 'resume',
      },
    ]);
  });

  it('does not let failing SessionStart or SessionEnd hook commands interrupt startup or close', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-reject',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      hooks: [
        { event: 'SessionStart', matcher: 'startup', command: 'exit 1', timeout: 5 },
        { event: 'SessionEnd', matcher: 'exit', command: 'exit 1', timeout: 5 },
      ],
    });

    await expect(session.createMain()).resolves.toBeDefined();
    await expect(session.close()).resolves.toBeUndefined();
  });

  it('does not let a synchronous metadata-event observer reject a prompt', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const observerError = new Error('metadata observer failed synchronously');
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>((event): Promise<void> => {
      if (event.type === 'session.meta.updated') throw observerError;
      return Promise.resolve();
    });
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-meta-sync-observer',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    await session.createMain();
    const warn = vi.spyOn(session.log, 'warn').mockImplementation(() => {});

    await expect(
      new SessionAPIImpl(session).prompt({
        agentId: 'main',
        input: [{ type: 'text', text: 'prompt survives sync observer failure' }],
      }),
    ).resolves.toMatchObject({ kind: 'started' });
    expect(warn).toHaveBeenCalledWith('session event delivery failed', {
      eventType: 'session.meta.updated',
      error: observerError,
    });

    await session.close();
  });

  it('does not let an asynchronously rejected metadata event reject a prompt', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const observerError = new Error('metadata observer rejected asynchronously');
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>((event) =>
      event.type === 'session.meta.updated' ? Promise.reject(observerError) : Promise.resolve(),
    );
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-meta-async-observer',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    await session.createMain();
    const warn = vi.spyOn(session.log, 'warn').mockImplementation(() => {});

    await expect(
      new SessionAPIImpl(session).prompt({
        agentId: 'main',
        input: [{ type: 'text', text: 'prompt survives async observer failure' }],
      }),
    ).resolves.toMatchObject({ kind: 'started' });
    await vi.waitFor(() => {
      expect(warn).toHaveBeenCalledWith('session event delivery failed', {
        eventType: 'session.meta.updated',
        error: observerError,
      });
    });

    await session.close();
  });

  it('isolates a synchronous MCP status-event observer failure', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const observerError = new Error('MCP status observer failed synchronously');
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>((event): Promise<void> => {
      if (event.type === 'mcp.server.status') throw observerError;
      return Promise.resolve();
    });
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-mcp-sync-observer',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const warn = vi.spyOn(session.log, 'warn').mockImplementation(() => {});
    const emitters = session as unknown as {
      onMcpServerStatusChange(entry: {
        name: string;
        transport: 'stdio';
        status: 'failed';
        toolCount: number;
        error: string;
      }): void;
    };

    expect(() => {
      emitters.onMcpServerStatusChange({
        name: 'example',
        transport: 'stdio',
        status: 'failed',
        toolCount: 0,
        error: 'connection failed',
      });
    }).not.toThrow();
    expect(warn).toHaveBeenCalledWith('session event delivery failed', {
      eventType: 'mcp.server.status',
      error: observerError,
    });

    await session.close();
  });

  it('consumes an asynchronously rejected initial MCP error event', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const observerError = new Error('MCP error observer rejected asynchronously');
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>((event) =>
      event.type === 'error' ? Promise.reject(observerError) : Promise.resolve(),
    );
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-mcp-async-observer',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const warn = vi.spyOn(session.log, 'warn').mockImplementation(() => {});
    const emitters = session as unknown as {
      emitInitialMcpLoadError(error: unknown): void;
    };

    emitters.emitInitialMcpLoadError(new Error('initial MCP load failed'));
    await vi.waitFor(() => {
      expect(warn).toHaveBeenCalledWith('session event delivery failed', {
        eventType: 'error',
        error: observerError,
      });
    });

    await session.close();
  });

  it('stops background tasks on close by default', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-bg-cleanup',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();
    const { proc, killSpy } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'exit cleanup'),
    );

    await session.close();

    expect(killSpy).toHaveBeenCalledWith('SIGTERM');
    expect(agent.background.getTask(taskId)?.status).toBe('killed');
  });

  it('does not steer background task notifications while closing the session', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-bg-cleanup-no-steer',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();
    const steerSpy = vi.spyOn(agent.turn, 'steer');
    const { proc, killSpy } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'exit cleanup without steer'),
    );

    await session.close();
    await new Promise((resolve) => setImmediate(resolve));

    expect(killSpy).toHaveBeenCalledWith('SIGTERM');
    expect(agent.background.getTask(taskId)?.status).toBe('killed');
    expect(steerSpy).not.toHaveBeenCalled();
  });

  it('keeps background tasks alive on close when keepAliveOnExit is true', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-bg-keepalive',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true },
    });
    const agent = await session.createMain();
    const { proc, killSpy } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'keep alive'),
    );

    await session.close();

    expect(killSpy).not.toHaveBeenCalled();
    expect(agent.background.getTask(taskId)?.status).toBe('running');
  });

  it('keeps background agent turns alive on close when keepAliveOnExit is true', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-bg-agent-keepalive',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true },
    });
    const main = await session.createMain();
    const { id: childId, agent: child } = await session.createAgent(
      { type: 'sub' },
      { parentAgentId: 'main' },
    );
    const turnSettled = createDeferred<void>();
    const waitSpy = vi
      .spyOn(child.turn, 'waitForCurrentTurn')
      .mockImplementation(() => turnSettled.promise as never);
    const cancelSpy = vi.spyOn(child.turn, 'cancel').mockImplementation(() => {
      turnSettled.resolve();
      return Promise.resolve();
    });
    vi.spyOn(child.turn, 'hasActiveTurn', 'get').mockReturnValue(true);
    const abortController = new AbortController();
    const abort = vi.spyOn(abortController, 'abort');
    const taskId = main.background.registerTask(
      agentTask(new Promise(() => {}), 'keep background agent alive', {
        abortController,
        agentId: childId,
        subagentType: 'coder',
      }),
    );

    await session.close();

    expect(cancelSpy).not.toHaveBeenCalled();
    expect(waitSpy).not.toHaveBeenCalled();
    expect(abort).not.toHaveBeenCalled();
    expect(main.background.getTask(taskId)?.status).toBe('running');
  });

  it('waitForBackgroundTasksOnPrint returns immediately when keepAliveOnExit is false', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-wait-disabled',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: false },
    });
    const agent = await session.createMain();
    const { proc, killSpy } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'no wait'),
    );

    await session.waitForBackgroundTasksOnPrint();

    expect(killSpy).not.toHaveBeenCalled();
    expect(agent.background.getTask(taskId)?.status).toBe('running');
    await session.close();
  });

  it('waitForBackgroundTasksOnPrint waits for background tasks to finish when keepAliveOnExit is true', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-wait',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true },
    });
    const agent = await session.createMain();
    const { proc } = pendingProcess(0);
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'wait for me'),
    );

    let settled = false;
    const waitPromise = session.waitForBackgroundTasksOnPrint().then(() => {
      settled = true;
    });

    await new Promise((resolve) => setImmediate(resolve));
    expect(settled).toBe(false);

    await proc.kill('SIGTERM');
    await waitPromise;
    expect(settled).toBe(true);
    expect(agent.background.getTask(taskId)?.status).toBe('completed');
    await session.close();
  });

  it('waitForBackgroundTasksOnPrint times out after printWaitCeilingS', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-wait-timeout',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      // Sub-second ceiling: the deadline path is identical, but the test no
      // longer waits a real second for the drain loop to time out.
      background: { keepAliveOnExit: true, printWaitCeilingS: 0.05 },
    });
    const agent = await session.createMain();
    const { proc } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'times out'),
    );

    await session.waitForBackgroundTasksOnPrint();

    expect(agent.background.getTask(taskId)?.status).toBe('running');
    await session.close();
  });

  it('handlePrintMainTurnCompleted finishes immediately by default once quiescent (steer mode)', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-mode-default',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    await session.createMain();

    // Default mode is 'steer'; with no pending background tasks the run finishes.
    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('finish');
    await session.close();
  });

  it('handlePrintMainTurnCompleted defaults to steer: continue while a task is pending, then finish', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-mode-default-steer',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();
    const { proc } = pendingProcess();
    agent.background.registerTask(new ProcessBackgroundTask(proc, 'sleep 60', 'steer by default'));

    // No background config at all: the print default is 'steer', so a pending
    // task keeps the run alive.
    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('continue');

    await proc.kill('SIGTERM');
    // Let the background manager observe the terminal status.
    await new Promise((resolve) => setTimeout(resolve, 50));

    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('finish');
    await session.close();
  });

  it('handlePrintMainTurnCompleted drains when printBackgroundMode is drain without keepAliveOnExit', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-mode-drain',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { printBackgroundMode: 'drain' },
    });
    const agent = await session.createMain();
    const { proc } = pendingProcess(0);
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'drain me'),
    );

    let settled = false;
    const promise = session.handlePrintMainTurnCompleted().then((action) => {
      settled = true;
      return action;
    });

    await new Promise((resolve) => setImmediate(resolve));
    expect(settled).toBe(false);

    await proc.kill('SIGTERM');
    await expect(promise).resolves.toBe('finish');
    expect(agent.background.getTask(taskId)?.status).toBe('completed');
    await session.close();
  });

  it('explicit printBackgroundMode exit overrides keepAliveOnExit (no drain)', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-mode-exit-override',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true, printBackgroundMode: 'exit' },
    });
    const agent = await session.createMain();
    const { proc, killSpy } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'no drain'),
    );

    await session.waitForBackgroundTasksOnPrint();
    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('finish');

    expect(killSpy).not.toHaveBeenCalled();
    expect(agent.background.getTask(taskId)?.status).toBe('running');
    await proc.kill('SIGTERM').catch(() => undefined);
    await session.close();
  });

  it('handlePrintMainTurnCompleted returns continue in steer mode while a task is pending, then finish once quiescent', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-mode-steer',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { printBackgroundMode: 'steer' },
    });
    const agent = await session.createMain();
    const { proc } = pendingProcess();
    agent.background.registerTask(new ProcessBackgroundTask(proc, 'sleep 60', 'steer me'));

    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('continue');

    await proc.kill('SIGTERM');
    // Let the background manager observe the terminal status.
    await new Promise((resolve) => setTimeout(resolve, 50));

    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('finish');
    await session.close();
  });

  it('handlePrintMainTurnCompleted finishes in steer mode once printMaxTurns is reached', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-mode-steer-cap',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { printBackgroundMode: 'steer', printMaxTurns: 1 },
    });
    const agent = await session.createMain();
    const { proc } = pendingProcess();
    agent.background.registerTask(new ProcessBackgroundTask(proc, 'sleep 60', 'cap me'));

    // First call: printSteerTurns becomes 1 (not over cap), task pending ⇒ continue.
    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('continue');
    // Second call: printSteerTurns becomes 2 (> printMaxTurns=1) ⇒ finish even though
    // the task is still running.
    await expect(session.handlePrintMainTurnCompleted()).resolves.toBe('finish');

    await proc.kill('SIGTERM').catch(() => undefined);
    await session.close();
  });

  it('waitForBackgroundTasksOnPrint waits for tasks spawned after the first enumeration', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-wait-fanout',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true },
    });
    const agent = await session.createMain();
    const first = pendingProcess(0);
    const firstTaskId = agent.background.registerTask(
      new ProcessBackgroundTask(first.proc, 'sleep 60', 'first'),
    );

    let settled = false;
    const waitPromise = session.waitForBackgroundTasksOnPrint().then(() => {
      settled = true;
    });

    // Let the first enumeration run and suspend on the first task.
    await new Promise((resolve) => setImmediate(resolve));
    expect(settled).toBe(false);

    // Fan out a second background task after the first enumeration.
    const second = pendingProcess(0);
    const secondTaskId = agent.background.registerTask(
      new ProcessBackgroundTask(second.proc, 'sleep 60', 'second'),
    );

    // Finish the first task; the wait must not settle while the second is running.
    await first.proc.kill('SIGTERM');
    await new Promise((resolve) => setImmediate(resolve));
    expect(settled).toBe(false);

    // Finish the second task; the wait should now settle.
    await second.proc.kill('SIGTERM');
    await waitPromise;
    expect(settled).toBe(true);
    expect(agent.background.getTask(firstTaskId)?.status).toBe('completed');
    expect(agent.background.getTask(secondTaskId)?.status).toBe('completed');
    await session.close();
  });

  it('suppresses notifications for every active task before awaiting any of them', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-wait-suppress-race',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true },
    });
    const agent = await session.createMain();
    const steerSpy = vi.spyOn(agent.turn, 'steer');

    // Detached tasks fire a completion notification unless suppressed.
    const first = pendingProcess(0);
    agent.background.registerTask(new ProcessBackgroundTask(first.proc, 'sleep 60', 'first'), {
      detached: true,
    });
    const second = pendingProcess(0);
    agent.background.registerTask(
      new ProcessBackgroundTask(second.proc, 'sleep 60', 'second'),
      { detached: true },
    );

    const waitPromise = session.waitForBackgroundTasksOnPrint();

    // Let the synchronous enumeration run so both tasks get suppressed.
    await new Promise((resolve) => setImmediate(resolve));

    // Complete both tasks after suppression but before the wait settles.
    await first.proc.kill('SIGTERM');
    await second.proc.kill('SIGTERM');
    await new Promise((resolve) => setImmediate(resolve));

    expect(steerSpy).not.toHaveBeenCalled();
    await waitPromise;
    await session.close();
  });

  it('lets the environment override config when deciding background task cleanup', async () => {
    vi.stubEnv('KIMI_CODE_BACKGROUND_KEEP_ALIVE_ON_EXIT', '0');
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-bg-env-cleanup',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true },
    });
    const agent = await session.createMain();
    const { proc, killSpy } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'env cleanup'),
    );

    await session.close();

    expect(killSpy).toHaveBeenCalledWith('SIGTERM');
    expect(agent.background.getTask(taskId)?.status).toBe('killed');
  });

  it('createMain enables print drain when drainAgentTasksOnStop is true', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-drain',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: true, printWaitCeilingS: 42 },
      drainAgentTasksOnStop: true,
    });
    const agent = await session.createMain();

    expect(agent.printDrainAgentTasksOnStop).toBe(true);
    await session.close();
  });

  it('createMain leaves print drain disabled by default', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-print-drain-off',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();

    expect(agent.printDrainAgentTasksOnStop).toBe(false);
    await session.close();
  });

  it('awaits foreground agent shutdown before closing', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-active-turn-cleanup',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();
    const shutdownSettled = createDeferred<void>();
    const shutdownSpy = vi.spyOn(agent.turn, 'shutdown').mockImplementation(async () => {
      await shutdownSettled.promise;
    });

    let closeSettled = false;
    const close = session.close().then(() => {
      closeSettled = true;
    });
    await Promise.resolve();
    expect(closeSettled).toBe(false);

    shutdownSettled.resolve();
    await close;
    expect(shutdownSpy).toHaveBeenCalledWith(expect.any(Error));
  });

  it('does not emit a terminal turn event after session shutdown begins', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const hookStartedPath = join(workDir, 'hook-started');
    const hookScriptPath = join(workDir, 'blocking-user-prompt-hook.cjs');
    await writeFile(
      hookScriptPath,
      [
        "const { writeFileSync } = require('node:fs');",
        "writeFileSync(process.argv[2], 'started');",
        'setInterval(() => {}, 1000);',
        '',
      ].join('\n'),
      'utf-8',
    );
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>(async () => {});
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-close-during-user-hook',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      hooks: [
        {
          event: 'UserPromptSubmit',
          command: `node ${JSON.stringify(hookScriptPath)} ${JSON.stringify(hookStartedPath)}`,
          timeout: 30,
        },
      ],
    });
    const agent = await session.createMain();

    agent.turn.prompt([{ type: 'text', text: 'run the hook' }]);
    await waitForFile(hookStartedPath);
    await session.close();

    const events = emitEvent.mock.calls.map(([event]) => event);
    expect(events).not.toContainEqual(expect.objectContaining({ type: 'turn.ended' }));
  });

  it('keeps background tasks alive and skips SessionEnd hooks when closing for reload', async () => {
    const { command, logPath, sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-reload-close',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
      background: { keepAliveOnExit: false },
      hooks: [
        { event: 'SessionStart', matcher: 'startup', command, timeout: 5 },
        { event: 'SessionEnd', matcher: 'exit', command, timeout: 5 },
      ],
    });
    const agent = await session.createMain();
    const stopSpy = vi.spyOn(agent.cron!, 'stop');
    const { proc, killSpy } = pendingProcess();
    const taskId = agent.background.registerTask(
      new ProcessBackgroundTask(proc, 'sleep 60', 'reload keeps alive'),
    );

    await session.closeForReload();

    expect(stopSpy).toHaveBeenCalledOnce();
    expect(killSpy).not.toHaveBeenCalled();
    expect(agent.background.getTask(taskId)?.status).toBe('running');
    expect(await readHookPayloads(logPath)).toMatchObject([
      {
        hook_event_name: 'SessionStart',
        session_id: 'session-reload-close',
        cwd: workDir,
        source: 'startup',
      },
    ]);
  });

  it('awaits foreground agent shutdown when closing for reload', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-reload-shutdown',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();
    const shutdownSettled = createDeferred<void>();
    const shutdownSpy = vi.spyOn(agent.turn, 'shutdown').mockImplementation(async () => {
      await shutdownSettled.promise;
    });

    let reloadCloseSettled = false;
    const reloadClose = session.closeForReload().then(() => {
      reloadCloseSettled = true;
    });
    await Promise.resolve();
    expect(reloadCloseSettled).toBe(false);

    shutdownSettled.resolve();
    await reloadClose;
    expect(shutdownSpy).toHaveBeenCalledWith(expect.any(Error));
  });

  it('suppresses a delayed agent event after close returns', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>(async () => {});
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-close-late-event',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();
    const releaseLateSetter = createDeferred<void>();
    const lateSetter = releaseLateSetter.promise.then(() => {
      agent.emitEvent({ type: 'warning', message: 'late setter event' });
    });

    await session.close();
    emitEvent.mockClear();
    releaseLateSetter.resolve();
    await lateSetter;

    expect(emitEvent).not.toHaveBeenCalled();
  });

  it('suppresses a delayed agent event after closeForReload returns', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>(async () => {});
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-reload-late-event',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const agent = await session.createMain();
    const releaseLateSetter = createDeferred<void>();
    const lateSetter = releaseLateSetter.promise.then(() => {
      agent.emitEvent({ type: 'warning', message: 'late reload setter event' });
    });

    await session.closeForReload();
    emitEvent.mockClear();
    releaseLateSetter.resolve();
    await lateSetter;

    expect(emitEvent).not.toHaveBeenCalled();
  });

  it('waits for a pending agent creation and prevents it from committing after close', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>(async () => {});
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-close-pending-create',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const bootstrapStarted = createDeferred<void>();
    const releaseBootstrap = createDeferred<void>();
    vi.spyOn(
      session as unknown as {
        bootstrapAgentProfile(agent: Agent, profile: unknown): Promise<void>;
      },
      'bootstrapAgentProfile',
    ).mockImplementation(async (agent) => {
      bootstrapStarted.resolve();
      await releaseBootstrap.promise;
      agent.emitEvent({ type: 'warning', message: 'late create event' });
    });

    const creating = session.createMain();
    await bootstrapStarted.promise;
    let closeSettled = false;
    const close = session.close().then(() => {
      closeSettled = true;
    });
    await Promise.resolve();
    expect(closeSettled).toBe(false);

    emitEvent.mockClear();
    releaseBootstrap.resolve();
    await expect(creating).rejects.toMatchObject({ code: 'session.closed' });
    await close;

    expect(session.getReadyAgent('main')).toBeUndefined();
    expect(emitEvent).not.toHaveBeenCalled();
  });

  it('waits for a pending agent resume and prevents it from committing after reload close', async () => {
    const { sessionDir, workDir } = await hookFixture();
    await writeFile(
      join(sessionDir, 'state.json'),
      JSON.stringify({
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-01T00:00:00.000Z',
        title: 'Pending Resume',
        isCustomTitle: false,
        agents: { main: { type: 'main' } },
        custom: {},
      }),
      'utf-8',
    );
    const emitEvent = vi.fn<SDKSessionRPC['emitEvent']>(async () => {});
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-reload-pending-resume',
      homedir: sessionDir,
      rpc: createSessionRpc({ emitEvent }),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const resumeStarted = createDeferred<void>();
    const releaseResume = createDeferred<void>();
    const resumeSpy = vi.spyOn(Agent.prototype, 'resume').mockImplementation(async function (
      this: Agent,
    ) {
      resumeStarted.resolve();
      await releaseResume.promise;
      this.emitEvent({ type: 'warning', message: 'late resume event' });
      return {};
    });

    try {
      const resuming = session.resume();
      await resumeStarted.promise;
      let closeSettled = false;
      const close = session.closeForReload().then(() => {
        closeSettled = true;
      });
      await Promise.resolve();
      expect(closeSettled).toBe(false);

      emitEvent.mockClear();
      releaseResume.resolve();
      await expect(resuming).rejects.toMatchObject({ code: 'session.closed' });
      await close;

      expect(session.getReadyAgent('main')).toBeUndefined();
      expect(emitEvent).not.toHaveBeenCalled();
    } finally {
      resumeSpy.mockRestore();
    }
  });

  it('aborts and joins an in-flight startup SessionStart before firing SessionEnd', async () => {
    const { sessionDir, workDir } = await hookFixture();
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-close-during-startup-hook',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const startEntered = createDeferred<void>();
    const releaseStart = createDeferred<void>();
    const order: string[] = [];
    let startSignal: AbortSignal | undefined;
    vi.spyOn(session.hookEngine, 'trigger').mockImplementation((event, args = {}) => {
      if (event === 'SessionStart') {
        startSignal = args.signal;
        order.push('SessionStart.begin');
        startEntered.resolve();
        return releaseStart.promise.then(() => {
          order.push('SessionStart.end');
          return [];
        });
      }
      if (event === 'SessionEnd') order.push('SessionEnd');
      return Promise.resolve([]);
    });
    vi.spyOn(
      session as unknown as { flushMetadata(): Promise<void> },
      'flushMetadata',
    ).mockResolvedValue();

    const creating = session.createMain();
    await startEntered.promise;
    const closing = session.close();
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(startSignal?.aborted).toBe(true);
    expect(order).toEqual(['SessionStart.begin']);

    releaseStart.resolve();
    await expect(creating).rejects.toMatchObject({ code: 'session.closed' });
    await closing;
    expect(order).toEqual(['SessionStart.begin', 'SessionStart.end', 'SessionEnd']);
  });

  it('aborts and joins an in-flight resume SessionStart before firing SessionEnd', async () => {
    const { sessionDir, workDir } = await hookFixture();
    await writeFile(
      join(sessionDir, 'state.json'),
      JSON.stringify({
        createdAt: '2026-01-01T00:00:00.000Z',
        updatedAt: '2026-01-01T00:00:00.000Z',
        title: 'Resume Hook Race',
        isCustomTitle: false,
        agents: {},
        custom: {},
      }),
      'utf-8',
    );
    const session = new Session({
      kaos: testKaos.withCwd(workDir),
      id: 'session-close-during-resume-hook',
      homedir: sessionDir,
      rpc: createSessionRpc(),
      skills: { explicitDirs: [join(workDir, 'missing-skills')] },
    });
    const startEntered = createDeferred<void>();
    const releaseStart = createDeferred<void>();
    const order: string[] = [];
    let startSignal: AbortSignal | undefined;
    vi.spyOn(session.hookEngine, 'trigger').mockImplementation((event, args = {}) => {
      if (event === 'SessionStart') {
        startSignal = args.signal;
        order.push('SessionStart.begin');
        startEntered.resolve();
        return releaseStart.promise.then(() => {
          order.push('SessionStart.end');
          return [];
        });
      }
      if (event === 'SessionEnd') order.push('SessionEnd');
      return Promise.resolve([]);
    });
    vi.spyOn(
      session as unknown as { flushMetadata(): Promise<void> },
      'flushMetadata',
    ).mockResolvedValue();

    const resuming = session.resume();
    await startEntered.promise;
    const closing = session.close();
    await new Promise<void>((resolve) => setImmediate(resolve));

    expect(startSignal?.aborted).toBe(true);
    expect(order).toEqual(['SessionStart.begin']);

    releaseStart.resolve();
    await expect(resuming).rejects.toMatchObject({ code: 'session.closed' });
    await closing;
    expect(order).toEqual(['SessionStart.begin', 'SessionStart.end', 'SessionEnd']);
  });
});

async function hookFixture(): Promise<{
  readonly command: string;
  readonly logPath: string;
  readonly sessionDir: string;
  readonly workDir: string;
}> {
  const dir = await makeTempDir();
  const workDir = join(dir, 'work');
  const sessionDir = join(dir, 'session');
  const logPath = join(dir, 'hooks.jsonl');
  const scriptPath = join(dir, 'record-hook.cjs');
  await mkdir(join(workDir, '.git'), { recursive: true });
  await mkdir(sessionDir, { recursive: true });
  await writeFile(
    scriptPath,
    [
      "const { appendFileSync } = require('node:fs');",
      "let input = '';",
      "process.stdin.on('data', (chunk) => { input += chunk; });",
      "process.stdin.on('end', () => { appendFileSync(process.argv[2], `${input.trim()}\\n`); });",
      '',
    ].join('\n'),
    'utf-8',
  );
  return {
    command: `node ${JSON.stringify(scriptPath)} ${JSON.stringify(logPath)}`,
    logPath,
    sessionDir,
    workDir,
  };
}

async function makeTempDir(): Promise<string> {
  const dir = await mkdtemp(join(tmpdir(), 'kimi-session-hooks-'));
  tempDirs.push(dir);
  return dir;
}

async function readHookPayloads(path: string): Promise<readonly Record<string, unknown>[]> {
  const text = await readFile(path, 'utf-8');
  return text
    .trim()
    .split('\n')
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as Record<string, unknown>);
}

function createSessionRpc(overrides: Partial<SDKSessionRPC> = {}): SDKSessionRPC {
  return {
    emitEvent: vi.fn(async () => {}),
    requestApproval: vi.fn(async () => ({ decision: 'cancelled' })),
    requestQuestion: vi.fn(async () => null),
    toolCall: vi.fn(async () => ({
      output: 'custom tools are not supported in this test',
      isError: true,
    })),
    ...overrides,
  } as SDKSessionRPC;
}

async function waitForFile(path: string): Promise<void> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      await readFile(path, 'utf-8');
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 10));
    }
  }
  throw lastError;
}

function createDeferred<T>(): {
  readonly promise: Promise<T>;
  resolve(value: T): void;
} {
  let resolveValue: (value: T) => void = () => {
    /* replaced below */
  };
  const promise = new Promise<T>((resolve) => {
    resolveValue = resolve;
  });
  return {
    promise,
    resolve: resolveValue,
  };
}

function pendingProcess(exitOnKill = 143): {
  readonly proc: KaosProcess;
  readonly killSpy: ReturnType<typeof vi.fn>;
} {
  let resolveWait: (n: number) => void = () => {
    /* replaced below */
  };
  const waitPromise = new Promise<number>((resolve) => {
    resolveWait = resolve;
  });
  let currentExitCode: number | null = null;
  const killSpy = vi.fn(async () => {
    if (currentExitCode !== null) return;
    currentExitCode = exitOnKill;
    resolveWait(exitOnKill);
  });
  const proc: KaosProcess = {
    stdin: { write: vi.fn(), end: vi.fn() } as unknown as Writable,
    stdout: Readable.from([]),
    stderr: Readable.from([]),
    pid: 54_321,
    get exitCode(): number | null {
      return currentExitCode;
    },
    wait: () => waitPromise,
    kill: killSpy as unknown as KaosProcess['kill'],
    dispose: vi.fn().mockResolvedValue(undefined) as KaosProcess['dispose'],
  };
  return { proc, killSpy };
}
