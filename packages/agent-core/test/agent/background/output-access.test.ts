/**
 * BackgroundManager output retrieval surface.
 */

import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { Readable } from 'node:stream';
import type { Writable } from 'node:stream';
import { join } from 'pathe';

import { createControlledPromise } from '@antfu/utils';
import type { KaosProcess } from '@moonshot-ai/kaos';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { BackgroundManager } from '../../../src/agent/background';
import {
  createBackgroundManager,
  registerProcess,
  waitForOutput,
} from './helpers';

function immediateProcess(exitCode: number, stdoutText = ''): KaosProcess {
  return {
    stdin: { write: vi.fn(), end: vi.fn() } as unknown as Writable,
    stdout: Readable.from(stdoutText ? [stdoutText] : []),
    stderr: Readable.from([]),
    pid: 50000 + exitCode,
    exitCode,
    wait: vi.fn().mockResolvedValue(exitCode) as KaosProcess['wait'],
    kill: vi.fn().mockResolvedValue(undefined) as KaosProcess['kill'],
    dispose: vi.fn().mockResolvedValue(undefined) as KaosProcess['dispose'],
  };
}

describe('BackgroundManager — readOutput / getOutputSnapshot', () => {
  let sessionDir: string;
  let manager: BackgroundManager;
  let persistence: NonNullable<ReturnType<typeof createBackgroundManager>['persistence']>;

  beforeEach(() => {
    sessionDir = mkdtempSync(join(tmpdir(), 'bpm-output-'));
    const fixture = createBackgroundManager({ sessionDir });
    manager = fixture.manager;
    persistence = fixture.persistence!;
  });

  afterEach(async () => {
    // Include tasks whose status flipped to terminal just before their final
    // persistence completed. Active-only listing would hide them and let the
    // temporary directory be removed underneath the queued write.
    await Promise.all(manager.list(false).map((task) => manager.wait(task.taskId)));
    rmSync(sessionDir, { recursive: true, force: true });
  });

  it('getOutputSnapshot returns output.log path when persisted output exists', async () => {
    const taskId = registerProcess(manager, immediateProcess(0, 'hello\n'), 'echo', 'demo');

    await waitForOutput(manager, taskId, 'hello');
    const snapshot = await manager.getOutputSnapshot(taskId, 1_000);

    expect(snapshot.outputPath).toBeDefined();
    expect(snapshot.outputPath).toContain(sessionDir);
    expect(snapshot.outputPath).toContain(taskId);
    expect(snapshot.outputPath!.endsWith('output.log')).toBe(true);
    expect(snapshot.fullOutputAvailable).toBe(true);
  });

  it('getOutputSnapshot omits outputPath when no persisted log file exists', async () => {
    const taskId = registerProcess(manager, immediateProcess(0), 'sleep 1', 'silent task');

    await manager.wait(taskId);
    const snapshot = await manager.getOutputSnapshot(taskId, 1_000);

    expect(snapshot.outputPath).toBeUndefined();
    expect(snapshot.fullOutputAvailable).toBe(false);
  });

  it('getOutputSnapshot returns an empty snapshot for unknown task ids', async () => {
    await expect(manager.getOutputSnapshot('bash-deadbeef', 1_000)).resolves.toEqual({
      outputSizeBytes: 0,
      previewBytes: 0,
      truncated: false,
      fullOutputAvailable: false,
      preview: '',
    });
  });

  it('readOutput returns live ring-buffer content while task is in memory', async () => {
    const taskId = registerProcess(
      manager,
      immediateProcess(0, 'live content\n'),
      'echo',
      'demo',
    );

    await waitForOutput(manager, taskId, 'live content');

    expect(await manager.readOutput(taskId)).toContain('live content');
  });

  it('readOutput prefers disk over the live ring buffer when persisted output exists', async () => {
    const taskId = registerProcess(manager, immediateProcess(0, 'ring-only\n'), 'echo', 'demo');

    await waitForOutput(manager, taskId, 'ring-only');
    await persistence.appendTaskOutput(taskId, 'disk-only\n');

    expect(await manager.readOutput(taskId)).toContain('disk-only');
  });

  it('readOutput falls back to disk for ghost tasks', async () => {
    const taskId = registerProcess(
      manager,
      immediateProcess(0, 'persisted line\n'),
      'echo',
      'demo',
    );
    await waitForOutput(manager, taskId, 'persisted line');
    await manager.wait(taskId);

    const fresh = createBackgroundManager({ sessionDir }).manager;
    await fresh.loadFromDisk();
    await fresh.reconcile();

    expect(await fresh.readOutput(taskId)).toContain('persisted line');
  });

  it('readOutput respects tail length', async () => {
    const taskId = registerProcess(
      manager,
      immediateProcess(0, 'aaaaa-bbbbb-ccccc-ddddd'),
      'echo',
      'demo',
    );

    await waitForOutput(manager, taskId, 'ddddd');

    expect(await manager.readOutput(taskId, 5)).toBe('ddddd');
  });

  it('wait exposes a completed process only after queued output reaches output.log', async () => {
    const outputWriteStarted = createControlledPromise<void>();
    const releaseOutputWrite = createControlledPromise<void>();
    const processDisposed = createControlledPromise<void>();
    const events: string[] = [];
    const appendTaskOutput = persistence.appendTaskOutput.bind(persistence);
    const writeTask = persistence.writeTask.bind(persistence);

    vi.spyOn(persistence, 'appendTaskOutput').mockImplementation(async (taskId, chunk) => {
      events.push('output write started');
      outputWriteStarted.resolve();
      await releaseOutputWrite;
      await appendTaskOutput(taskId, chunk);
      events.push('output write finished');
    });
    vi.spyOn(persistence, 'writeTask').mockImplementation(async (info) => {
      if (info.status !== 'completed') return;
      events.push('terminal write started');
      await writeTask(info);
    });

    const proc = immediateProcess(0, 'persisted before wait\n');
    vi.mocked(proc.dispose).mockImplementation(async () => {
      processDisposed.resolve();
    });
    const taskId = registerProcess(manager, proc, 'echo', 'demo');
    let waitReturned = false;
    const waitResult = manager.wait(taskId).then((info) => {
      waitReturned = true;
      return info;
    });

    await Promise.all([outputWriteStarted, processDisposed]);
    // Cross one event-loop boundary after the process streams drained and the
    // process was disposed, giving lifecycle finalization a chance to run
    // without relying on wall-clock time.
    await new Promise<void>((resolve) => setImmediate(resolve));
    const statusWhileOutputWriteBlocked = manager.getTask(taskId)?.status;
    const waitReturnedWhileOutputWriteBlocked = waitReturned;
    releaseOutputWrite.resolve();

    await expect(waitResult).resolves.toMatchObject({ status: 'completed' });
    expect(statusWhileOutputWriteBlocked).toBe('running');
    expect(waitReturnedWhileOutputWriteBlocked).toBe(false);
    expect(events).toEqual([
      'output write started',
      'output write finished',
      'terminal write started',
    ]);
    await expect(manager.readOutput(taskId)).resolves.toBe('persisted before wait\n');
    await expect(persistence.readTask(taskId)).resolves.toMatchObject({ status: 'completed' });
  });
});
