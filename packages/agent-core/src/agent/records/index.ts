import type { Agent } from '..';
import {
  agentRecordAppendAccepted,
  markAgentRecordAppendError,
} from './append-error';
import {
  AGENT_WIRE_PROTOCOL_VERSION,
  isNewerWireVersion,
  migrateWireRecord,
  resolveWireMigrations,
  type WireMigration,
  type WireMigrationRecord,
} from './migration';
import type { AgentRecord, AgentRecordPersistence } from './types';

export * from './types';
export { agentRecordAppendAccepted, markAgentRecordAppendError } from './append-error';
export { AGENT_WIRE_PROTOCOL_VERSION } from './migration';
export {
  FileSystemAgentRecordPersistence,
  InMemoryAgentRecordPersistence,
} from './persistence';
export type { FileSystemAgentRecordPersistenceOptions } from './persistence';
export { BlobStore, isBlobRef } from './blobref';
export type { BlobStoreOptions } from './blobref';

// Contract: restore MUST only rebuild in-memory state. It must not emit UI
// events, call the LLM, execute tools, start background work, make network
// requests, or touch the filesystem in a way that triggers external side effects.
//
// Prefer restoring by calling the same method that wrote the record, so live
// execution and resume share one state mutation path. For example,
// permission.set_mode replays through agent.permission.setMode(input.mode),
// not by assigning modeOverride here. records.logRecord, emitEvent, and
// emitStatusUpdated already gate on records.restoring, so those calls are safe
// during resume.
function restoreAgentRecord(agent: Agent, input: AgentRecord): void {
  switch (input.type) {
    case 'metadata':
      return;
    case 'forked':
      agent.goal.restoreForked(input);
      return;
    case 'turn.prompt':
      agent.turn.restorePrompt(
        input.input,
        input.origin,
        input.deferredPromptId,
        input.promptId,
        input.admissionId,
        input.turnId,
      );
      return;
    case 'turn.deferred_prompt_started':
      agent.turn.restoreDeferredPromptStarted(input.deferredPromptId, input.turnId);
      return;
    case 'turn.steer':
      agent.turn.restoreSteer(
        input.input,
        input.origin,
        input.expectedPromptId,
        input.ownerTurnId,
        input.ownerDeferredPromptId,
        input.admissionId,
        input.turnId,
      );
      return;
    case 'turn.cancel':
      if (
        input.outcomeId !== undefined &&
        input.outcomeTurnId !== undefined &&
        input.outcomeContent !== undefined
      ) {
        agent.turn.restoreOutcome(input.outcomeId, input.outcomeTurnId, input.outcomeContent);
      }
      void agent.turn.cancel(input.turnId, undefined, {
        restoredPromptId: input.promptId,
        restoredOwnerTurnId: input.ownerTurnId,
        restoredOwnerDeferredPromptId: input.ownerDeferredPromptId,
        restoredOutcomeId: input.outcomeId,
        restoredOutcomeTurnId: input.outcomeTurnId,
        restoredOutcomeContent: input.outcomeContent,
        restoredCancelledTurnInputs: input.cancelledTurnInputs,
      });
      return;
    case 'turn.outcome':
      agent.turn.restoreOutcome(input.outcomeId, input.turnId, input.content);
      return;
    case 'config.update':
      agent.config.update(input);
      return;
    case 'permission.set_mode':
      agent.permission.setMode(input.mode);
      return;
    case 'permission.record_approval_result':
      agent.permission.recordApprovalResult(input);
      return;
    case 'usage.record':
      agent.usage.record(input.model, input.usage, 'session');
      return;
    case 'full_compaction.begin':
      agent.fullCompaction.begin(input);
      return;
    case 'full_compaction.cancel':
      void agent.fullCompaction.cancel();
      return;
    case 'full_compaction.complete':
      agent.fullCompaction.markCompleted();
      return;
    case 'micro_compaction.apply':
      agent.microCompaction.apply(input.cutoff);
      return;
    case 'plan_mode.enter':
      agent.planMode.restoreEnter(input);
      return;
    case 'plan_mode.cancel':
      agent.planMode.cancel(input.id);
      return;
    case 'plan_mode.exit':
      agent.planMode.exit(input.id);
      return;
    case 'swarm_mode.enter':
      agent.swarmMode.restoreEnter(input.trigger);
      return;
    case 'swarm_mode.exit':
      agent.swarmMode.exit();
      return;
    case 'context.append_message':
      agent.context.appendMessage(
        input.message,
        input.consumedTurnInput,
        input.turnInputPart,
        input.materializedTurnOutcomeId,
      );
      return;
    case 'turn.input_consumed':
      agent.turn.consumeTurnInput(input.consumedTurnInput);
      return;
    case 'context.append_loop_event':
      agent.context.appendLoopEvent(input.event);
      // Advance the turn counter past internally-driven turns (goal
      // continuations, steer-launched turns) that allocate a turnId without a
      // `turn.prompt` record. Their loop events still carry the real turnId.
      if ('turnId' in input.event) {
        const restoredTurnId = Number.parseInt(input.event.turnId, 10);
        if (!Number.isNaN(restoredTurnId)) {
          agent.turn.observeRestoredTurnId(restoredTurnId);
        }
      }
      return;
    case 'context.clear':
      agent.context.clear();
      return;
    case 'context.apply_compaction':
      agent.context.applyCompaction(input);
      return;
    case 'context.undo':
      agent.context.undo(input.count);
      return;
    case 'tools.register_user_tool':
      agent.tools.registerUserTool(input);
      return;
    case 'tools.unregister_user_tool':
      agent.tools.unregisterUserTool(input.name);
      return;
    case 'tools.set_active_tools':
      agent.tools.setActiveTools(input.names);
      return;
    case 'tools.update_store':
      agent.tools.updateStore(input.key, input.value);
      return;
    case 'goal.create':
      agent.goal.restoreCreate(input);
      return;
    case 'goal.update':
      agent.goal.restoreUpdate(input);
      return;
    case 'goal.clear':
      agent.goal.restoreClear(input);
      return;
    // Observability records: no state to rebuild; only restore the
    // write-dedup cursors so a resumed session does not re-log snapshots
    // that are already durable in this wire log.
    case 'llm.tools_snapshot':
      agent.llmRequestRecorder.restoreToolsSnapshot(input.hash);
      return;
    case 'llm.request':
      return;
    case 'mcp.tools_discovered':
      agent.tools.restoreMcpDiscovery(input.serverName, input.hash);
      return;
  }
}

export interface RestoringContext {
  time?: number;
}

export interface AgentRecordsReplayOptions {
  readonly rewriteMigratedRecords?: boolean;
}

export class AgentRecords {
  private _restoring: RestoringContext | null = null;
  private metadataInitialized = false;
  private _replaying = false;
  /**
   * One-shot latch: the durable log is "open" once replay has completed (the
   * write-dedup cursors of observability records are restored) or, for agents
   * that never resume, once the first record has been logged live. Producers
   * of observability records (MCP discovery) park their writes until then —
   * logging earlier would both duplicate records that replay is about to
   * dedupe and append a stray metadata record ahead of replay.
   */
  private _opened = false;
  private readonly onOpenedCallbacks: Array<() => void> = [];

  constructor(
    private readonly agent: Agent,
    private readonly persistence?: AgentRecordPersistence,
  ) {}

  get restoring() {
    return this._restoring;
  }

  /**
   * Whether observability records may be written directly. False before the
   * log is opened (see `_opened`); producers should park and re-attempt from
   * an `onOpened` callback. Always true without persistence — there is no
   * durable log to protect.
   */
  get observabilityReady(): boolean {
    return this.persistence === undefined || this._opened;
  }

  /**
   * Register a callback fired once, when the log opens. Not fired for a
   * range-limited replay or another read-only projection — those agents are
   * transient previews and must not append new records.
   */
  onOpened(callback: () => void): void {
    if (this._opened) {
      callback();
      return;
    }
    this.onOpenedCallbacks.push(callback);
  }

  private markOpened(): void {
    if (this._opened) return;
    this._opened = true;
    const callbacks = this.onOpenedCallbacks.splice(0);
    for (const callback of callbacks) {
      callback();
    }
  }

  logRecord(record: AgentRecord): void {
    if (this._restoring !== null) return;
    const stamped: AgentRecord =
      record.time !== undefined ? record : { ...record, time: Date.now() };
    if (
      this.persistence !== undefined &&
      !this.metadataInitialized &&
      stamped.type !== 'metadata'
    ) {
      try {
        this.persistence.append({
          type: 'metadata',
          protocol_version: AGENT_WIRE_PROTOCOL_VERSION,
          created_at: Date.now(),
        });
        this.metadataInitialized = true;
      } catch (error) {
        if (agentRecordAppendAccepted(error) === true) {
          this.metadataInitialized = true;
        }
        // The caller's requested record has not been attempted yet, regardless
        // of whether the prerequisite metadata append itself was accepted.
        throw markAgentRecordAppendError(error, false);
      }
    }
    if (stamped.type === 'metadata') {
      this.metadataInitialized = true;
    }
    this.persistence?.append(stamped);
    // A live record was durably logged, so this agent is not waiting on a
    // replay: open the log for observability producers. Guarded against the
    // (currently hypothetical) mid-replay logRecord — opening there would
    // let observability writes race the dedup-cursor restore.
    if (!this._replaying) {
      try {
        this.markOpened();
      } catch (error) {
        // The record was accepted before log-open observers ran.
        throw markAgentRecordAppendError(error, true);
      }
    }
  }

  restore(record: AgentRecord): boolean {
    this._restoring = { time: record.time ?? Date.now() };
    try {
      restoreAgentRecord(this.agent, record);
      return this.agent.replayBuilder.finishRestoringRecord(record.type);
    } finally {
      this._restoring = null;
    }
  }

  async replay(options: AgentRecordsReplayOptions = {}): Promise<{ warning?: string }> {
    return this.replayRecords(options, true);
  }

  /** Restore records for a transient projection without opening the live log. */
  async replayReadOnly(options: AgentRecordsReplayOptions = {}): Promise<{ warning?: string }> {
    return this.replayRecords(options, false);
  }

  private async replayRecords(
    options: AgentRecordsReplayOptions,
    openAfterReplay: boolean,
  ): Promise<{ warning?: string }> {
    if (!this.persistence) throw new Error('No persistence provided for AgentRecords');
    const rewriteMigratedRecords = options.rewriteMigratedRecords ?? true;
    let migrations: readonly WireMigration[] = [];
    let hasMetadata = false;
    let shouldRewrite = false;
    let warning: string | undefined;
    const replayedRecords: AgentRecord[] | undefined = rewriteMigratedRecords ? [] : undefined;
    let completed = true;
    this._replaying = true;
    try {
      for await (const record of this.persistence.read()) {
        if (!hasMetadata) {
          if (record.type !== 'metadata') {
            throw new Error('AgentRecords replay expected metadata as the first record');
          }
          hasMetadata = true;
          this.metadataInitialized = true;
          const readVersion = record.protocol_version;
          if (isNewerWireVersion(readVersion)) {
            warning = `Session wire protocol version ${readVersion} is newer than the current version ${AGENT_WIRE_PROTOCOL_VERSION}. Records will be replayed without migration.`;
            shouldRewrite = false;
          } else {
            migrations = resolveWireMigrations(readVersion);
            shouldRewrite = readVersion !== AGENT_WIRE_PROTOCOL_VERSION;
          }
        }
        let migratedRecord = migrateWireRecord(
          record as WireMigrationRecord,
          migrations,
        ) as AgentRecord;
        if (migratedRecord.type === 'metadata') {
          migratedRecord = {
            ...migratedRecord,
            protocol_version: AGENT_WIRE_PROTOCOL_VERSION,
          };
        }
        replayedRecords?.push(migratedRecord);
        if (this.restore(migratedRecord)) {
          completed = false;
          break;
        }
      }
      if (completed && shouldRewrite && replayedRecords !== undefined) {
        this.persistence.rewrite(replayedRecords);
        await this.persistence.flush();
      }
      if (completed && this.agent.blobStore !== undefined) {
        for (const msg of this.agent.context.history) {
          await this.agent.blobStore.rehydrateParts(msg.content);
        }
      }
    } finally {
      this._replaying = false;
    }
    // Open only AFTER the migration rewrite has flushed — records appended by
    // onOpened callbacks before the rewrite would be wiped by it. A frozen
    // range or an explicitly read-only projection never opens: see onOpened.
    if (completed && openAfterReplay) {
      this.markOpened();
    }
    return { warning };
  }

  async flush(): Promise<void> {
    await this.persistence?.flush();
  }
}
