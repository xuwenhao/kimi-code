import type { Agent } from '..';
import type { AgentReplayRecord, AgentReplayRecordPayload } from '../../rpc/resumed';
import type { ContextMessage } from '../context';

export interface ReplayRangeOptions {
  readonly start?: number;
  readonly count?: number;
}

export interface ReplayBuilderOptions {
  readonly range?: ReplayRangeOptions;
}

const UNDO_BOUNDARY_RECORD_TYPES = new Set(['context.clear', 'context.apply_compaction']);

export class ReplayBuilder {
  postRestoring = false;
  captureLiveRecords = false;
  protected readonly records: AgentReplayRecord[] = [];
  private frozen = false;
  private segmentStart = 0;

  constructor(
    public readonly agent: Agent,
    private readonly options: ReplayBuilderOptions = {},
  ) {}

  push(record: AgentReplayRecordPayload): AgentReplayRecord | undefined {
    if (this.captureLiveRecords || this.agent.records.restoring || this.postRestoring) {
      if (this.frozen) return undefined;
      const stamped: AgentReplayRecord = {
        ...record,
        time: this.agent.records.restoring?.time ?? Date.now(),
      };
      this.records.push(stamped);
      return stamped;
    }
    return undefined;
  }

  patchLast<T extends AgentReplayRecord['type']>(
    type: T,
    patch: Partial<Extract<AgentReplayRecord, { type: T }>>,
  ): void {
    if (this.frozen) return;
    if (this.agent.records.restoring) {
      // A compaction remains open while unrelated config/permission/goal replay
      // records may be appended. Locate the latest matching record rather than
      // assuming the target is the array tail. For compactions, never rewrite a
      // prior terminal if a malformed/duplicate terminal arrives with no open
      // begin record.
      const target = this.records.findLast(
        (record) =>
          record.type === type &&
          (record.type !== 'compaction' || record.result === undefined),
      );
      if (target !== undefined) {
        Object.assign(target, patch);
      }
    }
  }

  removeLastMessages(removedMessages: ReadonlySet<ContextMessage>): void {
    if (this.frozen) return;
    if (removedMessages.size === 0) return;
    this.removeMessagesFrom(this.records, removedMessages);
  }

  finishRestoringRecord(type: string): boolean {
    const range = this.options.range;
    if (range === undefined) return false;
    if (this.frozen) return true;
    if (!UNDO_BOUNDARY_RECORD_TYPES.has(type)) return false;
    if (range.start === undefined) return false;

    const start = range.start;
    const nextSegmentStart = this.segmentStart + this.records.length;
    if (nextSegmentStart > start) {
      this.frozen = true;
      return true;
    }

    this.segmentStart = nextSegmentStart;
    this.records.splice(0);
    return false;
  }

  buildResult(): readonly AgentReplayRecord[] {
    const range = this.options.range;
    if (range !== undefined) {
      if (range.start === undefined && range.count !== undefined) {
        const offset = Math.max(0, this.records.length - range.count);
        return this.records.slice(offset);
      }
      const start = range.start ?? 0;
      const offset = Math.max(0, start - this.segmentStart);
      const count = range.count;
      const end = count === undefined ? undefined : offset + count;
      return this.records.slice(offset, end);
    }
    return this.records;
  }

  private removeMessagesFrom(
    records: AgentReplayRecord[],
    removedMessages: ReadonlySet<ContextMessage>,
  ): void {
    for (let i = records.length - 1; i >= 0; i--) {
      const record = records[i]!;
      if (record.type === 'message' && removedMessages.has(record.message)) {
        records.splice(i, 1);
      }
    }
  }
}
