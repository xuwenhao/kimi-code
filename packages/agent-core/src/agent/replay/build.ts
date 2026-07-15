import { LocalKaos } from '@moonshot-ai/kaos';

import type { AgentReplayRecord } from '../../rpc/resumed';
import { Agent } from '../index';
import type { AgentRecordPersistence } from '../records';
import type { ReplayRangeOptions } from '.';

export async function buildReplay(
  persistence: AgentRecordPersistence,
  range?: ReplayRangeOptions,
): Promise<readonly AgentReplayRecord[]> {
  const agent = new Agent({
    kaos: await LocalKaos.create(),
    persistence,
    type: 'sub',
    replay: { range },
  });
  // A replay projection is not a resumable runtime. Rebuild state only; do not
  // run Agent.resume's background reconciliation, pending-turn launch, or
  // compaction recovery, and keep record-open callbacks parked so they cannot
  // append observability/recovery records to the source being projected.
  await agent.records.replayReadOnly({ rewriteMigratedRecords: false });
  return agent.replayBuilder.buildResult();
}
