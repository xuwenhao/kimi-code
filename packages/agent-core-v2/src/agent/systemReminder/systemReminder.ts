import { createDecorator } from '#/_base/di/instantiation';

import type { ContextMessage, PromptOrigin } from '#/agent/contextMemory/types';

export interface IAgentSystemReminderService {
  readonly _serviceBrand: undefined;

  appendSystemReminder(
    content: string,
    origin: PromptOrigin,
    materializedTurnOutcomeId?: string,
  ): ContextMessage;
}

export const IAgentSystemReminderService = createDecorator<IAgentSystemReminderService>('agentSystemReminderService');
