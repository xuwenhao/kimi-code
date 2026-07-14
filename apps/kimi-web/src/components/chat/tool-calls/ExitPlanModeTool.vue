<!-- apps/kimi-web/src/components/chat/tool-calls/ExitPlanModeTool.vue -->
<!-- ExitPlanMode renders its plan as a markdown card instead of a raw output
     dump: the plan body arrives via the paired tool_use part's persisted
     plan_review toolData (seeded live from tool.call.started / preserved
     approval display while pending — see messagesToTurns.ts), so approved AND
     rejected plans survive a page reload. The card status is TEXT-ONLY (no
     ✓/✗ icon — plan approval shows words), derived from the tool result's
     outcome × approval-record join, the same 判定链 generic tool rows use. -->
<script setup lang="ts">
import { computed, ref, watch } from 'vue';
import { useI18n } from 'vue-i18n';
import type { FilePreviewRequest, ToolCall, ToolMedia, ToolPlan } from '../../../types';
import { toolGlyph, toolLabel } from '../../../lib/toolMeta';
import { STATUS_BADGE_VIEW, statusBadgeView } from '../../../lib/toolStatus';
import ToolRow from '../ToolRow.vue';
import Markdown from '../Markdown.vue';
import Badge from '../../ui/Badge.vue';
import ToolOutputBlock from './ToolOutputBlock.vue';

const props = withDefaults(
  defineProps<{
    tool: ToolCall;
    mobile?: boolean;
    stackPosition?: 'single' | 'first' | 'middle' | 'last';
    toolDiffPanel?: boolean;
  }>(),
  { mobile: false, stackPosition: 'single', toolDiffPanel: false },
);

const emit = defineEmits<{
  openMedia: [media: ToolMedia];
  openFile: [target: FilePreviewRequest];
  openToolDiff: [id: string];
}>();

const { t } = useI18n();

const plan = computed<ToolPlan | undefined>(() => props.tool.plan);
const hasPlanBody = computed(() => !!plan.value?.content);
const hasOutput = computed(() => !!props.tool.output && props.tool.output.length > 0);
const canExpand = computed(() => hasPlanBody.value || !!plan.value?.feedback || hasOutput.value);
const open = ref(props.tool.defaultExpanded === true && canExpand.value);

const label = computed(() => toolLabel(props.tool.name));
const glyph = computed(() => toolGlyph(props.tool.name));

const badge = computed<{ text: string; variant: 'neutral' | 'success' | 'warning' | 'danger' } | null>(
  () => {
    // Control-flow badge from the generic 判定链 (rejected / not run / invalid /
    // …) — same rendering as any other tool row, text-only on this card.
    const generic = statusBadgeView(props.tool.statusBadge);
    if (generic) return { text: t(generic.key), variant: generic.variant };
    // A genuinely failed call carries no badge but must not show "pending".
    if (props.tool.outcome === 'failed') {
      return { text: t('tools.status.failed'), variant: 'danger' };
    }
    switch (plan.value?.status) {
      case 'approved':
        return { text: t('approval.planCard.approved'), variant: 'success' };
      case 'auto_approved':
        return { text: t('approval.planCard.autoApproved'), variant: 'warning' };
      case 'rejected':
        return { text: t('approval.planCard.rejected'), variant: 'danger' };
      case 'revise':
        return { text: t('approval.planCard.revise'), variant: 'warning' };
      case 'cancelled':
        return { text: t(STATUS_BADGE_VIEW.cancelled.key), variant: STATUS_BADGE_VIEW.cancelled.variant };
      case 'denied':
        return { text: t(STATUS_BADGE_VIEW.denied.key), variant: STATUS_BADGE_VIEW.denied.variant };
      case 'not_run':
        return { text: t(STATUS_BADGE_VIEW.notRun.key), variant: STATUS_BADGE_VIEW.notRun.variant };
      case 'pending':
        return { text: t('approval.planCard.pending'), variant: 'neutral' };
      default:
        return null;
    }
  },
);

function toggle(): void {
  if (canExpand.value) open.value = !open.value;
}

watch(
  () => [props.tool.defaultExpanded, props.tool.output?.length, props.tool.status, props.tool.plan] as const,
  () => {
    if (props.tool.defaultExpanded === true && canExpand.value) open.value = true;
  },
);
</script>

<template>
  <ToolRow
    :icon="glyph"
    :name="label"
    :time="tool.timing"
    :open="open"
    :expandable="canExpand"
    :stacked="stackPosition !== 'single'"
    :stack-position="stackPosition"
    @toggle="toggle"
  >
    <template #trailing>
      <Badge v-if="badge" :variant="badge.variant" size="sm">{{ badge.text }}</Badge>
      <span v-if="plan?.chosenOption" class="chip">{{ plan.chosenOption }}</span>
    </template>

    <div v-if="hasPlanBody" class="plan-body">
      <Markdown :text="plan!.content!" :open-file="(target: FilePreviewRequest) => emit('openFile', target)" />
    </div>

    <div v-if="plan?.feedback" class="plan-feedback">
      <span class="plan-feedback-label">{{ t('approval.planCard.feedback') }}</span>
      <span class="plan-feedback-text">{{ plan.feedback }}</span>
    </div>

    <!-- Hook-blocked plan (not_run with no approval record): the block reason
         lives in the tool result output — show it in the expanded body.
         Defensive fallback: no plan card data at all (legacy transcript) —
         show the raw output like GenericTool would. -->
    <ToolOutputBlock v-if="hasOutput && (!plan || plan.status === 'not_run')" :lines="tool.output" empty-text="Waiting for output…" />
  </ToolRow>
</template>

<style scoped>
.plan-body {
  max-height: 50vh;
  overflow-y: auto;
}

.plan-feedback {
  margin-top: var(--space-3);
  padding-top: var(--space-2);
  border-top: 1px dashed var(--color-line);
  font-family: var(--font-ui);
  white-space: pre-wrap;
}
.plan-feedback-label {
  display: block;
  color: var(--color-warning);
  font-size: var(--text-xs);
  font-weight: var(--weight-medium);
  margin-bottom: 2px;
}
.plan-feedback-text {
  color: var(--color-text);
  font-size: var(--text-sm);
}
</style>
