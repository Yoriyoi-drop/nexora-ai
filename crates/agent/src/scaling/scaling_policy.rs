use super::AgentScalingConfig;

/// Keputusan scaling
#[derive(Debug, Clone, PartialEq)]
pub enum ScalingDecision {
    ScaleUp(usize),
    ScaleDown(usize),
    Hold,
}

/// Policy engine untuk scaling decisions
pub struct ScalingPolicy {
    config: AgentScalingConfig,
    consecutive_low: u32,
    consecutive_high: u32,
}

impl ScalingPolicy {
    pub fn new(config: AgentScalingConfig) -> Self {
        Self {
            config,
            consecutive_low: 0,
            consecutive_high: 0,
        }
    }

    /// Evaluasi apakah perlu scale berdasarkan metrics
    pub fn evaluate(
        &mut self,
        cpu_util: f64,
        mem_util: f64,
        queue_depth: usize,
        active_agents: usize,
    ) -> ScalingDecision {
        let util = cpu_util.max(mem_util);

        // Scale up: utilization > threshold ATAU queue depth > threshold
        if util > self.config.scale_up_threshold
            || queue_depth > self.config.queue_depth_threshold
        {
            self.consecutive_high += 1;
            self.consecutive_low = 0;

            if self.consecutive_high >= 1 {
                let max_add = self.config.max_agents.saturating_sub(active_agents);
                let to_add = self.config.scale_up_by.min(max_add);
                if to_add > 0 {
                    self.consecutive_high = 0;
                    return ScalingDecision::ScaleUp(to_add);
                }
            }
        }
        // Scale down: utilization rendah selama N cycle
        else if util < self.config.scale_down_threshold {
            self.consecutive_low += 1;
            self.consecutive_high = 0;

            let min_after = active_agents.saturating_sub(self.config.scale_down_by);
            if self.consecutive_low >= 3 && min_after >= self.config.min_agents {
                self.consecutive_low = 0;
                return ScalingDecision::ScaleDown(self.config.scale_down_by);
            }
        } else {
            self.consecutive_low = 0;
            self.consecutive_high = 0;
        }

        ScalingDecision::Hold
    }

    pub fn config(&self) -> &AgentScalingConfig {
        &self.config
    }
}
