export type AlertSeverity = 'info' | 'warning' | 'critical' | 'fatal';
export type AlertCondition = 'greater_than' | 'less_than' | 'rate_increase' | 'equals';

export interface AlertRule {
  id: string;
  name: string;
  description: string;
  source: 'system' | 'inference' | 'agent' | 'memory' | 'pipeline';
  metric: string;
  condition: AlertCondition;
  threshold: number;
  severity: AlertSeverity;
  enabled: boolean;
  cooldownSecs: number;
  lastTriggeredAt?: number;
}

export interface Alert {
  id: string;
  ruleId: string;
  name: string;
  message: string;
  severity: AlertSeverity;
  value: number;
  threshold: number;
  timestamp: number;
  acknowledged: boolean;
}

type MetricProvider = (metric: string) => number | undefined;

class AlertEngine {
  private rules: AlertRule[] = [];
  private alerts: Alert[] = [];
  private listeners: Array<(alerts: Alert[]) => void> = [];
  private metricProvider: MetricProvider = () => undefined;

  constructor() {
    this.rules = this.defaultRules();
  }

  private defaultRules(): AlertRule[] {
    return [
      { id: 'cpu-high', name: 'CPU Overload', description: 'CPU usage exceeds threshold', source: 'system', metric: 'cpuPercent', condition: 'greater_than', threshold: 90, severity: 'critical', enabled: true, cooldownSecs: 60 },
      { id: 'ram-high', name: 'RAM Exhaustion', description: 'RAM usage exceeds threshold', source: 'system', metric: 'ramPercent', condition: 'greater_than', threshold: 90, severity: 'fatal', enabled: true, cooldownSecs: 30 },
      { id: 'gpu-high', name: 'GPU Overload', description: 'GPU utilization exceeds threshold', source: 'system', metric: 'gpuUtilization', condition: 'greater_than', threshold: 95, severity: 'critical', enabled: true, cooldownSecs: 60 },
      { id: 'latency-spike', name: 'Latency Spike', description: 'Inference latency too high', source: 'inference', metric: 'latencyMs', condition: 'greater_than', threshold: 500, severity: 'warning', enabled: true, cooldownSecs: 120 },
      { id: 'token-drop', name: 'Token Speed Drop', description: 'Tokens per second too low', source: 'inference', metric: 'tokensPerSec', condition: 'less_than', threshold: 100, severity: 'warning', enabled: true, cooldownSecs: 120 },
      { id: 'hallucination-spike', name: 'Hallucination Spike', description: 'Hallucination rate too high', source: 'agent', metric: 'hallucinationRate', condition: 'greater_than', threshold: 0.1, severity: 'critical', enabled: true, cooldownSecs: 300 },
      { id: 'entropy-high', name: 'High Cognitive Entropy', description: 'Average entropy exceeds safe level', source: 'agent', metric: 'avgEntropy', condition: 'greater_than', threshold: 0.8, severity: 'critical', enabled: true, cooldownSecs: 120 },
      { id: 'memory-frag', name: 'Memory Fragmentation', description: 'Memory fragmentation too high', source: 'memory', metric: 'memoryFragmentation', condition: 'greater_than', threshold: 0.7, severity: 'warning', enabled: true, cooldownSecs: 180 },
      { id: 'error-rate', name: 'Error Rate Spike', description: 'Error rate exceeds threshold', source: 'inference', metric: 'errorRate', condition: 'greater_than', threshold: 0.05, severity: 'critical', enabled: true, cooldownSecs: 60 },
      { id: 'coherence-drop', name: 'Swarm Coherence Drop', description: 'Swarm coherence below threshold', source: 'agent', metric: 'swarmCoherence', condition: 'less_than', threshold: 0.5, severity: 'critical', enabled: true, cooldownSecs: 120 },
    ];
  }

  setMetricProvider(provider: MetricProvider) {
    this.metricProvider = provider;
  }

  subscribe(listener: (alerts: Alert[]) => void) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener);
    };
  }

  getRules(): AlertRule[] {
    return this.rules;
  }

  getAlerts(): Alert[] {
    return this.alerts;
  }

  addRule(rule: AlertRule) {
    this.rules.push(rule);
  }

  removeRule(ruleId: string) {
    this.rules = this.rules.filter(r => r.id !== ruleId);
  }

  toggleRule(ruleId: string) {
    this.rules = this.rules.map(r =>
      r.id === ruleId ? { ...r, enabled: !r.enabled } : r
    );
  }

  acknowledgeAlert(alertId: string) {
    this.alerts = this.alerts.map(a =>
      a.id === alertId ? { ...a, acknowledged: true } : a
    );
  }

  clearAlerts() {
    this.alerts = [];
  }

  evaluate() {
    const now = Date.now();
    const newAlerts: Alert[] = [];

    for (const rule of this.rules) {
      if (!rule.enabled) continue;

      const value = this.metricProvider(rule.metric);
      if (value === undefined) continue;

      // Check cooldown
      if (rule.lastTriggeredAt && (now - rule.lastTriggeredAt) < rule.cooldownSecs * 1000) {
        continue;
      }

      let triggered = false;
      switch (rule.condition) {
        case 'greater_than':
          triggered = value > rule.threshold;
          break;
        case 'less_than':
          triggered = value < rule.threshold;
          break;
        case 'rate_increase':
          triggered = value > rule.threshold;
          break;
        case 'equals':
          triggered = Math.abs(value - rule.threshold) < 0.01;
          break;
      }

      if (triggered) {
        const alert: Alert = {
          id: `alert-${now}-${rule.id}`,
          ruleId: rule.id,
          name: rule.name,
          message: `${rule.description}: ${value.toFixed(1)} (threshold: ${rule.threshold})`,
          severity: rule.severity,
          value,
          threshold: rule.threshold,
          timestamp: now,
          acknowledged: false,
        };
        newAlerts.push(alert);

        // Update cooldown
        rule.lastTriggeredAt = now;
      }
    }

    if (newAlerts.length > 0) {
      this.alerts = [...newAlerts, ...this.alerts].slice(0, 50);
      this.listeners.forEach(l => l(this.alerts));
    }
  }
}

export const alertEngine = new AlertEngine();
