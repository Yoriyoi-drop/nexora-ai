import { useEffect, useCallback, useRef } from 'react';
import { useNexoraStore, type TabView } from './store/nexoraStore';
import { telemetryService } from './services/telemetryService';
import { alertEngine } from './services/alertEngine';
import { ErrorBoundary } from './components/ErrorBoundary';
import TopBar from './components/TopBar';
import Sidebar from './components/Sidebar';
import TopologyView from './components/views/TopologyView';
import MemoryObservatory from './components/views/MemoryObservatory';
import TokenFlowView from './components/views/TokenFlowView';
import EntropyView from './components/views/EntropyView';
import TimelineView from './components/views/TimelineView';
import InfrastructureView from './components/views/InfrastructureView';
import SelfObservationAgent from './components/SelfObservationAgent';
import ModelPanel from './components/ModelPanel';
import ConnectionDialog from './components/ConnectionDialog';

export default function App() {
  const {
    activeTab, isLive, tickSimulation,
    updateFromSystemMetrics, updateFromAIMetrics, updateFromAggregatedTelemetry,
    setConnectionStatus,
  } = useNexoraStore();
  const alertIntervalRef = useRef<ReturnType<typeof setInterval>>(undefined);

  // Initialize telemetry on mount
  useEffect(() => {
    telemetryService.start().then(async () => {
      // Try to auto-connect if previously configured
      const status = await telemetryService.checkConnection();
      setConnectionStatus(status.connected ? 'connected' : 'disconnected');
    });

    return () => {
      telemetryService.stop();
    };
  }, [setConnectionStatus]);

  // Subscribe to ALL telemetry data
  useEffect(() => {
    const unsub = telemetryService.subscribe((update) => {
      if (update.systemMetrics) {
        updateFromSystemMetrics(update.systemMetrics);
      }
      if (update.aiMetrics) {
        updateFromAIMetrics(update.aiMetrics);
      }
      if (update.aggregated) {
        updateFromAggregatedTelemetry(update.aggregated);
      }
    });
    return unsub;
  }, [updateFromSystemMetrics, updateFromAIMetrics, updateFromAggregatedTelemetry]);

  // Wire alert engine to store metrics
  useEffect(() => {
    alertEngine.setMetricProvider((metric: string) => {
      const state = useNexoraStore.getState();
      const sm = state.systemMetrics;
      const realInf = state.realInference;
      const realHal = state.realHallucinations;
      const lookup: Record<string, number | undefined> = {
        cpuPercent: sm.cpuPercent,
        ramPercent: sm.ramPercent,
        gpuUtilization: sm.gpuUtilization,
        latencyMs: realInf?.latency_ms ?? sm.inferenceLatencyMs,
        tokensPerSec: sm.totalTokensPerSec,
        hallucinationRate: realHal?.hallucination_rate ?? sm.hallucinationRate,
        avgEntropy: sm.avgEntropy,
        memoryFragmentation: sm.memoryFragmentation,
        errorRate: realInf?.error_rate,
        swarmCoherence: sm.swarmCoherence,
      };
      return lookup[metric];
    });

    alertIntervalRef.current = setInterval(() => {
      alertEngine.evaluate();
    }, 3000);

    return () => {
      if (alertIntervalRef.current) clearInterval(alertIntervalRef.current);
    };
  }, []);

  // Simulation tick (for mock data fallback)
  useEffect(() => {
    if (!isLive) return;
    const interval = setInterval(tickSimulation, 1200);
    return () => clearInterval(interval);
  }, [isLive, tickSimulation]);

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'c' && (e.metaKey || e.ctrlKey)) return; // Allow copy
    const tabs: TabView[] = ['topology', 'memory', 'tokenflow', 'entropy', 'timeline', 'infrastructure'];
    const numKeys = ['1', '2', '3', '4', '5', '6'];
    const idx = numKeys.indexOf(e.key);
    if (idx >= 0 && idx < tabs.length) {
      useNexoraStore.getState().setActiveTab(tabs[idx]);
    }
  }, []);

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return (
    <ErrorBoundary>
      <div className="flex flex-col h-screen w-screen overflow-hidden bg-slate-950 text-slate-200">
        {/* Scanline overlay */}
        <div
          className="pointer-events-none fixed inset-0 z-50 opacity-[0.025]"
          style={{
            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.4) 2px, rgba(0,0,0,0.4) 4px)',
          }}
        />

        {/* Top bar */}
        <TopBar />

        {/* Main content */}
        <div className="flex flex-1 overflow-hidden">
          <Sidebar />
          <main className="flex-1 overflow-hidden relative">
            {activeTab === 'topology' && <TopologyView />}
            {activeTab === 'memory' && <MemoryObservatory />}
            {activeTab === 'tokenflow' && <TokenFlowView />}
            {activeTab === 'entropy' && <EntropyView />}
            {activeTab === 'timeline' && <TimelineView />}
            {activeTab === 'infrastructure' && <InfrastructureView />}
          </main>
        </div>

        {/* Floating overlays */}
        <SelfObservationAgent />
        <ModelPanel />
        <ConnectionDialog />
      </div>
    </ErrorBoundary>
  );
}
