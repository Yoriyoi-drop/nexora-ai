import React, { useMemo } from 'react';
import { useNexoraStore, TabView } from '../store/nexoraStore';
import { Network, Database, Layers, Server, GitCompare, Activity, TrendingUp, ChevronRight, Wifi, WifiOff } from 'lucide-react';

const tabs: { id: TabView; icon: React.ReactNode; label: string; sub: string }[] = [
  { id: 'topology', icon: <Network size={15} />, label: 'Neural Topology', sub: 'Agent graph & clusters' },
  { id: 'memory', icon: <Database size={15} />, label: 'Memory Observatory', sub: 'Semantic terrain' },
  { id: 'tokenflow', icon: <Activity size={15} />, label: 'Token Flow', sub: 'Particle simulation' },
  { id: 'entropy', icon: <TrendingUp size={15} />, label: 'Cognitive Entropy', sub: 'Reasoning analytics' },
  { id: 'timeline', icon: <GitCompare size={15} />, label: 'Timeline & Diff', sub: 'Multi-run comparison' },
  { id: 'infrastructure', icon: <Server size={15} />, label: 'Infrastructure', sub: 'Distributed systems' },
];

const severityColor = (s: string) =>
  s === 'critical' ? 'text-red-400' : s === 'warn' ? 'text-amber-400' : 'text-slate-400';

const typeColor = (t: string) => {
  const map: Record<string, string> = {
    reasoning: 'text-cyan-400', memory: 'text-violet-400', planning: 'text-blue-400',
    validation: 'text-emerald-400', hallucination: 'text-red-400', completion: 'text-teal-400',
  };
  return map[t] || 'text-slate-400';
};

export default function Sidebar() {
  const { activeTab, setActiveTab, systemMetrics, cognitiveEvents, dataSource, connectionStatus, realSystemMetrics } = useNexoraStore();
  const recentEvents = useMemo(() => cognitiveEvents.slice(0, 8), [cognitiveEvents]);

  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  return (
    <aside className="flex flex-col w-52 border-r border-slate-800/80 glass-dark shrink-0 overflow-hidden">
      {/* Connection status */}
      <div className="px-2 pt-2 pb-1">
        <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-[9px] font-mono ${isLiveData ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-slate-800/30 text-slate-500 border border-slate-800/50'}`}>
          {isLiveData ? <Wifi size={10} /> : <WifiOff size={10} />}
          <span>{isLiveData ? 'Live Data' : 'Mock Data'}</span>
          {isLiveData && <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse ml-auto" />}
        </div>
      </div>

      <nav className="flex flex-col gap-0.5 p-2 border-b border-slate-800/50">
        {tabs.map((tab) => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`relative flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-left transition-all group overflow-hidden ${
              activeTab === tab.id
                ? 'bg-cyan-500/10 border border-cyan-500/30 text-cyan-400'
                : 'text-slate-400 hover:text-white hover:bg-slate-800/60 border border-transparent'
            }`}>
            {activeTab === tab.id && (
              <div className="absolute left-0 top-0 bottom-0 w-0.5 bg-gradient-to-b from-cyan-400 to-cyan-600 animate-pulse" />
            )}
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <span className={`relative z-10 transition-all duration-300 ${activeTab === tab.id ? 'text-cyan-400 scale-110' : 'text-slate-500 group-hover:text-slate-300 group-hover:scale-105'}`}>
              {tab.icon}
            </span>
            <div className="flex-1 min-w-0 relative z-10">
              <div className={`text-[11px] font-medium leading-none truncate transition-all duration-300 ${activeTab === tab.id ? 'text-cyan-400' : ''}`}>
                {tab.label}
              </div>
              <div className="text-[9px] text-slate-500 leading-none mt-0.5 truncate">{tab.sub}</div>
            </div>
            {activeTab === tab.id && <ChevronRight size={10} className="text-cyan-500 shrink-0 animate-pulse" />}
          </button>
        ))}
      </nav>

      {/* Quick Stats */}
      <div className="p-2 border-b border-slate-800/50">
        <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 px-1 font-['Orbitron']">System Health</div>
        <div className="space-y-1.5">
          {[
            { label: isLiveData ? 'CPU Usage' : 'Reasoning Depth', value: isLiveData ? `${systemMetrics.cpuPercent.toFixed(0)}%` : systemMetrics.reasoningDepthAvg.toFixed(1), max: 100, pct: isLiveData ? systemMetrics.cpuPercent / 100 : systemMetrics.reasoningDepthAvg / 10, color: '#06b6d4' },
            { label: isLiveData ? 'RAM Usage' : 'Mem Fragmentation', value: isLiveData ? `${systemMetrics.ramPercent.toFixed(0)}%` : `${(systemMetrics.memoryFragmentation * 100).toFixed(0)}%`, max: 100, pct: isLiveData ? systemMetrics.ramPercent / 100 : systemMetrics.memoryFragmentation, color: '#8b5cf6' },
            { label: 'Swarm Coherence', value: `${(systemMetrics.swarmCoherence * 100).toFixed(0)}%`, max: 100, pct: systemMetrics.swarmCoherence, color: '#10b981' },
            { label: 'Anomalies', value: String(systemMetrics.cognitiveAnomalies), max: 10, pct: systemMetrics.cognitiveAnomalies / 10, color: '#ef4444' },
          ].map((m) => (
            <div key={m.label} className="group">
              <div className="flex justify-between items-center mb-0.5">
                <span className="text-[9px] text-slate-500 group-hover:text-slate-400 transition-colors">{m.label}</span>
                <span className="text-[10px] font-mono font-['Space_Grotesk']" style={{ color: m.color }}>{m.value}</span>
              </div>
              <div className="h-1 bg-slate-800 rounded-full overflow-hidden relative">
                <div className="h-full rounded-full transition-all duration-700 relative"
                  style={{ width: `${Math.min(100, (m.pct ?? 0) * 100)}%`, background: `linear-gradient(90deg, ${m.color}88, ${m.color})`, boxShadow: `0 0 8px ${m.color}40` }}>
                  {(m.pct ?? 0) > 0.7 && (
                    <div className="absolute inset-0 bg-white/20 animate-pulse" />
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Cognitive Event Bus */}
      <div className="flex-1 overflow-hidden flex flex-col p-2">
        <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5 px-1 flex items-center gap-1.5 font-['Orbitron']">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_8px_#10b981]" />
          Cognitive Bus
        </div>
        <div className="flex-1 overflow-y-auto space-y-0.5">
          {recentEvents.map((evt) => (
            <div key={evt.id} className="px-1.5 py-1 rounded bg-slate-900/60 border border-slate-800/50 hover:border-slate-700/80 hover:bg-slate-900/80 transition-all duration-300 group">
              <div className="flex items-center gap-1 mb-0.5">
                <span className={`text-[9px] font-mono font-bold uppercase ${typeColor(evt.type)} font-['Space_Grotesk']`}>{evt.type.slice(0, 6)}</span>
                <span className="text-[8px] text-slate-600 ml-auto">{new Date(evt.timestamp).toLocaleTimeString()}</span>
              </div>
              <div className={`text-[9px] ${severityColor(evt.severity)} leading-tight truncate group-hover:text-slate-300 transition-colors`}>{evt.message}</div>
              <div className="text-[8px] text-slate-600 group-hover:text-slate-500 transition-colors">{evt.agentId}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Delta Lake footer */}
      <div className="p-2 border-t border-slate-800/50">
        <div className="flex items-center gap-1.5 px-1.5 py-1 rounded bg-slate-900/60 border border-slate-800/50 hover:border-violet-500/30 transition-all duration-300 group">
          <Layers size={10} className="text-violet-400 group-hover:scale-110 transition-transform duration-300" />
          <div>
            <div className="text-[9px] text-slate-500 font-['Orbitron']">
              {isLiveData ? 'System Uptime' : 'Delta Lake'}
            </div>
            <div className="text-[10px] font-mono text-violet-400 font-['Space_Grotesk']">
              {isLiveData
                ? `${Math.floor((realSystemMetrics?.uptime_secs ?? 0) / 3600)}h ${Math.floor(((realSystemMetrics?.uptime_secs ?? 0) % 3600) / 60)}m`
                : `${systemMetrics.deltaLakeSizeTB} TB`}
            </div>
          </div>
          <div className="ml-auto">
            <div className="text-[9px] text-emerald-400 flex items-center gap-0.5">
              <span className="w-1 h-1 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_6px_#10b981]" />
              <span className="font-['Space_Grotesk']">Live</span>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
}
