import { useState, useEffect } from 'react';
import { useNexoraStore } from '../store/nexoraStore';
import { alertEngine } from '../services/alertEngine';
import {
  Activity, Cpu, Zap, Brain, AlertTriangle,
  Play, Pause, SkipBack, GitCompare, Settings, Bell,
  Wifi, WifiOff, Database
} from 'lucide-react';

export default function TopBar() {
  const { systemMetrics, isLive, setIsLive, timeOffset, setTimeOffset, toggleCompareMode, compareMode, dataSource, connectionStatus } = useNexoraStore();
  const [time, setTime] = useState(new Date());
  const [alertCount, setAlertCount] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setTime(new Date()), 1000);
    const unsub = alertEngine.subscribe((a) => {
      setAlertCount(a.filter(alert => !alert.acknowledged).length);
    });
    return () => { clearInterval(t); unsub(); };
  }, []);

  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  const metrics = [
    {
      icon: <Brain size={13} />,
      label: 'Active Agents',
      value: `${systemMetrics.activeAgents}/${systemMetrics.totalAgents}`,
      color: 'text-cyan-400',
      bg: 'bg-cyan-400/10',
    },
    {
      icon: <Zap size={13} />,
      label: 'Token/s',
      value: systemMetrics.totalTokensPerSec.toLocaleString(),
      color: 'text-violet-400',
      bg: 'bg-violet-400/10',
    },
    {
      icon: <Activity size={13} />,
      label: 'Entropy Avg',
      value: (systemMetrics.avgEntropy * 100).toFixed(1) + '%',
      color: systemMetrics.avgEntropy > 0.6 ? 'text-red-400' : systemMetrics.avgEntropy > 0.4 ? 'text-amber-400' : 'text-emerald-400',
      bg: systemMetrics.avgEntropy > 0.6 ? 'bg-red-400/10' : systemMetrics.avgEntropy > 0.4 ? 'bg-amber-400/10' : 'bg-emerald-400/10',
    },
    {
      icon: <Cpu size={13} />,
      label: isLiveData ? 'CPU' : 'GPU Util',
      value: isLiveData
        ? `${systemMetrics.cpuPercent.toFixed(0)}%`
        : `${(systemMetrics.gpuUtilization * 100).toFixed(0)}%`,
      color: isLiveData
        ? (systemMetrics.cpuPercent > 90 ? 'text-red-400' : 'text-blue-400')
        : (systemMetrics.gpuUtilization > 0.85 ? 'text-red-400' : 'text-blue-400'),
      bg: isLiveData
        ? (systemMetrics.cpuPercent > 90 ? 'bg-red-400/10' : 'bg-blue-400/10')
        : (systemMetrics.gpuUtilization > 0.85 ? 'bg-red-400/10' : 'bg-blue-400/10'),
    },
    {
      icon: <Database size={13} />,
      label: isLiveData ? 'RAM' : 'NATS/s',
      value: isLiveData
        ? `${systemMetrics.ramPercent.toFixed(0)}%`
        : systemMetrics.natsEventRate.toLocaleString(),
      color: isLiveData
        ? (systemMetrics.ramPercent > 90 ? 'text-red-400' : 'text-teal-400')
        : 'text-teal-400',
      bg: isLiveData
        ? (systemMetrics.ramPercent > 90 ? 'bg-red-400/10' : 'bg-teal-400/10')
        : 'bg-teal-400/10',
    },
    {
      icon: <AlertTriangle size={13} />,
      label: 'Hallucination',
      value: (systemMetrics.hallucinationRate * 100).toFixed(2) + '%',
      color: systemMetrics.hallucinationRate > 0.05 ? 'text-red-400' : 'text-emerald-400',
      bg: systemMetrics.hallucinationRate > 0.05 ? 'bg-red-400/10' : 'bg-emerald-400/10',
    },
  ];

  return (
    <header className="flex items-center justify-between px-4 py-2 border-b border-slate-800/80 glass-dark shrink-0" style={{ height: 52 }}>
      {/* Logo */}
      <div className="flex items-center gap-3 shrink-0">
        <div className="relative">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500 to-violet-600 flex items-center justify-center">
            <Brain size={16} className="text-white" />
          </div>
          {isLive && (
            <span className="absolute -top-0.5 -right-0.5 w-2.5 h-2.5 bg-emerald-400 rounded-full border-2 border-slate-950 animate-pulse" />
          )}
        </div>
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-bold tracking-widest text-white font-mono">NEXORA</span>
            {isLiveData && (
              <span className="text-[8px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-mono border border-emerald-500/30 flex items-center gap-1">
                <Wifi size={8} /> LIVE
              </span>
            )}
            {!isLiveData && connectionStatus === 'disconnected' && (
              <span className="text-[8px] bg-slate-700/30 text-slate-500 px-1.5 py-0.5 rounded-full font-mono flex items-center gap-1">
                <WifiOff size={8} /> MOCK
              </span>
            )}
          </div>
          <div className="text-[9px] text-slate-500 tracking-widest uppercase">Cognition Observatory v2.4</div>
        </div>
      </div>

      {/* Metrics bar */}
      <div className="flex items-center gap-1 overflow-hidden">
        {metrics.map((m, i) => (
          <div key={i} className={`flex items-center gap-1.5 px-2 py-1 rounded ${m.bg} border border-white/5`}>
            <span className={m.color}>{m.icon}</span>
            <div>
              <div className={`text-xs font-bold font-mono leading-none ${m.color}`}>{m.value}</div>
              <div className="text-[9px] text-slate-500 leading-none mt-0.5">{m.label}</div>
            </div>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 shrink-0">
        {/* Coherence indicator */}
        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-800/60 border border-slate-700/50">
          <div className="text-[9px] text-slate-500">SWARM COHERENCE</div>
          <div className="w-16 h-1.5 rounded-full bg-slate-700 overflow-hidden">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-500 to-violet-500 transition-all duration-700"
              style={{ width: `${systemMetrics.swarmCoherence * 100}%` }}
            />
          </div>
          <div className="text-[10px] font-mono text-cyan-400">{(systemMetrics.swarmCoherence * 100).toFixed(0)}%</div>
        </div>

        {/* Timeline controls */}
        <div className="flex items-center gap-1 bg-slate-800/60 rounded border border-slate-700/50 p-0.5">
          <button
            className="p-1.5 rounded hover:bg-slate-700 transition-colors text-slate-400 hover:text-white"
            onClick={() => { setIsLive(false); setTimeOffset(timeOffset - 5); }}
            title="Rewind 5s"
          >
            <SkipBack size={12} />
          </button>
          <button
            className={`p-1.5 rounded transition-colors ${isLive ? 'bg-emerald-500/20 text-emerald-400' : 'hover:bg-slate-700 text-slate-400 hover:text-white'}`}
            onClick={() => setIsLive(!isLive)}
            title={isLive ? 'Pause stream' : 'Resume live'}
          >
            {isLive ? <Pause size={12} /> : <Play size={12} />}
          </button>
        </div>

        {/* Compare toggle */}
        <button
          className={`flex items-center gap-1.5 px-2 py-1.5 rounded text-xs border transition-all ${compareMode ? 'bg-violet-500/20 border-violet-500/50 text-violet-400' : 'border-slate-700/50 text-slate-400 hover:text-white hover:border-slate-600'}`}
          onClick={toggleCompareMode}
        >
          <GitCompare size={12} />
          <span className="text-[10px]">Compare</span>
        </button>

        {/* Alerts */}
        <button className="relative p-1.5 rounded border border-slate-700/50 text-slate-400 hover:text-amber-400 transition-colors">
          <Bell size={13} />
          {alertCount > 0 && (
            <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 rounded-full text-[9px] text-white flex items-center justify-center font-bold">
              {alertCount}
            </span>
          )}
        </button>

        <button className="p-1.5 rounded border border-slate-700/50 text-slate-400 hover:text-white transition-colors">
          <Settings size={13} />
        </button>

        {/* Clock */}
        <div className="text-[10px] font-mono text-slate-500 border-l border-slate-700 pl-2">
          <div className="text-slate-300">{time.toLocaleTimeString()}</div>
          <div>{time.toLocaleDateString()}</div>
        </div>
      </div>
    </header>
  );
}
