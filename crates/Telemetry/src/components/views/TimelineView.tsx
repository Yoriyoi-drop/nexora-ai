import { useNexoraStore } from '../../store/nexoraStore';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Bar, LineChart, Line, Legend
} from 'recharts';
import { GitCompare, Clock, Rewind, TrendingDown, TrendingUp, Wifi } from 'lucide-react';

const RUN_A_COLOR = '#06b6d4';
const RUN_B_COLOR = '#8b5cf6';

export default function TimelineView() {
  const { timeline, compareMode, compareRunA, compareRunB, toggleCompareMode, setTimeOffset, timeOffset, dataSource, connectionStatus } = useNexoraStore();

  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  // Simulate run B with different characteristics
  const runBData = timeline.map(snap => ({
    time: snap.label,
    // Run A
    entropyA: parseFloat((snap.entropy * 100).toFixed(1)),
    halluA: parseFloat((snap.hallucinationRate * 100).toFixed(2)),
    depthA: parseFloat(snap.reasoningDepth.toFixed(1)),
    throughputA: snap.tokenThroughput,
    fragA: parseFloat((snap.memoryFragmentation * 100).toFixed(1)),
    agentsA: snap.activeAgents,
    // Run B (simulated - slightly worse)
    entropyB: parseFloat(((snap.entropy + 0.08) * 100).toFixed(1)),
    halluB: parseFloat(((snap.hallucinationRate + 0.025) * 100).toFixed(2)),
    depthB: parseFloat((snap.reasoningDepth - 0.8).toFixed(1)),
    throughputB: snap.tokenThroughput * 0.82,
    fragB: parseFloat(((snap.memoryFragmentation + 0.05) * 100).toFixed(1)),
    agentsB: snap.activeAgents - 3,
  }));

  const latest = timeline[timeline.length - 1];

  const diffs = [
    {
      label: 'Hallucination Rate',
      runA: latest?.hallucinationRate,
      runB: latest ? latest.hallucinationRate + 0.025 : 0,
      unit: '%',
      scale: 100,
      good: 'lower',
    },
    {
      label: 'Reasoning Depth',
      runA: latest?.reasoningDepth,
      runB: latest ? latest.reasoningDepth - 0.8 : 0,
      unit: '',
      scale: 1,
      good: 'higher',
    },
    {
      label: 'Avg Entropy',
      runA: latest?.entropy,
      runB: latest ? latest.entropy + 0.08 : 0,
      unit: '%',
      scale: 100,
      good: 'lower',
    },
    {
      label: 'Token Throughput',
      runA: latest?.tokenThroughput,
      runB: latest ? latest.tokenThroughput * 0.82 : 0,
      unit: 'K t/s',
      scale: 1 / 1000,
      good: 'higher',
    },
    {
      label: 'Mem Fragmentation',
      runA: latest?.memoryFragmentation,
      runB: latest ? latest.memoryFragmentation + 0.05 : 0,
      unit: '%',
      scale: 100,
      good: 'lower',
    },
    {
      label: 'Active Agents',
      runA: latest?.activeAgents,
      runB: latest ? latest.activeAgents - 3 : 0,
      unit: '',
      scale: 1,
      good: 'higher',
    },
  ];

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-800/50 bg-slate-950/50 shrink-0">
        <div className="flex items-center gap-2">
          <Clock size={13} className="text-cyan-400" />
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">Temporal Snapshot Engine</div>
          {isLiveData && (
            <span className="text-[8px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-mono border border-emerald-500/30 flex items-center gap-1">
              <Wifi size={8} /> LIVE
            </span>
          )}
        </div>

        {/* Rewind control */}
        <div className="flex items-center gap-2 ml-4">
          <Rewind size={12} className="text-slate-500" />
          <input
            type="range"
            min={0}
            max={59}
            value={timeOffset}
            onChange={e => { setTimeOffset(Number(e.target.value)); }}
            className="w-32 accent-cyan-500"
          />
          <span className="text-[10px] font-mono text-cyan-400">
            {timeOffset === 0 ? 'NOW' : `-${timeOffset * 30}s`}
          </span>
        </div>

        <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-900/50 border border-slate-700/50">
          <div className="w-2 h-2 rounded-full" style={{ background: RUN_A_COLOR }} />
          <span className="text-[9px] font-mono text-slate-300">{compareRunA}</span>
        </div>

        <button
          onClick={toggleCompareMode}
          className={`flex items-center gap-1.5 px-2 py-1 rounded border text-[9px] transition-all ${compareMode ? 'bg-violet-500/20 border-violet-500/40 text-violet-400' : 'border-slate-700/50 text-slate-500 hover:text-white'}`}
        >
          <GitCompare size={10} />
          {compareMode ? 'Comparing' : 'Compare Runs'}
        </button>

        {compareMode && (
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-slate-900/50 border border-slate-700/50">
            <div className="w-2 h-2 rounded-full" style={{ background: RUN_B_COLOR }} />
            <span className="text-[9px] font-mono text-slate-300">{compareRunB}</span>
          </div>
        )}

        {compareMode && (
          <div className="ml-auto text-[9px] text-slate-500">
            ← cognitive git diff for AI runs →
          </div>
        )}
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* Diff Summary (compare mode) */}
        {compareMode && (
          <div className="glass rounded-xl p-4">
            <div className="flex items-center gap-2 mb-4">
              <GitCompare size={14} className="text-violet-400" />
              <span className="text-sm font-semibold text-white">Run Comparison: {compareRunA} vs {compareRunB}</span>
              <span className="ml-2 text-[9px] text-slate-500 bg-slate-800 px-2 py-0.5 rounded-full font-mono">cognitive git diff</span>
            </div>
            <div className="grid grid-cols-3 gap-3">
              {diffs.map(d => {
                const valA = d.runA ? d.runA * d.scale : 0;
                const valB = d.runB * d.scale;
                const diff = valB - valA;
                const isGood = d.good === 'higher' ? diff > 0 : diff < 0;
                const pct = valA ? ((Math.abs(diff) / valA) * 100).toFixed(1) : '0';
                return (
                  <div key={d.label} className="bg-slate-900/60 rounded-xl p-3 border border-slate-800/50">
                    <div className="text-[9px] text-slate-500 mb-2">{d.label}</div>
                    <div className="flex items-center gap-2 mb-1">
                      <div className="flex items-center gap-1">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ background: RUN_A_COLOR }} />
                        <span className="text-[10px] font-mono text-slate-300">{valA.toFixed(d.unit === 'K t/s' ? 1 : 2)}{d.unit}</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ background: RUN_B_COLOR }} />
                        <span className="text-[10px] font-mono text-slate-300">{valB.toFixed(d.unit === 'K t/s' ? 1 : 2)}{d.unit}</span>
                      </div>
                    </div>
                    <div className={`flex items-center gap-1 text-[10px] font-bold ${isGood ? 'text-emerald-400' : 'text-red-400'}`}>
                      {isGood ? <TrendingUp size={10} /> : <TrendingDown size={10} />}
                      {diff > 0 ? '+' : ''}{diff.toFixed(d.unit === 'K t/s' ? 1 : 2)}{d.unit}
                      <span className="text-[9px] font-normal">({pct}%)</span>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Entropy timeline */}
        <div className="glass rounded-xl p-4">
          <div className="text-xs font-semibold text-slate-200 mb-3">Entropy Timeline{compareMode ? ` — ${compareRunA} vs ${compareRunB}` : ''}</div>
          <ResponsiveContainer width="100%" height={150}>
            <AreaChart data={runBData}>
              <defs>
                <linearGradient id="eA" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={RUN_A_COLOR} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={RUN_A_COLOR} stopOpacity={0} />
                </linearGradient>
                <linearGradient id="eB" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor={RUN_B_COLOR} stopOpacity={0.3} />
                  <stop offset="95%" stopColor={RUN_B_COLOR} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
              <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} interval={8} />
              <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} />
              <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }} />
              <Legend wrapperStyle={{ fontSize: 9, color: '#64748b' }} />
              <Area type="monotone" dataKey="entropyA" name={compareRunA} stroke={RUN_A_COLOR} fill="url(#eA)" strokeWidth={1.5} dot={false} />
              {compareMode && <Area type="monotone" dataKey="entropyB" name={compareRunB} stroke={RUN_B_COLOR} fill="url(#eB)" strokeWidth={1.5} dot={false} />}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Two column: hallucination + throughput */}
        <div className="grid grid-cols-2 gap-4">
          <div className="glass rounded-xl p-4">
            <div className="text-xs font-semibold text-slate-200 mb-3">Hallucination Rate %</div>
            <ResponsiveContainer width="100%" height={130}>
              <AreaChart data={runBData}>
                <defs>
                  <linearGradient id="hA" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="hB" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f97316" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} interval={8} />
                <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }} />
                <Area type="monotone" dataKey="halluA" name={compareRunA} stroke="#ef4444" fill="url(#hA)" strokeWidth={1.5} dot={false} />
                {compareMode && <Area type="monotone" dataKey="halluB" name={compareRunB} stroke="#f97316" fill="url(#hB)" strokeWidth={1.5} dot={false} />}
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="glass rounded-xl p-4">
            <div className="text-xs font-semibold text-slate-200 mb-3">Token Throughput</div>
            <ResponsiveContainer width="100%" height={130}>
              <LineChart data={runBData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} interval={8} />
                <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }} />
                <Line type="monotone" dataKey="throughputA" name={compareRunA} stroke={RUN_A_COLOR} strokeWidth={1.5} dot={false} />
                {compareMode && <Line type="monotone" dataKey="throughputB" name={compareRunB} stroke={RUN_B_COLOR} strokeWidth={1.5} dot={false} />}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Reasoning depth + Active agents */}
        <div className="grid grid-cols-2 gap-4">
          <div className="glass rounded-xl p-4">
            <div className="text-xs font-semibold text-slate-200 mb-3">Reasoning Depth</div>
            <ResponsiveContainer width="100%" height={110}>
              <LineChart data={runBData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} interval={8} />
                <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }} />
                <Line type="monotone" dataKey="depthA" name={compareRunA} stroke="#10b981" strokeWidth={1.5} dot={false} />
                {compareMode && <Line type="monotone" dataKey="depthB" name={compareRunB} stroke={RUN_B_COLOR} strokeWidth={1.5} dot={false} strokeDasharray="4 2" />}
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="glass rounded-xl p-4">
            <div className="text-xs font-semibold text-slate-200 mb-3">Active Agent Count</div>
            <ResponsiveContainer width="100%" height={110}>
              <BarChart data={runBData.filter((_, i) => i % 5 === 0)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
                <XAxis dataKey="time" tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} />
                <YAxis tick={{ fill: '#475569', fontSize: 8 }} tickLine={false} axisLine={false} domain={[35, 50]} />
                <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 10 }} />
                <Bar dataKey="agentsA" name={compareRunA} fill={RUN_A_COLOR} fillOpacity={0.7} radius={[2, 2, 0, 0]} />
                {compareMode && <Bar dataKey="agentsB" name={compareRunB} fill={RUN_B_COLOR} fillOpacity={0.7} radius={[2, 2, 0, 0]} />}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Snapshot cards */}
        <div className="glass rounded-xl p-4">
          <div className="text-xs font-semibold text-slate-200 mb-3">Temporal Snapshots (500ms intervals)</div>
          <div className="flex gap-2 overflow-x-auto pb-2">
            {timeline.slice(-12).reverse().map((snap, i) => (
              <div key={i} className="shrink-0 w-36 bg-slate-900/60 rounded-lg p-2.5 border border-slate-800/50 hover:border-cyan-500/30 transition-colors cursor-pointer">
                <div className="text-[9px] text-slate-500 mb-1.5 font-mono">{snap.label}</div>
                <div className="space-y-0.5">
                  <div className="flex justify-between text-[9px]">
                    <span className="text-slate-500">Entropy</span>
                    <span className="font-mono text-amber-400">{(snap.entropy * 100).toFixed(0)}%</span>
                  </div>
                  <div className="flex justify-between text-[9px]">
                    <span className="text-slate-500">Halluc</span>
                    <span className="font-mono text-red-400">{(snap.hallucinationRate * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between text-[9px]">
                    <span className="text-slate-500">Depth</span>
                    <span className="font-mono text-cyan-400">{snap.reasoningDepth.toFixed(1)}</span>
                  </div>
                  <div className="flex justify-between text-[9px]">
                    <span className="text-slate-500">Agents</span>
                    <span className="font-mono text-emerald-400">{snap.activeAgents}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
