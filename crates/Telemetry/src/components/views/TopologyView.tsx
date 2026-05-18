import { useState } from 'react';
import { useNexoraStore } from '../../store/nexoraStore';
import { useCanvasRenderer } from '../../hooks/useCanvasRenderer';
import { SEMANTIC_COLORS, CLUSTER_COLORS } from '../../constants/topology';
import TemporalReplaySlider from '../TemporalReplaySlider';
import { Wifi } from 'lucide-react';

export default function TopologyView() {
  const [replayTime, setReplayTime] = useState(0);
  const [showReplay, setShowReplay] = useState(false);
  const { agents, selectedAgent, setSelectedAgent, dataSource, connectionStatus } = useNexoraStore();

  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  const { canvasRef } = useCanvasRenderer({ agents, selectedAgent });

  // Click to select agent
  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const scaleX = canvas.width / 1000;
    const scaleY = canvas.height / 680;

    let closestId: string | null = null;
    let minDist = 20;
    agents.forEach(a => {
      const dx = a.x * scaleX - mx;
      const dy = a.y * scaleY - my;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < minDist) { minDist = dist; closestId = a.id; }
    });
    setSelectedAgent(closestId);
  };

  const sel = agents.find(a => a.id === selectedAgent);

  return (
    <div className="relative w-full h-full flex flex-col">
      {/* Status header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-800/50 bg-slate-950/50 shrink-0">
        <div className="flex items-center gap-2">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">Agent Topology</div>
          {isLiveData && (
            <span className="text-[8px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-mono border border-emerald-500/30 flex items-center gap-1">
              <Wifi size={8} /> LIVE
            </span>
          )}
        </div>
      </div>
      <div className="flex-1 relative">
      <canvas ref={canvasRef} className="w-full h-full cursor-crosshair" onClick={handleClick} />

      {/* Temporal Replay Slider */}
      {showReplay && (
        <TemporalReplaySlider
          currentTime={replayTime}
          maxTime={120}
          onTimeChange={setReplayTime}
        />
      )}

      {/* Replay Toggle Button */}
      <button
        onClick={() => setShowReplay(!showReplay)}
        className="absolute top-4 left-4 px-3 py-1.5 rounded-lg glass border border-cyan-500/20 hover:border-cyan-500/40 transition-all group"
      >
        <span className="text-[10px] text-cyan-400 font-['Orbitron'] group-hover:text-cyan-300">
          {showReplay ? 'HIDE REPLAY' : 'TEMPORAL REPLAY'}
        </span>
      </button>

      {/* Agent Detail Panel */}
      {sel && (
        <div className="absolute bottom-4 right-4 w-72 glass rounded-xl p-4 border border-cyan-500/20">
          <div className="flex items-start justify-between mb-3">
            <div>
              <div className="text-sm font-bold text-white font-mono">{sel.name}</div>
              <div className="text-xs text-slate-400">{sel.role} · {sel.cluster}</div>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full" style={{ background: SEMANTIC_COLORS[sel.status as keyof typeof SEMANTIC_COLORS], boxShadow: `0 0 6px ${SEMANTIC_COLORS[sel.status as keyof typeof SEMANTIC_COLORS]}` }} />
              <span className="text-xs capitalize font-mono" style={{ color: SEMANTIC_COLORS[sel.status as keyof typeof SEMANTIC_COLORS] }}>{sel.status}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 mb-3">
            {[
              { label: 'Model', value: sel.model, color: 'text-cyan-400' },
              { label: 'Tokens/s', value: sel.tokensPerSec.toFixed(0), color: 'text-violet-400' },
              { label: 'GPU Load', value: `${sel.gpuLoad.toFixed(0)}%`, color: sel.gpuLoad > 80 ? 'text-red-400' : 'text-blue-400' },
              { label: 'Entropy', value: `${(sel.entropy * 100).toFixed(1)}%`, color: sel.entropy > 0.6 ? 'text-red-400' : 'text-emerald-400' },
              { label: 'Reason Depth', value: sel.reasoningDepth, color: 'text-teal-400' },
              { label: 'Hallucinations', value: sel.hallucinations, color: sel.hallucinations > 3 ? 'text-red-400' : 'text-slate-300' },
            ].map((m, i) => (
              <div key={i} className="bg-slate-900/60 rounded-lg p-2">
                <div className="text-[9px] text-slate-500">{m.label}</div>
                <div className={`text-sm font-bold font-mono ${m.color}`}>{m.value}</div>
              </div>
            ))}
          </div>

          {/* Entropy bar */}
          <div>
            <div className="flex justify-between text-[9px] text-slate-500 mb-0.5">
              <span>Reasoning Entropy</span>
              <span className={sel.entropy > 0.6 ? 'text-red-400' : sel.entropy > 0.4 ? 'text-amber-400' : 'text-emerald-400'}>
                {sel.entropy > 0.6 ? 'HALLUCINATION RISK' : sel.entropy > 0.4 ? 'CREATIVE' : 'STABLE'}
              </span>
            </div>
            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${sel.entropy * 100}%`,
                  background: sel.entropy > 0.6
                    ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                    : sel.entropy > 0.4
                      ? 'linear-gradient(90deg, #06b6d4, #f59e0b)'
                      : 'linear-gradient(90deg, #06b6d4, #10b981)',
                }}
              />
            </div>
          </div>

          <div className="mt-2 flex gap-1">
            <div className="flex-1 bg-slate-900/60 rounded px-2 py-1">
              <div className="text-[8px] text-slate-500">Last Event</div>
              <div className="text-[10px] font-mono text-cyan-400">{sel.lastEvent}</div>
            </div>
            <div className="bg-slate-900/60 rounded px-2 py-1">
              <div className="text-[8px] text-slate-500">Thoughts</div>
              <div className="text-[10px] font-mono text-violet-400">{sel.thoughtsPending}</div>
            </div>
          </div>
        </div>
      )}

      {/* Legend */}
      <div className="absolute top-4 right-4 glass rounded-lg p-3">
        <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-2">Cognitive States</div>
        {Object.entries(SEMANTIC_COLORS).filter(([key]) => !['active', 'idle', 'warning', 'critical', 'isolated'].includes(key)).map(([status, color]) => (
          <div key={status} className="flex items-center gap-2 mb-1">
            <span className="w-2 h-2 rounded-full" style={{ background: color as string }} />
            <span className="text-[10px] text-slate-400 capitalize">{status.replace(/([A-Z])/g, ' $1').trim()}</span>
          </div>
        ))}
        <div className="mt-2 pt-2 border-t border-slate-700/50">
          <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-1.5">Clusters</div>
          {Object.entries(CLUSTER_COLORS).map(([name, color]) => (
            <div key={name} className="flex items-center gap-2 mb-1">
              <span className="w-2 h-2 rounded" style={{ background: color }} />
              <span className="text-[10px] text-slate-400">{name}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Scan line effect */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden rounded-xl">
        <div
          className="absolute w-full h-0.5 opacity-5 bg-cyan-400"
          style={{ animation: 'scan-line 8s linear infinite', top: 0 }}
        />
      </div>
    </div>
  </div>
  );
}
