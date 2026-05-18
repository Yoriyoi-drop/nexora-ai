import { useState } from 'react';
import { useNexoraStore } from '../store/nexoraStore';
import { Cpu, ChevronRight, X } from 'lucide-react';

const TYPE_COLORS: Record<string, string> = {
  llm: '#06b6d4',
  vision: '#8b5cf6',
  embedding: '#10b981',
  reasoning: '#f59e0b',
  validator: '#ef4444',
};

export default function ModelPanel() {
  const [open, setOpen] = useState(false);
  const { models } = useNexoraStore();

  return (
    <>
      {/* Trigger */}
      <button
        onClick={() => setOpen(true)}
        className="fixed right-0 top-1/3 z-40 flex flex-col items-center justify-center gap-1 px-1.5 py-3 glass-dark border border-l-0 border-slate-700/50 rounded-l-lg hover:border-cyan-500/30 transition-all group"
      >
        <Cpu size={13} className="text-slate-400 group-hover:text-cyan-400 transition-colors" />
        <div className="text-[8px] text-slate-500 writing-vertical rotate-180" style={{ writingMode: 'vertical-rl' }}>MODELS</div>
        <ChevronRight size={10} className="text-slate-500 group-hover:text-cyan-400 transition-colors" />
      </button>

      {/* Slide panel */}
      {open && (
        <div className="fixed right-0 top-0 h-screen w-80 z-50 glass-dark border-l border-slate-700/50 flex flex-col overflow-hidden shadow-2xl">
          <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/50">
            <div className="flex items-center gap-2">
              <Cpu size={14} className="text-cyan-400" />
              <span className="text-sm font-semibold text-slate-200">AI Models</span>
              <span className="text-[9px] text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded-full">{models.length} active</span>
            </div>
            <button onClick={() => setOpen(false)} className="text-slate-500 hover:text-white transition-colors">
              <X size={14} />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-3 space-y-3">
            {models.map(model => (
              <div key={model.id} className="bg-slate-900/60 rounded-xl p-3 border border-slate-800/50 hover:border-slate-700 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <div className="text-xs font-bold text-slate-100 font-mono">{model.name}</div>
                    <div className="flex items-center gap-1 mt-0.5">
                      <span
                        className="text-[8px] font-bold uppercase px-1 py-0.5 rounded"
                        style={{ color: TYPE_COLORS[model.type], background: TYPE_COLORS[model.type] + '20' }}
                      >
                        {model.type}
                      </span>
                      <span className="text-[9px] text-slate-500">{model.provider}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-[10px] font-mono text-emerald-400">{model.activeAgents} agents</div>
                    <div className="text-[9px] text-slate-500">{model.quantization}</div>
                  </div>
                </div>

                {/* GPU Memory */}
                <div className="mb-2">
                  <div className="flex justify-between text-[9px] mb-0.5">
                    <span className="text-slate-500">GPU Memory</span>
                    <span className="font-mono text-slate-300">{model.gpuMemory}/{model.gpuMemoryTotal} GB</span>
                  </div>
                  <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${(model.gpuMemory / model.gpuMemoryTotal) * 100}%`,
                        background: model.gpuMemory / model.gpuMemoryTotal > 0.85
                          ? 'linear-gradient(90deg, #f59e0b, #ef4444)'
                          : 'linear-gradient(90deg, #06b6d4, #8b5cf6)',
                      }}
                    />
                  </div>
                </div>

                {/* Stats grid */}
                <div className="grid grid-cols-3 gap-1.5">
                  {[
                    { label: 'Throughput', value: `${model.throughput.toLocaleString()} t/s`, color: '#06b6d4' },
                    { label: 'Latency', value: `${model.latency}ms`, color: model.latency > 200 ? '#f59e0b' : '#10b981' },
                    { label: 'Accuracy', value: `${(model.accuracy * 100).toFixed(0)}%`, color: '#8b5cf6' },
                  ].map(s => (
                    <div key={s.label} className="bg-slate-950/50 rounded p-1.5 text-center">
                      <div className="text-[8px] text-slate-500">{s.label}</div>
                      <div className="text-[9px] font-bold font-mono" style={{ color: s.color }}>{s.value}</div>
                    </div>
                  ))}
                </div>

                {/* Temperature + accuracy bar */}
                <div className="mt-2 flex items-center gap-2">
                  <span className="text-[8px] text-slate-500">Temp</span>
                  <div className="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${model.temperature * 100}%`,
                        background: `hsl(${200 - model.temperature * 200}, 80%, 60%)`,
                      }}
                    />
                  </div>
                  <span className="text-[9px] font-mono text-slate-400">{model.temperature}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Footer summary */}
          <div className="p-3 border-t border-slate-800/50 grid grid-cols-2 gap-2">
            {[
              { label: 'Total GPU Mem', value: `${models.reduce((a, m) => a + m.gpuMemory, 0)} GB` },
              { label: 'Avg Latency', value: `${Math.floor(models.reduce((a, m) => a + m.latency, 0) / models.length)}ms` },
              { label: 'Combined T/s', value: models.reduce((a, m) => a + m.throughput, 0).toLocaleString() },
              { label: 'Active Models', value: String(models.length) },
            ].map(s => (
              <div key={s.label} className="bg-slate-900/60 rounded p-2">
                <div className="text-[8px] text-slate-500">{s.label}</div>
                <div className="text-xs font-bold font-mono text-cyan-400">{s.value}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}
