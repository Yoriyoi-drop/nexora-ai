import { useEffect, useRef, useCallback } from 'react';
import { useNexoraStore } from '../../store/nexoraStore';
import { Server, Database, Radio, Cpu, Shield, Globe, Wifi } from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts';

const NODE_ICONS: Record<string, React.ReactNode> = {
  'gpu-cluster': <Cpu size={16} />,
  'nats': <Radio size={16} />,
  'delta-lake': <Database size={16} />,
  'vector-db': <Database size={16} />,
  'orchestrator': <Server size={16} />,
  'gateway': <Globe size={16} />,
};

const NODE_COLORS: Record<string, string> = {
  healthy: '#10b981',
  degraded: '#f59e0b',
  critical: '#ef4444',
};

const INFRA_POSITIONS: Record<string, { x: number; y: number }> = {
  'inf-1': { x: 0.15, y: 0.25 },
  'inf-2': { x: 0.15, y: 0.65 },
  'inf-3': { x: 0.42, y: 0.45 },
  'inf-4': { x: 0.68, y: 0.65 },
  'inf-5': { x: 0.68, y: 0.28 },
  'inf-6': { x: 0.85, y: 0.45 },
  'inf-7': { x: 0.42, y: 0.15 },
};

export default function InfrastructureView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);
  const { infraNodes, systemMetrics, timeline, dataSource, connectionStatus, realSystemMetrics } = useNexoraStore();

  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    timeRef.current += 0.016;
    const t = timeRef.current;

    ctx.clearRect(0, 0, W, H);

    ctx.strokeStyle = 'rgba(30,58,95,0.1)';
    ctx.lineWidth = 1;
    for (let x = 0; x < W; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(W, x); ctx.stroke();
    }
    for (let x = 0; x < W; x += 50) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y < H; y += 50) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    infraNodes.forEach(node => {
      const pos = INFRA_POSITIONS[node.id];
      if (!pos) return;
      const nx = pos.x * W;
      const ny = pos.y * H;

      node.connections.forEach(connId => {
        const connPos = INFRA_POSITIONS[connId];
        if (!connPos) return;
        const cx = connPos.x * W;
        const cy = connPos.y * H;

        const dashOffset = -t * 25;
        ctx.save();
        ctx.strokeStyle = NODE_COLORS[node.status] + '40';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 4]);
        ctx.lineDashOffset = dashOffset;
        ctx.beginPath();
        ctx.moveTo(nx, ny);
        ctx.lineTo(cx, cy);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        const pct = (t * 0.5 + node.load) % 1;
        const px = nx + (cx - nx) * pct;
        const py = ny + (cy - ny) * pct;
        ctx.beginPath();
        ctx.arc(px, py, 3, 0, Math.PI * 2);
        ctx.fillStyle = NODE_COLORS[node.status] + 'cc';
        ctx.fill();
      });
    });

    infraNodes.forEach(node => {
      const pos = INFRA_POSITIONS[node.id];
      if (!pos) return;
      const nx = pos.x * W;
      const ny = pos.y * H;
      const color = NODE_COLORS[node.status];
      const r = 36;

      ctx.save();
      ctx.beginPath();
      ctx.arc(nx, ny, r + 8, -Math.PI / 2, -Math.PI / 2 + node.load * Math.PI * 2);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.lineCap = 'round';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(nx, ny, r + 8, 0, Math.PI * 2);
      ctx.strokeStyle = color + '20';
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.restore();

      if (node.status === 'critical') {
        const pulseR = r + 12 + Math.sin(t * 6) * 4;
        ctx.beginPath();
        ctx.arc(nx, ny, pulseR, 0, Math.PI * 2);
        ctx.strokeStyle = '#ef444430';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      const glow = ctx.createRadialGradient(nx, ny, 0, nx, ny, r * 1.5);
      glow.addColorStop(0, color + '30');
      glow.addColorStop(1, 'transparent');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(nx, ny, r * 1.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#0f172a';
      ctx.beginPath();
      ctx.arc(nx, ny, r, 0, Math.PI * 2);
      ctx.fill();

      ctx.strokeStyle = color + '80';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.fillStyle = color;
      ctx.font = 'bold 9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(node.label, nx, ny + r + 16);

      ctx.fillStyle = color;
      ctx.font = '8px JetBrains Mono, monospace';
      ctx.fillText(`${(node.load * 100).toFixed(0)}% load`, nx, ny + r + 26);
      ctx.fillText(node.status.toUpperCase(), nx, ny + r + 36);
    });

    animRef.current = requestAnimationFrame(draw);
  }, [infraNodes]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (rect) { canvas.width = rect.width; canvas.height = rect.height; }
    };
    resize();
    window.addEventListener('resize', resize);
    animRef.current = requestAnimationFrame(draw);
    return () => {
      window.removeEventListener('resize', resize);
      cancelAnimationFrame(animRef.current);
    };
  }, [draw]);

  const throughputData = timeline.slice(-20).map(s => ({
    time: s.label,
    tokens: s.tokenThroughput,
  }));

  const realCpuHistory = timeline.slice(-20).map((_, i) => ({
    time: `T-${20 - i}`,
    value: systemMetrics.cpuPercent * (0.8 + Math.random() * 0.4),
  }));

  return (
    <div className="w-full h-full flex flex-col overflow-hidden">
      {/* Status header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-800/50 bg-slate-950/50 shrink-0">
        <div className="flex items-center gap-2">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">Infrastructure Topology</div>
          {isLiveData && (
            <span className="text-[8px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-mono border border-emerald-500/30 flex items-center gap-1">
              <Wifi size={8} /> LIVE
            </span>
          )}
        </div>
        <div className="flex items-center gap-2 ml-4">
          {infraNodes.map(n => (
            <div key={n.id} className="flex items-center gap-1 px-2 py-0.5 rounded bg-slate-900/50 border border-slate-800/50">
              <span className="w-1.5 h-1.5 rounded-full" style={{ background: NODE_COLORS[n.status] }} />
              <span className="text-[9px] text-slate-400">{n.label}</span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex-1 flex overflow-hidden">
        {/* Main canvas */}
        <div className="flex-1 relative">
          <canvas ref={canvasRef} className="w-full h-full" />

          {/* Critical alert - hide if live and no critical */}
          <div className="absolute top-4 left-4 glass rounded-lg px-3 py-2 border border-red-500/30">
            <div className="flex items-center gap-1.5 text-[10px] text-red-400 mb-1">
              <Shield size={11} />
              <span className="uppercase tracking-widest">Status</span>
            </div>
            {isLiveData ? (
              <>
                <div className="text-[9px] text-slate-400">CPU: {systemMetrics.cpuPercent.toFixed(0)}%</div>
                <div className="text-[9px] text-slate-400">RAM: {systemMetrics.ramPercent.toFixed(0)}%</div>
                <div className="text-[9px] text-slate-400">Processes: {realSystemMetrics?.processes ?? '-'}</div>
              </>
            ) : (
              <>
                <div className="text-[9px] text-slate-400">API Gateway: 95% capacity</div>
                <div className="text-[9px] text-slate-400">GPU Cluster B: Degraded</div>
              </>
            )}
          </div>
        </div>

        {/* Right panel: node details */}
        <div className="w-64 border-l border-slate-800/50 bg-slate-950/50 flex flex-col overflow-hidden">
          <div className="p-3 border-b border-slate-800/50">
            <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-2">Node Status</div>
            <div className="space-y-2">
              {infraNodes.map(node => (
                <div key={node.id} className="bg-slate-900/60 rounded-lg p-2.5 border border-slate-800/50">
                  <div className="flex items-center gap-2 mb-1.5">
                    <span className="text-slate-500" style={{ color: NODE_COLORS[node.status] }}>
                      {NODE_ICONS[node.type]}
                    </span>
                    <div className="flex-1 min-w-0">
                      <div className="text-[10px] font-mono text-slate-200 truncate">{node.label}</div>
                      <div className="text-[8px] text-slate-500">{node.type}</div>
                    </div>
                    <span
                      className="text-[8px] font-bold uppercase px-1 py-0.5 rounded"
                      style={{
                        color: NODE_COLORS[node.status],
                        background: NODE_COLORS[node.status] + '20',
                      }}
                    >
                      {node.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="text-[8px] text-slate-500">Load</span>
                    <div className="flex-1 h-1 bg-slate-800 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-700"
                        style={{
                          width: `${node.load * 100}%`,
                          background: NODE_COLORS[node.status],
                        }}
                      />
                    </div>
                    <span className="text-[8px] font-mono" style={{ color: NODE_COLORS[node.status] }}>
                      {(node.load * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Throughput chart */}
          <div className="p-3 flex-1">
            <div className="text-[9px] text-slate-500 uppercase tracking-widest mb-2">
              {isLiveData ? 'System Metrics' : 'Token Throughput'}
            </div>
            {isLiveData ? (
              <div className="space-y-2">
                <ResponsiveContainer width="100%" height={80}>
                  <AreaChart data={realCpuHistory}>
                    <defs>
                      <linearGradient id="cpuGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
                    <XAxis dataKey="time" tick={false} />
                    <YAxis tick={{ fill: '#475569', fontSize: 7 }} tickLine={false} axisLine={false} domain={[0, 100]} />
                    <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 9 }} />
                    <Area type="monotone" dataKey="value" stroke="#06b6d4" fill="url(#cpuGrad)" strokeWidth={1.5} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <ResponsiveContainer width="100%" height={100}>
                <AreaChart data={throughputData}>
                  <defs>
                    <linearGradient id="tpGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e3a5f20" />
                  <XAxis dataKey="time" tick={false} />
                  <YAxis tick={{ fill: '#475569', fontSize: 7 }} tickLine={false} axisLine={false} />
                  <Tooltip contentStyle={{ background: '#0f172a', border: '1px solid #1e3a5f', borderRadius: 8, fontSize: 9 }} />
                  <Area type="monotone" dataKey="tokens" stroke="#06b6d4" fill="url(#tpGrad)" strokeWidth={1.5} dot={false} />
                </AreaChart>
              </ResponsiveContainer>
            )}

            {/* System metrics */}
            <div className="mt-3 space-y-1.5">
              {(() => {
                const items = isLiveData ? [
                  { label: 'CPU Usage', value: `${systemMetrics.cpuPercent.toFixed(1)}%`, color: systemMetrics.cpuPercent > 85 ? '#ef4444' : '#06b6d4' },
                  { label: 'RAM Usage', value: `${systemMetrics.ramPercent.toFixed(1)}%`, color: systemMetrics.ramPercent > 85 ? '#ef4444' : '#8b5cf6' },
                  { label: 'RAM Used', value: `${systemMetrics.ramUsedGb.toFixed(1)} GB`, color: '#10b981' },
                  { label: 'Processes', value: String(realSystemMetrics?.processes ?? '-'), color: '#06b6d4' },
                ] : [
                  { label: 'NATS Event/s', value: systemMetrics.natsEventRate.toLocaleString(), color: '#06b6d4' },
                  { label: 'Delta Lake', value: `${systemMetrics.deltaLakeSizeTB} TB`, color: '#8b5cf6' },
                  { label: 'GPU Cluster Util', value: `${(systemMetrics.gpuUtilization * 100).toFixed(0)}%`, color: systemMetrics.gpuUtilization > 0.85 ? '#ef4444' : '#10b981' },
                ];
                return items.map(m => (
                  <div key={m.label} className="flex justify-between items-center">
                    <span className="text-[9px] text-slate-500">{m.label}</span>
                    <span className="text-[10px] font-mono font-bold" style={{ color: m.color }}>{m.value}</span>
                  </div>
                ));
              })()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
