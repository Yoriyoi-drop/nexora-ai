import { useEffect, useRef, useCallback } from 'react';
import { useNexoraStore } from '../../store/nexoraStore';
import { Wifi } from 'lucide-react';

interface Particle {
  x: number;
  y: number;
  vx: number;
  vy: number;
  life: number;
  maxLife: number;
  color: string;
  size: number;
  trail: { x: number; y: number }[];
  bottleneck: boolean;
}

const FLOW_NODES = [
  { id: 'omnis', label: 'Omnis', x: 0.12, y: 0.25, color: '#06b6d4' },
  { id: 'vortex', label: 'Vortex', x: 0.12, y: 0.55, color: '#8b5cf6' },
  { id: 'kronos', label: 'Kronos', x: 0.12, y: 0.78, color: '#3b82f6' },
  { id: 'nexum', label: 'Nexum', x: 0.4, y: 0.4, color: '#f59e0b' },
  { id: 'axiom', label: 'Axiom', x: 0.62, y: 0.45, color: '#10b981' },
  { id: 'aether', label: 'Aether', x: 0.4, y: 0.72, color: '#ec4899' },
  { id: 'cipher', label: 'Cipher', x: 0.82, y: 0.28, color: '#06b6d4' },
  { id: 'spectra', label: 'Spectra', x: 0.82, y: 0.62, color: '#f59e0b' },
  { id: 'swift', label: 'Swift', x: 0.12, y: 0.43, color: '#10b981' },
];

const FLOW_EDGES = [
  { from: 'omnis', to: 'nexum', volume: 4200, bottleneck: false },
  { from: 'vortex', to: 'nexum', volume: 3800, bottleneck: false },
  { from: 'swift', to: 'kronos', volume: 2100, bottleneck: true },
  { from: 'aether', to: 'nexum', volume: 5600, bottleneck: false },
  { from: 'nexum', to: 'axiom', volume: 8900, bottleneck: true },
  { from: 'axiom', to: 'cipher', volume: 3200, bottleneck: false },
  { from: 'axiom', to: 'spectra', volume: 4100, bottleneck: false },
  { from: 'kronos', to: 'axiom', volume: 1800, bottleneck: false },
];

export default function TokenFlowView() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);
  const particlesRef = useRef<Particle[]>([]);
  const { tokenFlows, dataSource, connectionStatus } = useNexoraStore();

  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  const spawnParticle = useCallback((edge: typeof FLOW_EDGES[0], W: number, H: number) => {
    const fromNode = FLOW_NODES.find(n => n.id === edge.from);
    const toNode = FLOW_NODES.find(n => n.id === edge.to);
    if (!fromNode || !toNode) return;

    const fx = fromNode.x * W;
    const fy = fromNode.y * H;
    const tx = toNode.x * W;
    const ty = toNode.y * H;

    const spread = 3;
    const speed = 1.5 + Math.random() * 1.5;
    const dx = tx - fx;
    const dy = ty - fy;
    const dist = Math.sqrt(dx * dx + dy * dy);
    const life = dist / speed;

    const color = edge.bottleneck
      ? `hsl(${0 + Math.random() * 30}, 90%, 60%)`
      : fromNode.color;

    particlesRef.current.push({
      x: fx + (Math.random() - 0.5) * spread,
      y: fy + (Math.random() - 0.5) * spread,
      vx: (dx / dist) * speed,
      vy: (dy / dist) * speed,
      life,
      maxLife: life,
      color,
      size: edge.bottleneck ? 2.5 + Math.random() : 1.5 + Math.random() * 1,
      trail: [],
      bottleneck: edge.bottleneck,
    });
  }, []);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    timeRef.current += 0.016;
    const t = timeRef.current;

    // Fade trail
    ctx.fillStyle = 'rgba(3, 7, 18, 0.25)';
    ctx.fillRect(0, 0, W, H);

    // Draw edges (tubes)
    FLOW_EDGES.forEach(edge => {
      const fromNode = FLOW_NODES.find(n => n.id === edge.from);
      const toNode = FLOW_NODES.find(n => n.id === edge.to);
      if (!fromNode || !toNode) return;

      const fx = fromNode.x * W;
      const fy = fromNode.y * H;
      const tx = toNode.x * W;
      const ty = toNode.y * H;

      // Edge tube
      ctx.save();
      ctx.strokeStyle = edge.bottleneck ? '#ef444420' : '#1e3a5f40';
      ctx.lineWidth = edge.bottleneck ? (4 + Math.sin(t * 5) * 1) : (2 + edge.volume / 5000);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(fx, fy);

      // Curve through midpoint
      const mx = (fx + tx) / 2 + (Math.random() - 0.5) * 0;
      const my = (fy + ty) / 2;
      ctx.quadraticCurveTo(mx, my, tx, ty);
      ctx.stroke();

      // Volume label
      const lx = (fx + tx) / 2;
      const ly = (fy + ty) / 2 - 8;
      ctx.fillStyle = edge.bottleneck ? '#ef4444' : '#475569';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`${(edge.volume / 1000).toFixed(1)}K t/s`, lx, ly);
      ctx.restore();
    });

    // Spawn new particles
    FLOW_EDGES.forEach(edge => {
      const spawnRate = Math.ceil(edge.volume / 2000);
      for (let s = 0; s < spawnRate; s++) {
        if (Math.random() < 0.3) spawnParticle(edge, W, H);
      }
    });

    // Update & draw particles
    particlesRef.current = particlesRef.current.filter(p => p.life > 0);
    particlesRef.current.forEach(p => {
      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > 8) p.trail.shift();

      p.x += p.vx;
      p.y += p.vy;
      p.life -= 1;

      const progress = 1 - p.life / p.maxLife;
      const alpha = progress < 0.1 ? progress * 10 : progress > 0.85 ? (1 - progress) * 6.67 : 1;

      // Draw trail
      if (p.trail.length > 1) {
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(p.trail[0].x, p.trail[0].y);
        p.trail.forEach(pt => ctx.lineTo(pt.x, pt.y));
        ctx.strokeStyle = p.color + Math.floor(alpha * 40).toString(16).padStart(2, '0');
        ctx.lineWidth = p.size * 0.5;
        ctx.stroke();
        ctx.restore();
      }

      // Particle glow
      const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.size * 4);
      glow.addColorStop(0, p.color + Math.floor(alpha * 180).toString(16).padStart(2, '0'));
      glow.addColorStop(1, 'transparent');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size * 4, 0, Math.PI * 2);
      ctx.fill();

      // Core
      ctx.fillStyle = p.color + Math.floor(alpha * 255).toString(16).padStart(2, '0');
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.fill();

      // Bottleneck pulse
      if (p.bottleneck) {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size * 2 + Math.sin(t * 10) * 2, 0, Math.PI * 2);
        ctx.strokeStyle = '#ef4444' + Math.floor(alpha * 100).toString(16).padStart(2, '0');
        ctx.lineWidth = 0.5;
        ctx.stroke();
      }
    });

    // Draw nodes
    FLOW_NODES.forEach(node => {
      const nx = node.x * W;
      const ny = node.y * H;
      const r = 22;

      // Pulse ring
      const pulseR = r + 5 + Math.sin(t * 2 + node.x * 10) * 3;
      ctx.beginPath();
      ctx.arc(nx, ny, pulseR, 0, Math.PI * 2);
      ctx.strokeStyle = node.color + '30';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Node bg
      ctx.fillStyle = 'rgba(15,23,42,0.9)';
      ctx.beginPath();
      ctx.arc(nx, ny, r, 0, Math.PI * 2);
      ctx.fill();

      // Node border
      ctx.strokeStyle = node.color;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Glow
      const ng = ctx.createRadialGradient(nx, ny, 0, nx, ny, r);
      ng.addColorStop(0, node.color + '20');
      ng.addColorStop(1, 'transparent');
      ctx.fillStyle = ng;
      ctx.beginPath();
      ctx.arc(nx, ny, r, 0, Math.PI * 2);
      ctx.fill();

      // Label
      ctx.fillStyle = '#e2e8f0';
      ctx.font = 'bold 9px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      const lines = node.label.split('-');
      lines.forEach((line, i) => {
        ctx.fillText(line, nx, ny + (i - (lines.length - 1) / 2) * 11);
      });
    });

    animRef.current = requestAnimationFrame(draw);
  }, [spawnParticle]);

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

  return (
    <div className="relative w-full h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-800/50 bg-slate-950/50 shrink-0">
        <div className="flex items-center gap-2">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">Token Particle Simulation</div>
          {isLiveData && (
            <span className="text-[8px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-mono border border-emerald-500/30 flex items-center gap-1">
              <Wifi size={8} /> LIVE
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 ml-4">
          {tokenFlows.map((flow, i) => (
            <div key={i} className={`flex items-center gap-1.5 px-2 py-0.5 rounded text-[10px] ${flow.bottleneck ? 'text-red-400 bg-red-400/10' : 'text-slate-400 bg-slate-800/50'}`}>
              <span className={`w-1.5 h-1.5 rounded-full ${flow.bottleneck ? 'bg-red-400' : 'bg-slate-500'}`} />
              {flow.source}→{flow.target}: {(flow.volume / 1000).toFixed(1)}K t/s
              {flow.bottleneck && ' ⚡'}
            </div>
          ))}
        </div>
      </div>
      <div className="flex-1 relative overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full" />

        {/* Bottleneck alert */}
        <div className="absolute bottom-4 left-4 glass rounded-lg px-3 py-2 border border-red-500/20">
          <div className="text-[9px] text-red-400 uppercase tracking-widest mb-1 flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse" />
            Bottleneck Detected
          </div>
          <div className="text-[10px] text-slate-400">Nexum → Axiom: 89% capacity</div>
          <div className="text-[10px] text-slate-400">Swift → Kronos: Loop detected</div>
        </div>

        {/* Legend */}
        <div className="absolute bottom-4 right-4 glass rounded-lg p-3">
          <div className="text-[9px] text-slate-500 mb-2 uppercase tracking-widest">Particle Legend</div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 rounded-full bg-cyan-400" />
            <span className="text-[10px] text-slate-400">Normal token flow</span>
          </div>
          <div className="flex items-center gap-2 mb-1">
            <div className="w-3 h-3 rounded-full bg-red-400" />
            <span className="text-[10px] text-slate-400">Bottleneck / overload</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-5 h-0.5 bg-slate-600" style={{ borderTop: '1px dashed #475569' }} />
            <span className="text-[10px] text-slate-400">Edge tube (volume ∝ width)</span>
          </div>
        </div>
      </div>
    </div>
  );
}
