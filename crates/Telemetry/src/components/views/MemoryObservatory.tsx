import { useEffect, useRef, useCallback } from 'react';
import { useNexoraStore } from '../../store/nexoraStore';
import { Wifi } from 'lucide-react';

const TYPE_COLORS: Record<string, { fill: string; glow: string }> = {
  episodic: { fill: '#06b6d4', glow: '#06b6d460' },
  semantic: { fill: '#8b5cf6', glow: '#8b5cf660' },
  procedural: { fill: '#10b981', glow: '#10b98160' },
  working: { fill: '#f59e0b', glow: '#f59e0b60' },
};

export default function MemoryObservatory() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);
  const { memoryNodes, dataSource, connectionStatus } = useNexoraStore();

  const isLiveData = dataSource === 'live' && connectionStatus === 'connected';

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    timeRef.current += 0.012;
    const t = timeRef.current;

    ctx.clearRect(0, 0, W, H);

    // === PERSISTENT SEMANTIC TERRAIN ===
    // Draw terrain as height map using memory node strengths
    const terrainRes = 40;
    const cellW = W / terrainRes;
    const cellH = H / terrainRes;

    for (let gx = 0; gx < terrainRes; gx++) {
      for (let gy = 0; gy < terrainRes; gy++) {
        const wx = (gx / terrainRes) * 800;
        const wy = (gy / terrainRes) * 600;

        // Calculate influence from nearby memory nodes
        let totalInfluence = 0;
        let totalWeight = 0;
        let dominantType = 'semantic';
        let maxTypeWeight = 0;

        memoryNodes.forEach(node => {
          const dx = node.x - wx;
          const dy = node.y - wy;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 100) {
            const w = (1 - dist / 100) * node.strength;
            totalInfluence += w;
            totalWeight += w;
            if (w > maxTypeWeight) {
              maxTypeWeight = w;
              dominantType = node.type;
            }
          }
        });

        const influence = totalWeight > 0 ? totalInfluence / (totalWeight + 1) : 0;
        const colors = TYPE_COLORS[dominantType] || TYPE_COLORS.semantic;

        if (influence > 0.05) {
          // "Mountain" - high-strength knowledge
          const heightFactor = Math.min(1, influence * 3);
          ctx.fillStyle = colors.fill + Math.floor(heightFactor * 60).toString(16).padStart(2, '0');
          ctx.fillRect(gx * cellW, gy * cellH, cellW, cellH);
        }
      }
    }

    // Draw forgotten abyss regions (low strength, old access)
    const now = Date.now();
    memoryNodes.forEach(node => {
      const age = (now - node.lastAccess) / 3600000; // hours
      const scaleX = W / 800;
      const scaleY = H / 600;
      const nx = node.x * scaleX;
      const ny = node.y * scaleY;

      if (node.strength < 0.2 && age > 0.5) {
        // Forgotten abyss
        const darkGrad = ctx.createRadialGradient(nx, ny, 0, nx, ny, 20);
        darkGrad.addColorStop(0, 'rgba(0,0,0,0.5)');
        darkGrad.addColorStop(1, 'transparent');
        ctx.fillStyle = darkGrad;
        ctx.beginPath();
        ctx.arc(nx, ny, 20, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // Draw memory edges
    memoryNodes.forEach(node => {
      const scaleX = W / 800;
      const scaleY = H / 600;
      const nx = node.x * scaleX;
      const ny = node.y * scaleY;

      node.connections.forEach(connId => {
        const conn = memoryNodes.find(m => m.id === connId);
        if (!conn) return;
        const cx2 = conn.x * scaleX;
        const cy2 = conn.y * scaleY;
        ctx.beginPath();
        ctx.moveTo(nx, ny);
        ctx.lineTo(cx2, cy2);
        ctx.strokeStyle = TYPE_COLORS[node.type].fill + '20';
        ctx.lineWidth = 0.5;
        ctx.stroke();
      });
    });

    // Draw memory nodes
    memoryNodes.forEach((node, i) => {
      const scaleX = W / 800;
      const scaleY = H / 600;

      // Slight drift animation
      const driftX = Math.sin(t * 0.5 + i * 0.3) * 2;
      const driftY = Math.cos(t * 0.4 + i * 0.2) * 2;
      const nx = node.x * scaleX + driftX;
      const ny = node.y * scaleY + driftY;

      const colors = TYPE_COLORS[node.type] || TYPE_COLORS.semantic;
      const r = 3 + node.strength * 6;

      // Instability crack (low stability nodes)
      if (node.stability < 0.3) {
        ctx.strokeStyle = '#ef444460';
        ctx.lineWidth = 1;
        for (let c = 0; c < 3; c++) {
          const angle = (c / 3) * Math.PI * 2 + t;
          ctx.beginPath();
          ctx.moveTo(nx, ny);
          ctx.lineTo(nx + Math.cos(angle) * (r + 5), ny + Math.sin(angle) * (r + 5));
          ctx.stroke();
        }
      }

      // Glow
      const glow = ctx.createRadialGradient(nx, ny, 0, nx, ny, r * 3);
      glow.addColorStop(0, colors.glow);
      glow.addColorStop(1, 'transparent');
      ctx.fillStyle = glow;
      ctx.beginPath();
      ctx.arc(nx, ny, r * 3, 0, Math.PI * 2);
      ctx.fill();

      // Core
      ctx.fillStyle = colors.fill;
      ctx.globalAlpha = 0.5 + node.strength * 0.5;
      ctx.beginPath();
      ctx.arc(nx, ny, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;

      // "Mountain peak" for high-strength nodes
      if (node.strength > 0.8) {
        ctx.beginPath();
        ctx.arc(nx, ny, r + 3 + Math.sin(t * 2 + i) * 1.5, 0, Math.PI * 2);
        ctx.strokeStyle = colors.fill + '60';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });

    // Legend
    ctx.fillStyle = 'rgba(3,7,18,0.7)';
    ctx.fillRect(W - 150, 12, 138, 112);
    ctx.strokeStyle = 'rgba(30,58,95,0.5)';
    ctx.lineWidth = 1;
    ctx.strokeRect(W - 150, 12, 138, 112);

    ctx.fillStyle = '#94a3b8';
    ctx.font = 'bold 9px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('SEMANTIC TERRAIN', W - 144, 27);

    const types = ['episodic', 'semantic', 'procedural', 'working'];
    const descriptions = ['Episodic Events', 'Semantic Knowledge', 'Procedural Skills', 'Working Memory'];
    types.forEach((type, i) => {
      const c = TYPE_COLORS[type];
      ctx.fillStyle = c.fill;
      ctx.beginPath();
      ctx.arc(W - 138, 42 + i * 18, 4, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#64748b';
      ctx.font = '9px JetBrains Mono, monospace';
      ctx.fillText(descriptions[i], W - 130, 46 + i * 18);
    });

    ctx.fillStyle = '#64748b';
    ctx.font = '8px JetBrains Mono, monospace';
    ctx.fillText('■ = High strength (mountain)', W - 144, 118);

    animRef.current = requestAnimationFrame(draw);
  }, [memoryNodes]);

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

  const stats = [
    { label: 'Total Nodes', value: '80' },
    { label: 'High Strength (>0.8)', value: String(useNexoraStore.getState().memoryNodes.filter(m => m.strength > 0.8).length) },
    { label: 'Unstable (<0.3)', value: String(useNexoraStore.getState().memoryNodes.filter(m => m.stability < 0.3).length) },
    { label: 'Avg Access Count', value: Math.floor(useNexoraStore.getState().memoryNodes.reduce((a, m) => a + m.accessCount, 0) / 80).toString() },
  ];

  return (
    <div className="relative w-full h-full flex flex-col">
      {/* Stats header */}
      <div className="flex items-center gap-3 px-4 py-2 border-b border-slate-800/50 bg-slate-950/50 shrink-0">
        <div className="flex items-center gap-2">
          <div className="text-[10px] text-slate-500 uppercase tracking-widest">Memory Observatory</div>
          {isLiveData && (
            <span className="text-[8px] bg-emerald-500/20 text-emerald-400 px-1.5 py-0.5 rounded-full font-mono border border-emerald-500/30 flex items-center gap-1">
              <Wifi size={8} /> LIVE
            </span>
          )}
        </div>
        {stats.map(s => (
          <div key={s.label} className="flex items-center gap-2 px-3 py-1 bg-slate-900/60 rounded-lg border border-slate-800/50">
            <div className="text-[9px] text-slate-500">{s.label}</div>
            <div className="text-xs font-bold font-mono text-violet-400">{s.value}</div>
          </div>
        ))}
        <div className="ml-auto flex items-center gap-2 text-[10px] text-slate-500">
          <span className="w-2 h-2 rounded bg-red-500" />
          <span>Unstable (cracking)</span>
          <span className="w-2 h-2 rounded bg-slate-800 border border-slate-600 ml-2" />
          <span>Forgotten abyss</span>
        </div>
      </div>

      <div className="flex-1 relative overflow-hidden">
        <canvas ref={canvasRef} className="w-full h-full" />
      </div>
    </div>
  );
}
