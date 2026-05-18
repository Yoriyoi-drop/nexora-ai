import { useRef, useCallback, useEffect } from 'react';
import { Agent } from '../store/nexoraStore';
import { CANVAS_CONFIG, ANIMATION_CONFIG, SEMANTIC_COLORS, CLUSTER_COLORS, EDGE_STYLES } from '../constants/topology';

interface UseCanvasRendererProps {
  agents: Agent[];
  selectedAgent: string | null;
}

export function useCanvasRenderer({ agents, selectedAgent }: UseCanvasRendererProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animRef = useRef<number>(0);
  const timeRef = useRef(0);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    timeRef.current += ANIMATION_CONFIG.TIME_STEP;
    const t = timeRef.current;

    ctx.clearRect(0, 0, W, H);

    // Scale agents to canvas
    const scaleX = W / CANVAS_CONFIG.BASE_WIDTH;
    const scaleY = H / CANVAS_CONFIG.BASE_HEIGHT;

    drawBackgroundLayer(ctx, W, H, t);
    drawCognitiveHeatmap(ctx, agents, scaleX, scaleY, t, W, H);
    drawGridLayer(ctx, W, H);
    drawClusterBlobs(ctx, agents, scaleX, scaleY);
    drawEdges(ctx, agents, scaleX, scaleY, t);
    drawAgentNodes(ctx, agents, scaleX, scaleY, selectedAgent, t);
    drawHUD();

    animRef.current = requestAnimationFrame(draw);
  }, [agents, selectedAgent]);

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

  return { canvasRef };
}

function drawBackgroundLayer(ctx: CanvasRenderingContext2D, W: number, H: number, t: number) {
  ctx.save();
  for (let i = 0; i < CANVAS_CONFIG.BACKGROUND_PARTICLE_COUNT; i++) {
    const px = (Math.sin(i * 123.456 + t * ANIMATION_CONFIG.BACKGROUND_SPEED) * 0.5 + 0.5) * W;
    const py = (Math.cos(i * 789.012 + t * ANIMATION_CONFIG.BACKGROUND_SPEED * 0.8) * 0.5 + 0.5) * H;
    const size = 0.5 + Math.sin(t + i) * 0.3;
    ctx.beginPath();
    ctx.arc(px, py, size, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(6, 182, 212, 0.03)';
    ctx.fill();
  }
  ctx.restore();
}

function drawCognitiveHeatmap(ctx: CanvasRenderingContext2D, agents: Agent[], scaleX: number, scaleY: number, t: number, W: number, H: number) {
  ctx.save();
  ctx.filter = 'blur(8px)';
  
  // Heat zones around high-activity agents
  agents.forEach(agent => {
    if (agent.status === 'active' || agent.entropy > 0.5) {
      const ax = agent.x * scaleX;
      const ay = agent.y * scaleY;
      const heatRadius = CANVAS_CONFIG.HEAT_RADIUS_BASE + agent.entropy * 40;
      
      const heatGrad = ctx.createRadialGradient(ax, ay, 0, ax, ay, heatRadius);
      if (agent.status === 'critical' || agent.entropy > 0.7) {
        heatGrad.addColorStop(0, 'rgba(239, 68, 68, 0.15)');
        heatGrad.addColorStop(0.5, 'rgba(239, 68, 68, 0.05)');
        heatGrad.addColorStop(1, 'transparent');
      } else if (agent.cluster === 'Memory') {
        heatGrad.addColorStop(0, 'rgba(139, 92, 246, 0.12)');
        heatGrad.addColorStop(0.5, 'rgba(139, 92, 246, 0.04)');
        heatGrad.addColorStop(1, 'transparent');
      } else {
        heatGrad.addColorStop(0, 'rgba(6, 182, 212, 0.12)');
        heatGrad.addColorStop(0.5, 'rgba(6, 182, 212, 0.04)');
        heatGrad.addColorStop(1, 'transparent');
      }
      
      ctx.fillStyle = heatGrad;
      ctx.beginPath();
      ctx.arc(ax, ay, heatRadius, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  // Entropy fog - ambient cognitive noise
  for (let i = 0; i < CANVAS_CONFIG.FOG_POINT_COUNT; i++) {
    const fx = (Math.sin(i * 0.7 + t * ANIMATION_CONFIG.FOG_SPEED) * 0.5 + 0.5) * W;
    const fy = (Math.cos(i * 1.3 + t * ANIMATION_CONFIG.FOG_SPEED * 0.6) * 0.5 + 0.5) * H;
    const fogSize = 150 + Math.sin(t * 0.5 + i) * 50;
    
    const fogGrad = ctx.createRadialGradient(fx, fy, 0, fx, fy, fogSize);
    fogGrad.addColorStop(0, 'rgba(139, 92, 246, 0.06)');
    fogGrad.addColorStop(0.5, 'rgba(139, 92, 246, 0.02)');
    fogGrad.addColorStop(1, 'transparent');
    
    ctx.fillStyle = fogGrad;
    ctx.beginPath();
    ctx.arc(fx, fy, fogSize, 0, Math.PI * 2);
    ctx.fill();
  }

  // Anomaly diffusion - spreading from critical agents
  agents.filter(a => a.status === 'critical' || a.entropy > 0.6).forEach(agent => {
    const ax = agent.x * scaleX;
    const ay = agent.y * scaleY;
    const diffusionRadius = CANVAS_CONFIG.DIFFUSION_RADIUS_BASE + Math.sin(t * 2) * 20;
    
    const diffusionGrad = ctx.createRadialGradient(ax, ay, 0, ax, ay, diffusionRadius);
    diffusionGrad.addColorStop(0, 'rgba(239, 68, 68, 0.08)');
    diffusionGrad.addColorStop(0.3, 'rgba(239, 68, 68, 0.03)');
    diffusionGrad.addColorStop(1, 'transparent');
    
    ctx.fillStyle = diffusionGrad;
    ctx.beginPath();
    ctx.arc(ax, ay, diffusionRadius, 0, Math.PI * 2);
    ctx.fill();
  });
  
  ctx.restore();
}

function drawGridLayer(ctx: CanvasRenderingContext2D, W: number, H: number) {
  ctx.save();
  ctx.filter = 'blur(0.5px)';
  ctx.strokeStyle = 'rgba(30,58,95,0.08)';
  ctx.lineWidth = 0.5;
  const gridSz = CANVAS_CONFIG.GRID_SIZE;
  for (let x = 0; x < W; x += gridSz) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y < H; y += gridSz) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
  ctx.restore();
}

function drawClusterBlobs(ctx: CanvasRenderingContext2D, agents: Agent[], scaleX: number, scaleY: number) {
  ctx.save();
  ctx.filter = 'blur(2px)';
  const clusterCenters: Record<string, { x: number; y: number }> = {};
  agents.forEach(a => {
    if (!clusterCenters[a.cluster]) clusterCenters[a.cluster] = { x: 0, y: 0 };
    clusterCenters[a.cluster].x += a.x * scaleX;
    clusterCenters[a.cluster].y += a.y * scaleY;
  });
  const clusterCounts: Record<string, number> = {};
  agents.forEach(a => { clusterCounts[a.cluster] = (clusterCounts[a.cluster] || 0) + 1; });
  Object.entries(clusterCenters).forEach(([name, pos]) => {
    const count = clusterCounts[name];
    const cx = pos.x / count;
    const cy = pos.y / count;

    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, CANVAS_CONFIG.CLUSTER_BLOB_RADIUS);
    const c = CLUSTER_COLORS[name] || '#06b6d4';
    grad.addColorStop(0, c + '20');
    grad.addColorStop(0.5, c + '10');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, CANVAS_CONFIG.CLUSTER_BLOB_RADIUS, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();

  // Cluster labels (sharp, foreground)
  Object.entries(clusterCenters).forEach(([name, pos]) => {
    const count = clusterCounts[name];
    const cx = pos.x / count;
    const cy = pos.y / count;
    const c = CLUSTER_COLORS[name] || '#06b6d4';

    ctx.fillStyle = c + '50';
    ctx.font = 'bold 10px Orbitron, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(name.toUpperCase(), cx, cy - 95);
  });
}

function drawEdges(ctx: CanvasRenderingContext2D, agents: Agent[], scaleX: number, scaleY: number, t: number) {
  // MIDGROUND LAYER - Edges with subtle blur for depth
  ctx.save();
  ctx.filter = 'blur(0.3px)';
  agents.forEach(agent => {
    const ax = agent.x * scaleX;
    const ay = agent.y * scaleY;

    agent.connections.forEach((targetId, idx) => {
      const target = agents.find(a => a.id === targetId);
      if (!target) return;
      const tx = target.x * scaleX;
      const ty = target.y * scaleY;

      const edgeStyle = agent.status === 'active' && idx % 3 === 0 
        ? EDGE_STYLES.active 
        : idx % 2 === 0 
          ? EDGE_STYLES.direct 
          : EDGE_STYLES.async;

      ctx.strokeStyle = CLUSTER_COLORS[agent.cluster] + '30';
      ctx.lineWidth = edgeStyle.lineWidth;
      
      if (edgeStyle.lineDash.length > 0) {
        ctx.setLineDash(edgeStyle.lineDash);
        const dashOffset = -t * 20;
        ctx.lineDashOffset = dashOffset;
      }
      
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      ctx.setLineDash([]);
    });
  });
  ctx.restore();

  // FOREGROUND LAYER - Active edges with sharp rendering
  agents.forEach(agent => {
    const ax = agent.x * scaleX;
    const ay = agent.y * scaleY;

    agent.connections.forEach((targetId, idx) => {
      const target = agents.find(a => a.id === targetId);
      if (!target) return;
      const tx = target.x * scaleX;
      const ty = target.y * scaleY;

      const edgeStyle = agent.status === 'active' && idx % 3 === 0 
        ? EDGE_STYLES.active 
        : idx % 2 === 0 
          ? EDGE_STYLES.direct 
          : EDGE_STYLES.async;

      if (edgeStyle.pulse) {
        const pulseCount = 2;
        for (let i = 0; i < pulseCount; i++) {
          const pct = ((t * 0.4 + i / pulseCount) % 1);
          const px = ax + (tx - ax) * pct;
          const py = ay + (ty - ay) * pct;
          
          const pulseSize = 2 + Math.sin(t * 8 + i) * 1;
          const gradient = ctx.createRadialGradient(px, py, 0, px, py, pulseSize * 2);
          gradient.addColorStop(0, CLUSTER_COLORS[agent.cluster] + 'ff');
          gradient.addColorStop(0.5, CLUSTER_COLORS[agent.cluster] + '80');
          gradient.addColorStop(1, 'transparent');
          
          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.arc(px, py, pulseSize * 2, 0, Math.PI * 2);
          ctx.fill();
        }
      } else {
        const pct = (t * 0.3) % 1;
        const px = ax + (tx - ax) * pct;
        const py = ay + (ty - ay) * pct;
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = CLUSTER_COLORS[agent.cluster] + '60';
        ctx.fill();
      }
    });
  });
}

function drawAgentNodes(ctx: CanvasRenderingContext2D, agents: Agent[], scaleX: number, scaleY: number, selectedAgent: string | null, t: number) {
  agents.forEach(agent => {
    const ax = agent.x * scaleX;
    const ay = agent.y * scaleY;
    const isSelected = agent.id === selectedAgent;
    const baseR = isSelected ? 10 : 7;

    const color = SEMANTIC_COLORS[agent.status as keyof typeof SEMANTIC_COLORS] || '#475569';

    let alpha = 1;
    if (agent.entropy > 0.6) {
      alpha = 0.6 + 0.4 * Math.sin(t * ANIMATION_CONFIG.ENTROPY_FLICKER_SPEED + agent.x);
    }

    ctx.globalAlpha = alpha;

    // Multi-layered outer glow for depth
    const glow1 = ctx.createRadialGradient(ax, ay, 0, ax, ay, baseR * 4);
    glow1.addColorStop(0, color + '30');
    glow1.addColorStop(0.5, color + '15');
    glow1.addColorStop(1, 'transparent');
    ctx.fillStyle = glow1;
    ctx.beginPath();
    ctx.arc(ax, ay, baseR * 4, 0, Math.PI * 2);
    ctx.fill();

    // Anomaly ripple for critical states
    if (agent.status === 'critical') {
      const rippleR = baseR + 8 + Math.sin(t * ANIMATION_CONFIG.PULSE_SPEED) * 4;
      const rippleGrad = ctx.createRadialGradient(ax, ay, baseR, ax, ay, rippleR);
      rippleGrad.addColorStop(0, '#ef4444' + '40');
      rippleGrad.addColorStop(1, 'transparent');
      ctx.fillStyle = rippleGrad;
      ctx.beginPath();
      ctx.arc(ax, ay, rippleR, 0, Math.PI * 2);
      ctx.fill();
    }

    // Enhanced pulse ring for active reasoning
    if (agent.status === 'active') {
      const pulseCount = 2;
      for (let i = 0; i < pulseCount; i++) {
        const pulsePhase = (t * ANIMATION_CONFIG.PULSE_SPEED + i * Math.PI / pulseCount) % (Math.PI * 2);
        const pulseR = baseR + 5 + Math.sin(pulsePhase) * 4;
        const pulseAlpha = (Math.sin(pulsePhase) + 1) / 2 * 0.4;
        ctx.beginPath();
        ctx.arc(ax, ay, pulseR, 0, Math.PI * 2);
        ctx.strokeStyle = color + Math.floor(pulseAlpha * 255).toString(16).padStart(2, '0');
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    // Memory recall glow for memory systems
    if (agent.cluster === 'Memory') {
      const memoryGlow = ctx.createRadialGradient(ax, ay, 0, ax, ay, baseR * 2.5);
      memoryGlow.addColorStop(0, '#8b5cf6' + '50');
      memoryGlow.addColorStop(1, 'transparent');
      ctx.fillStyle = memoryGlow;
      ctx.beginPath();
      ctx.arc(ax, ay, baseR * 2.5, 0, Math.PI * 2);
      ctx.fill();
    }

    // Selection ring with animation
    if (isSelected) {
      const selPulse = baseR + 6 + Math.sin(t * 5) * 2;
      ctx.beginPath();
      ctx.arc(ax, ay, selPulse, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff80';
      ctx.lineWidth = 2;
      ctx.stroke();
      
      ctx.beginPath();
      ctx.arc(ax, ay, baseR + 3, 0, Math.PI * 2);
      ctx.strokeStyle = '#ffffff40';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Core node with gradient
    const nodeGrad = ctx.createRadialGradient(ax - 2, ay - 2, 0, ax, ay, baseR);
    nodeGrad.addColorStop(0, color + 'ff');
    nodeGrad.addColorStop(0.7, color + 'cc');
    nodeGrad.addColorStop(1, color + '88');
    ctx.fillStyle = nodeGrad;
    ctx.beginPath();
    ctx.arc(ax, ay, baseR, 0, Math.PI * 2);
    ctx.fill();

    // Entropy overlay (red-tint for high entropy)
    if (agent.entropy > 0.5) {
      const entropyAlpha = (agent.entropy - 0.5) * 0.8;
      const entropyFlicker = 0.7 + 0.3 * Math.sin(t * 20 + agent.id.length);
      ctx.fillStyle = `rgba(239,68,68,${entropyAlpha * entropyFlicker})`;
      ctx.beginPath();
      ctx.arc(ax, ay, baseR, 0, Math.PI * 2);
      ctx.fill();
    }

    // Orchestrator white glow
    if (agent.cluster === 'Orchestration') {
      ctx.strokeStyle = '#ffffff60';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(ax, ay, baseR + 2, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.globalAlpha = 1;

    // Label with premium font
    if (isSelected || baseR > 7) {
      ctx.fillStyle = '#e2e8f0';
      ctx.font = `${isSelected ? 'bold 11px' : '10px'} Space Grotesk, sans-serif`;
      ctx.textAlign = 'center';
      ctx.fillText(agent.name.split('-')[0], ax, ay + baseR + 12);
    }
  });
}

function drawHUD() {
  // This would need access to store, so we'll keep it simple for now
  // In a full refactor, this would be a separate component
}
