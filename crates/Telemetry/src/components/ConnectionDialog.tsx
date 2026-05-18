import { useState, useEffect } from 'react';
import { useNexoraStore, ConnectionStatus } from '../store/nexoraStore';
import { telemetryService } from '../services/telemetryService';
import { Plug, PlugZap, X, Loader2, AlertCircle, CheckCircle2, Wifi, WifiOff } from 'lucide-react';

const STATUS_CONFIG: Record<ConnectionStatus, { color: string; label: string; icon: React.ReactNode }> = {
  disconnected: { color: 'text-slate-400', label: 'Disconnected', icon: <Plug size={12} /> },
  connecting: { color: 'text-amber-400', label: 'Connecting...', icon: <Loader2 size={12} className="animate-spin" /> },
  connected: { color: 'text-emerald-400', label: 'Connected', icon: <PlugZap size={12} /> },
  error: { color: 'text-red-400', label: 'Connection Error', icon: <AlertCircle size={12} /> },
};

export default function ConnectionDialog() {
  const [open, setOpen] = useState(false);
  const [url, setUrl] = useState('http://localhost:8080');
  const { connectionStatus, connectToAI, disconnectFromAI, aiConnectionUrl } = useNexoraStore();
  const [statusInfo, setStatusInfo] = useState({ connected: false, url: '', ai_metrics: false });

  useEffect(() => {
    if (aiConnectionUrl) setUrl(aiConnectionUrl);
  }, [aiConnectionUrl]);

  useEffect(() => {
    const check = async () => {
      const s = await telemetryService.checkConnection();
      setStatusInfo(s);
    };
    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleConnect = async () => {
    if (!url.trim()) return;
    await connectToAI(url.trim());
  };

  const handleDisconnect = async () => {
    await disconnectFromAI();
  };

  const cfg = STATUS_CONFIG[connectionStatus];

  return (
    <>
      {/* Connection status indicator */}
      <button
        onClick={() => setOpen(true)}
        className="fixed bottom-4 right-4 z-40 flex items-center gap-2 px-3 py-2 glass rounded-xl border border-slate-700/50 hover:border-cyan-500/30 transition-all group"
        title={`Nexora AI: ${cfg.label}`}
      >
        <div className={`flex items-center gap-1.5 ${cfg.color}`}>
          {connectionStatus === 'connected' ? (
            <Wifi size={13} className="text-emerald-400" />
          ) : connectionStatus === 'connecting' ? (
            <Loader2 size={13} className="animate-spin text-amber-400" />
          ) : (
            <WifiOff size={13} className="text-slate-400" />
          )}
          <span className="text-[9px] font-mono">{cfg.label}</span>
        </div>
        {connectionStatus === 'connected' && (
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse shadow-[0_0_6px_#10b981]" />
        )}
      </button>

      {/* Dialog */}
      {open && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm">
          <div className="w-96 glass rounded-xl border border-slate-700/50 shadow-2xl overflow-hidden">
            {/* Header */}
            <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/50">
              <div className="flex items-center gap-2">
                <PlugZap size={14} className="text-cyan-400" />
                <span className="text-sm font-semibold text-slate-200">Connect to Nexora AI</span>
              </div>
              <button onClick={() => setOpen(false)} className="text-slate-500 hover:text-white transition-colors">
                <X size={14} />
              </button>
            </div>

            <div className="p-4 space-y-4">
              {/* Connection form */}
              <div>
                <label className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 block">API URL</label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={url}
                    onChange={(e) => setUrl(e.target.value)}
                    placeholder="http://localhost:8080"
                    disabled={connectionStatus === 'connecting' || connectionStatus === 'connected'}
                    className="flex-1 px-3 py-2 bg-slate-900 border border-slate-700/50 rounded-lg text-xs font-mono text-slate-200 placeholder-slate-600 focus:outline-none focus:border-cyan-500/50 transition-colors disabled:opacity-50"
                  />
                  {connectionStatus === 'connected' ? (
                    <button
                      onClick={handleDisconnect}
                      className="px-3 py-2 bg-red-500/20 border border-red-500/30 rounded-lg text-red-400 text-xs hover:bg-red-500/30 transition-colors shrink-0"
                    >
                      Disconnect
                    </button>
                  ) : (
                    <button
                      onClick={handleConnect}
                      disabled={connectionStatus === 'connecting' || !url.trim()}
                      className="px-3 py-2 bg-cyan-500/20 border border-cyan-500/30 rounded-lg text-cyan-400 text-xs hover:bg-cyan-500/30 transition-colors shrink-0 disabled:opacity-50"
                    >
                      {connectionStatus === 'connecting' ? (
                        <Loader2 size={12} className="animate-spin" />
                      ) : 'Connect'}
                    </button>
                  )}
                </div>
              </div>

              {/* Status */}
              <div className="bg-slate-900/60 rounded-lg p-3 border border-slate-800/50 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-slate-500 uppercase">Connection</span>
                  <span className={`text-[10px] font-mono ${cfg.color}`}>{cfg.label}</span>
                </div>

                {connectionStatus === 'connected' && (
                  <div className="flex items-center gap-2 text-[10px] text-emerald-400">
                    <CheckCircle2 size={10} />
                    <span>Connected to {url}</span>
                  </div>
                )}

                {connectionStatus === 'error' && (
                  <div className="flex items-center gap-2 text-[10px] text-red-400">
                    <AlertCircle size={10} />
                    <span>Failed to connect. Check if Nexora AI is running.</span>
                  </div>
                )}

                <div className="flex items-center gap-4 text-[9px] text-slate-500">
                  <span className="flex items-center gap-1">
                    <span className={`w-1.5 h-1.5 rounded-full ${connectionStatus === 'connected' ? 'bg-emerald-400' : 'bg-slate-600'}`} />
                    System Metrics
                  </span>
                  <span className="flex items-center gap-1">
                    <span className={`w-1.5 h-1.5 rounded-full ${statusInfo.ai_metrics ? 'bg-emerald-400' : 'bg-slate-600'}`} />
                    AI Metrics
                  </span>
                </div>
              </div>

              {/* Endpoints */}
              <div className="text-[9px] text-slate-500 space-y-1">
                <div className="uppercase tracking-widest mb-1">Expected Endpoints</div>
                <div className="font-mono bg-slate-900/60 px-2 py-1 rounded">{url}/health</div>
                <div className="font-mono bg-slate-900/60 px-2 py-1 rounded">{url}/metrics</div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
