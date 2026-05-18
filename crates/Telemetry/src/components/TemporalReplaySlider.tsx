import { useState, useRef, useEffect } from 'react';
import { Play, Pause, SkipBack, SkipForward, RotateCcw } from 'lucide-react';

interface TemporalReplaySliderProps {
  onTimeChange: (time: number) => void;
  currentTime: number;
  maxTime: number;
}

export default function TemporalReplaySlider({ onTimeChange, currentTime, maxTime }: TemporalReplaySliderProps) {
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1);
  const sliderRef = useRef<HTMLInputElement>(null);
  const animationRef = useRef<number>(0);
  const localTimeRef = useRef(currentTime);

  useEffect(() => {
    localTimeRef.current = currentTime;
  }, [currentTime]);

  useEffect(() => {
    if (isPlaying) {
      const animate = () => {
        const newTime = localTimeRef.current + 0.016 * playbackSpeed;
        const clampedTime = newTime >= maxTime ? 0 : newTime;
        localTimeRef.current = clampedTime;
        onTimeChange(clampedTime);
        animationRef.current = requestAnimationFrame(animate);
      };
      animationRef.current = requestAnimationFrame(animate);
    } else {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPlaying, playbackSpeed, maxTime, onTimeChange]);

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    onTimeChange(value);
  };

  const togglePlay = () => {
    setIsPlaying(!isPlaying);
  };

  const skipBack = () => {
    onTimeChange(Math.max(0, currentTime - 5));
  };

  const skipForward = () => {
    onTimeChange(Math.min(maxTime, currentTime + 5));
  };

  const reset = () => {
    onTimeChange(0);
    setIsPlaying(false);
  };

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    const ms = Math.floor((time % 1) * 100);
    return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  return (
    <div className="absolute bottom-4 left-4 right-4 glass rounded-xl p-4 border border-cyan-500/20">
      <div className="flex items-center gap-4">
        {/* Playback Controls */}
        <div className="flex items-center gap-2 shrink-0">
          <button
            onClick={reset}
            className="p-2 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-700/50 hover:border-cyan-500/30 transition-all group"
            title="Reset"
          >
            <RotateCcw size={14} className="text-slate-400 group-hover:text-cyan-400 transition-colors" />
          </button>
          <button
            onClick={skipBack}
            className="p-2 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-700/50 hover:border-cyan-500/30 transition-all group"
            title="Skip Back 5s"
          >
            <SkipBack size={14} className="text-slate-400 group-hover:text-cyan-400 transition-colors" />
          </button>
          <button
            onClick={togglePlay}
            className="p-3 rounded-lg bg-cyan-500/20 hover:bg-cyan-500/30 border border-cyan-500/40 hover:border-cyan-500/60 transition-all group"
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              <Pause size={16} className="text-cyan-400 group-hover:text-cyan-300 transition-colors" />
            ) : (
              <Play size={16} className="text-cyan-400 group-hover:text-cyan-300 transition-colors ml-0.5" />
            )}
          </button>
          <button
            onClick={skipForward}
            className="p-2 rounded-lg bg-slate-900/60 hover:bg-slate-800/80 border border-slate-700/50 hover:border-cyan-500/30 transition-all group"
            title="Skip Forward 5s"
          >
            <SkipForward size={14} className="text-slate-400 group-hover:text-cyan-400 transition-colors" />
          </button>
        </div>

        {/* Time Slider */}
        <div className="flex-1 flex flex-col gap-1">
          <div className="flex justify-between items-center">
            <span className="text-[10px] text-slate-500 font-['Orbitron']">TEMPORAL REPLAY</span>
            <div className="flex items-center gap-3">
              <span className="text-[11px] font-mono text-cyan-400 font-['Space_Grotesk']">
                {formatTime(currentTime)}
              </span>
              <span className="text-[10px] text-slate-500">/</span>
              <span className="text-[11px] font-mono text-slate-400 font-['Space_Grotesk']">
                {formatTime(maxTime)}
              </span>
            </div>
          </div>
          <div className="relative">
            <input
              ref={sliderRef}
              type="range"
              min="0"
              max={maxTime}
              step="0.01"
              value={currentTime}
              onChange={handleSliderChange}
              className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan-400 [&::-webkit-slider-thumb]:shadow-[0_0_8px_#06b6d4] [&::-webkit-slider-thumb]:cursor-pointer [&::-moz-range-thumb]:w-3 [&::-moz-range-thumb]:h-3 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:bg-cyan-400 [&::-moz-range-thumb]:shadow-[0_0_8px_#06b6d4] [&::-moz-range-thumb]:cursor-pointer"
              style={{
                background: `linear-gradient(to right, #06b6d4 0%, #06b6d4 ${(currentTime / maxTime) * 100}%, #1e293b ${(currentTime / maxTime) * 100}%, #1e293b 100%)`,
              }}
            />
            {/* Glow effect on slider track */}
            <div
              className="absolute top-0 left-0 h-2 rounded-lg pointer-events-none"
              style={{
                width: `${(currentTime / maxTime) * 100}%`,
                background: 'linear-gradient(to right, rgba(6, 182, 212, 0.3), rgba(6, 182, 212, 0.1))',
                filter: 'blur(4px)',
              }}
            />
          </div>
        </div>

        {/* Playback Speed */}
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-[9px] text-slate-500 font-['Orbitron']">SPEED</span>
          <div className="flex gap-1">
            {[0.5, 1, 2, 4].map((speed) => (
              <button
                key={speed}
                onClick={() => setPlaybackSpeed(speed)}
                className={`px-2 py-1 rounded text-[10px] font-mono font-['Space_Grotesk'] transition-all ${
                  playbackSpeed === speed
                    ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/40'
                    : 'bg-slate-900/60 text-slate-400 border border-slate-700/50 hover:border-slate-600'
                }`}
              >
                {speed}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Event Markers */}
      <div className="mt-3 pt-3 border-t border-slate-700/50">
        <div className="flex items-center gap-4 text-[9px]">
          <span className="text-slate-500 font-['Orbitron']">EVENTS</span>
          <div className="flex gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-[0_0_6px_#06b6d4]" />
              <span className="text-slate-400">Reasoning</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-violet-400 shadow-[0_0_6px_#8b5cf6]" />
              <span className="text-slate-400">Memory</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-red-400 shadow-[0_0_6px_#ef4444]" />
              <span className="text-slate-400">Conflict</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400 shadow-[0_0_6px_#10b981]" />
              <span className="text-slate-400">Verified</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
