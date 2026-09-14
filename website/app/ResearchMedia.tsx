"use client";

import { useEffect, useRef, useState } from "react";
import { ExternalLink, Pause, Play, RotateCcw } from "lucide-react";
import { assetUrl } from "./asset-url";

type Clip = { src: string; label: string };

// Retains Leqian's audited MP4 sources and native muted/inline playback.
// One transport controls both clips so the experimental comparison stays aligned.
export function VideoPair({ clips, caption }: { clips: [Clip, Clip]; caption: string }) {
  const refs = useRef<(HTMLVideoElement | null)[]>([]);
  const [playing, setPlaying] = useState(false);
  const [duration, setDuration] = useState(0);
  const [time, setTime] = useState(0);
  const [error, setError] = useState("");
  const [ready, setReady] = useState([false, false]);
  const playToken = useRef(0);

  useEffect(() => {
    // Metadata may arrive before hydration attaches React's event handlers.
    const videos = refs.current.filter((v): v is HTMLVideoElement => v !== null);
    const refresh = () => {
      setReady(refs.current.map(v => Boolean(v && v.readyState >= 1)));
      const lengths = videos.map(v => v.duration);
      if (lengths.length === 2 && lengths.every(Number.isFinite)) setDuration(Math.min(...lengths));
    };
    videos.forEach(v => v.addEventListener("loadedmetadata", refresh));
    refresh();
    return () => videos.forEach(v => v.removeEventListener("loadedmetadata", refresh));
  }, []);

  function pause() {
    playToken.current += 1;
    refs.current.forEach(v => v?.pause());
    setPlaying(false);
  }
  function seek(value: number) {
    refs.current.forEach(v => { if (v && Number.isFinite(v.duration)) v.currentTime = Math.min(value, v.duration); });
    setTime(value);
  }
  async function play() {
    const token = ++playToken.current;
    setError("");
    seek(time >= duration - .05 ? 0 : time);
    try {
      await Promise.all(refs.current.map(v => v?.play()));
      if (token !== playToken.current) { refs.current.forEach(v => v?.pause()); return; }
      setPlaying(true);
    } catch {
      pause();
      setError("Playback could not start. Open the MP4 below.");
    }
  }
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(() => {
      const [a, b] = refs.current;
      if (!a || !b) return;
      if (Math.abs(a.currentTime - b.currentTime) > .10 && !b.seeking) b.currentTime = a.currentTime;
      setTime(a.currentTime);
    }, 150);
    return () => clearInterval(timer);
  }, [playing]);
  return <figure className="video-pair">
    <div className="paired-clips">{clips.map((clip, index) => <div key={clip.src}>
      <div className="clip-label">{clip.label}<a href={assetUrl(clip.src)} target="_blank" rel="noreferrer" title={`Open ${clip.label} MP4`} aria-label={`Open ${clip.label} MP4`}><ExternalLink size={14}/></a></div>
      <video ref={v => { refs.current[index] = v; }} src={assetUrl(clip.src)} muted playsInline preload="metadata"
        aria-label={clip.label} onLoadedMetadata={() => {
          setReady(old => old.map((v, i) => i === index || v));
          const lengths = refs.current.map(v => v?.duration ?? NaN);
          if (lengths.every(Number.isFinite)) setDuration(Math.min(...lengths));
        }} onEnded={pause} onError={() => { pause(); setError("Video unavailable. Open the MP4 link to retry."); }} />
    </div>)}</div>
    <div className="transport">
      <button type="button" onClick={playing ? pause : play} disabled={!ready.every(Boolean)} title={playing ? "Pause both videos" : "Play both videos"} aria-label={playing ? "Pause both videos" : "Play both videos"}>{playing ? <Pause size={18}/> : <Play size={18}/>}</button>
      <button type="button" onClick={() => { pause(); seek(0); }} title="Restart both videos" aria-label="Restart both videos"><RotateCcw size={17}/></button>
      <input type="range" min="0" max={duration || 1} step="0.01" value={Math.min(time, duration || 1)} aria-label="Paired video progress" disabled={!duration}
        onChange={e => { pause(); seek(Number(e.target.value)); }}/>
      <span className="timecode">{time.toFixed(1)} / {duration.toFixed(1)} s</span>
    </div>
    {error && <p role="alert">{error}</p>}
    <figcaption>{caption}</figcaption>
  </figure>;
}

export function PCAView({ task, raw, difference }: { task: string; raw: string; difference: string }) {
  const [mode, setMode] = useState<"raw" | "difference">("raw");
  const [visible, setVisible] = useState(false);
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const observer = new IntersectionObserver(entries => {
      setVisible(entries.some(entry => entry.isIntersecting));
    }, { rootMargin: "240px" });
    if (host.current) observer.observe(host.current);
    return () => observer.disconnect();
  }, []);
  const src = assetUrl(mode === "raw" ? raw : difference);
  return <div className="pca-view" ref={host}>
    <div className="pca-toolbar">
      <div role="tablist" aria-label={`${task} PCA views`}>
        <button type="button" role="tab" aria-selected={mode === "raw"} onClick={() => setMode("raw")}>Raw activations</button>
        <button type="button" role="tab" aria-selected={mode === "difference"} onClick={() => setMode("difference")}>Matched differences</button>
      </div>
      <a href={src} target="_blank" rel="noreferrer" aria-label={`Open ${task} PCA separately`} title="Open separately"><ExternalLink size={18}/></a>
    </div>
    <div role="tabpanel" className="pca-stage">
      {visible && <iframe key={src} src={src} title={`${task}: ${mode === "raw" ? "raw activation projection" : "matched-difference PCA"}`} sandbox="allow-scripts allow-same-origin allow-downloads"/>}
    </div>
  </div>;
}
