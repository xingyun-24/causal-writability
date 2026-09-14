"use client";
import { useState } from "react";
import figures from "../public/paper/appendix/manifest.json";

export default function SupplementGallery() {
  const groups = [...new Set(figures.map(figure => figure.group))];
  const [open, setOpen] = useState<Record<string, boolean>>({ [groups[0]]: true });
  return <div className="supplementGroups">{groups.map(group => {
    const selected = figures.filter(figure => figure.group === group);
    return <details className="supplementGroup" key={group} open={Boolean(open[group])}
      onToggle={e => { const shown = e.currentTarget.open; setOpen(old => old[group] === shown ? old : { ...old, [group]: shown }); }}>
      <summary><span>{group}</span><small>{selected.length} figures</small></summary>
      {open[group] && <div className="atlasGrid">{selected.map(figure => <article className="atlasCard" key={figure.number}>
        <div><span>FIGURE {figure.number}</span><h3>{figure.title}</h3><p>{figure.note}</p>
          <a className="captionLink" href={`/paper/main.pdf#page=${figure.page}`} target="_blank" rel="noreferrer">Caption and methods in paper ↗</a>
        </div>
        <a href={figure.src} target="_blank" rel="noreferrer"><img src={figure.src} alt={`Figure ${figure.number}: ${figure.title}`} loading="lazy"/></a>
      </article>)}</div>}
    </details>;
  })}</div>;
}
