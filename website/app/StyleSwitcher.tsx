"use client";
import { useEffect, useState } from "react";

export default function StyleSwitcher() {
  const [style, setStyle] = useState("leqian");
  useEffect(() => {
    const selected = new URLSearchParams(window.location.search).get("style") === "clean" ? "clean" : "leqian";
    setStyle(selected);
    document.documentElement.dataset.pageStyle = selected;
  }, []);
  function choose(selected: string) {
    setStyle(selected);
    document.documentElement.dataset.pageStyle = selected;
    const url = new URL(window.location.href);
    url.searchParams.set("style", selected);
    history.replaceState(null, "", url);
  }
  return <div className="styleSwitcher" role="group" aria-label="Visual style comparison">
    <button type="button" aria-pressed={style === "leqian"} onClick={() => choose("leqian")}>A · Leqian</button>
    <button type="button" aria-pressed={style === "clean"} onClick={() => choose("clean")}>B · Editorial</button>
  </div>;
}
