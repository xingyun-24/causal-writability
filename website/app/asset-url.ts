/** Resolve paper assets for both local previews and the GitHub Pages subpath. */
export function assetUrl(path: string): string {
  const base = import.meta.env.BASE_URL || "/";
  if (!path.startsWith("/") || path.startsWith("//") || base === "/" || path.startsWith(base)) return path;
  return `${base}${path.slice(1)}`;
}
