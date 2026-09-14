# Spring causal-route interactive

Open `spring-route-geometry.html` directly or embed it from the project page:

```html
<iframe
  src="/interactives/spring-route-geometry.html"
  title="Interactive three-dimensional view of the held-out Spring causal route"
  loading="lazy"
  sandbox="allow-scripts allow-same-origin">
</iframe>
```

Bundle contents:

- `spring-route-geometry.html`: standalone, dependency-free canvas interaction;
- `spring-route-geometry-data.json`: the same compact held-out coordinates;
- `spring-route-geometry-fallback.png`: static/no-JavaScript fallback;
- `manifest.json`: model setting, sample counts and source information.

The interactive displays the canonical Large Short seed-3408 50K checkpoint
at functional block B6. PCA and phase alignment are fit on disjoint fit
trajectories; all 128 displayed points are held out.

Regenerate with `website/scripts/build_spring_route_geometry.py` and explicit
paths to the frozen coordinates, phase-plane CSV, and summary JSON. The builder
does not hard-code private source paths.
