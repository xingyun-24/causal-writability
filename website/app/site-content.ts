// Set the final public URLs and trailer here when they are ready.
export const publication = {
  title: "A Chosen Future Can Still Be Rewritten",
  subtitle: "Causal Writability in Video Models",
  paper: "/paper/main.pdf",
  code: null as string | null,
  arxiv: null as string | null,
  projectUrl: "https://xingyun-24.github.io/causal-writability/",
  trailer: (import.meta.env.VITE_LOCAL_TRAILER || null) as string | null,
  trailerPoster: import.meta.env.VITE_LOCAL_TRAILER_POSTER || "/paper/current/figure1.png",
  trailerCaptions: null as string | null,
};

export const citation = `@misc{wang2026causalwritability,
  title={A Chosen Future Can Still Be Rewritten: Causal Writability in Video Models},
  author={Xingyun Wang and Haomin Zheng and Man Yuan and Leqian Yang and Ziming Liu},
  year={2026},
  note={Preprint}
}`;
