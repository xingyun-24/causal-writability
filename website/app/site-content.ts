import { assetUrl } from "./asset-url";

// Add the arXiv identifier after the paper is announced.
export const publication = {
  title: "A Chosen Future Can Still Be Rewritten",
  subtitle: "Causal Writability in Video Models",
  paper: assetUrl("/paper/main.pdf"),
  code: "https://github.com/xingyun-24/causal-writability" as string | null,
  arxiv: null as string | null,
  projectUrl: "https://xingyun-24.github.io/causal-writability/",
  trailer: assetUrl("/videos/overview.mp4"),
  trailerPoster: assetUrl("/videos/overview-poster.png"),
  trailerCaptions: null as string | null,
};

export const citation = `@misc{wang2026causalwritability,
  title={A Chosen Future Can Still Be Rewritten: Causal Writability in Video Models},
  author={Xingyun Wang and Haomin Zheng and Man Yuan and Leqian Yang and Ziming Liu},
  year={2026},
  note={Preprint}
}`;
