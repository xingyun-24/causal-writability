# Paper Results

`index.json` links the final main-paper PDFs to their supporting tables.
Filenames inherited from the experiment code may use an older figure number;
the `figure` field in the index is the paper number.

| Paper figure | Included data |
|---|---|
| 1 | Conceptual illustration; no numerical data table |
| 2 | Short/Long 704-rollout metrics and original endpoint/ambiguous-cue video examples |
| 3 | Phase coordinates, low-rank results, controller summaries and three natural-behavior grids |
| 4 | 90 seed/checkpoint behavior records, local layer profiles, future-fate and Short/Long summaries |
| 5 | Pretrained Direct/Neutral-first behavior, controller and boundary summaries |
| 6 | Target transport, paired Q/K/V comparisons and the 48-receiver gain sweep |

`examples/videos.json` maps the supplied MP4s to their experiment, receiver and
model. Some videos illustrate appendix examples rather than a displayed main-paper
frame strip; their entries state that distinction.

These are selected supporting tables. This is not a complete export of every
receiver-level appendix file or an exact-layout figure rebuilding package.
Server-only file-location fields were removed from exported data; scientific
measurements and sample identifiers were retained.
