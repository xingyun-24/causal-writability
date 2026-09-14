Overleaf collaboration project

Main document: main.tex
Compiler: pdfLaTeX
Bibliography: BibTeX, refs.bib

Upload the ZIP using New Project > Upload Project, then compile main.tex.
Use Share to invite collaborators with edit access.

Editing locations:
- main.tex: abstract and main sections 1-5 (including section 5's Figure 6).
- sections/06_discussion.tex: Discussion.
- sections/07_related_work.tex: Related Work.
- sections/reproducibility.tex and sections/ai_use.tex: declarations.
- appendix_revision_20260822.tex: appendix, including Figures 32 and 33.
- refs.bib: bibliography; edit this file rather than a generated .bbl file.
- figures/: publication PDF assets referenced by the manuscript.

Figure 1 defaults to Haomin's updated layout. The alternative asset is also
included because main.tex retains its existing figureoneoriginal switch.
All caption text, image widths, and trim settings are preserved.

Plotting scripts, raw data, checkpoints, build logs, and old manuscript copies
are not needed to compile and are intentionally omitted. Regenerate figures
in the research repository and replace the corresponding PDF here, keeping
its filename. Coordinate simultaneous figure edits with coauthors.

This package is a source snapshot. Once collaboration starts, use the shared
Overleaf project as the manuscript editing source and export its source ZIP
before merging changes back into the local repository.
