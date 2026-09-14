# Source and Third-Party Notices

The Spring simulation, training, evaluator and residual-patching implementation
are from the authors' Spring V4 standalone experiment code. Condition Q/K/V hooks
are retained from the paper's downstream experiments. The release adds path-based
entry points and documentation; it does not replace the physical simulator or
intervention semantics with a new implementation.

The vendored DiffSynth implementation is accompanied by its original license in
`third_party/DIFFSYNTH_LICENSE`. Preserve it when redistributing that component.
The model pipeline depends on the separately distributed Wan2.1 VAE. Its upstream
download is linked in the model registry; the VAE is not included in the code tree.

The authors' original code is licensed under MIT; see LICENSE.
Third-party code and separately distributed weights retain their applicable terms.
Model hosting and the complete public release remain pending.
