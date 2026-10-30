# Asset sources and distribution boundaries

This repository does not redistribute model weights, dataset images, Inception weights or large feature caches. Obtain these assets from their original providers under the applicable terms.

`assets/` includes the four small generated sample grids already used in the paper, a raster export of the paper's conceptual overview, and plots regenerated from the bundled CSVs. These are presentation assets, not full evaluation batches. Their source hashes and export descriptions are recorded in `PROVENANCE.json`.

| Asset | Source | Treatment |
| --- | --- | --- |
| Experiment workers and repository helpers | Original `multistep-consistency-release` and accompanying reproduction utilities | MIT license; copyright 2026 Yanshu Wang, Rui Liu, and Tong Yang. The anonymous submission notice has been replaced with author attribution. |
| Official FID evaluator | [OpenAI guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations) | Archived `evaluator.py` copied unchanged with the original OpenAI MIT license. File hash identifies the actual bundled version. |
| Consistency checkpoints | OpenAI Hugging Face model repositories linked in [the reproduction guide](REPRODUCIBILITY.md#environment-and-external-assets) | External downloads only. Review model-card terms and record the resolved revisions. |
| LSUN and ImageNet reference batches | OpenAI evaluation reference links in [the reproduction guide](REPRODUCIBILITY.md#environment-and-external-assets) | External downloads only; original dataset terms still apply. Public availability does not imply unrestricted redistribution. |
| PyTorch FID feature extractor | [pytorch-fid](https://github.com/mseitzer/pytorch-fid) | Installed dependency, not vendored; its pretrained weights may download on first use. |
| Official Inception graph | URL embedded in the archived evaluator | External download; not included in this package. |

The source-code MIT license does not cover third-party data or weights. The sample grids are generated visual examples from the paper, not a redistributed reference dataset; no broader rights to the underlying models or datasets are granted.
