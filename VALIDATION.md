# Validation status

Last checked: 2026-10-30.

## Completed checks

- All 26 official FID records agree with the archived per-step evaluator JSONs.
- Sampling counts, seeds, batch sizes and dtypes agree with archived run metadata.
- Reported latency agrees with archived elapsed time divided by sample count; this is not a new hardware benchmark.
- The LSUN projected-bootstrap N8-versus-N6 interval includes zero. The four Table 2 RGB sliced-W2 values agree with the underlying JSON.
- Eight standard-library tests cover all command stages, worker option names, generation configuration, invalid steps, dry-run behavior, missing prerequisites, refusal to overwrite outputs and rejection of unsupported class-conditioned Jacobian inputs.
- One synthetic CPU numerical test checks paired-bootstrap repeatability and zero paired differences for identical feature inputs. It does not validate the paper's scientific claims.
- The cross-model PNG and SVG plots were regenerated from the archived CSVs.
- Python source syntax, source-copy hashes, local documentation links, English-only text, archive contents and common credential/private-path patterns are checked during packaging.

The CPU numerical test and plotting checks ran on macOS ARM64 using Python 3.9, NumPy 1.26.4, SciPy 1.13.1 and Matplotlib 3.9.4. This small test environment is not the original GPU experiment environment. The intended full-rerun environment is Python 3.10/3.11 on Linux CUDA. The GitHub Actions configuration is provided but has not been run on GitHub.

## Not yet validated

- End-to-end generation on the original H100 hardware.
- Full official TensorFlow FID recomputation from regenerated images.
- Pixel-space and Jacobian reruns on the full datasets.
- A clean installation of both full GPU and TensorFlow dependency sets.
- Exact ImageNet checkpoint revision and original software lockfile recovery.
- End-to-end Algorithm 1 selector reproduction or mathematical proof verification.

Bitwise reproduction has not been established.

## Remaining validation and maintenance

1. Check the applicable model and dataset terms before downloading or redistributing external assets.
2. Add the final public paper URL once available.
3. Run the documented GPU smoke test and, ideally, the complete pipeline in a recorded environment.
4. If files change, regenerate the checksum manifest with `python refresh_manifest.py`, then rerun the checks.
