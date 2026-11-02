# Experimental mainline MTP switching validation, 2026-11-02

**Draft: rendered-text equality remains unresolved (3/14 paired hashes match).**

The opt-in integration keeps a solo pipeline beside batch MTP. A physical single-GPU
control is measured separately. This report does not add a new MTP algorithm or
claim the historical custom engine's numbers for this mainline build.

Engine source: `4398ff2e8ee44768104587585a5ce6a7e77e99d3`, based on upstream
`c62d5e3a15b40ea51c96aae9e8d36928965f53ab` plus pending #4909 and the switching patch.
See `source.json` for transferred/normalized source hashes and `hip-build.json`
for the complete HIP build. All engine objects were rebuilt; fixed dependency
sources were reused. The same binary runs both dual-GPU arms.

Measured on two RX 7900 XTX, 64 GB host RAM, Linux, gfx1100, ROCm10.
The model is Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS. Context capacity is 131072;
the actual prompt counts are in the raw usage records. Output limit is 512,
or 128 for the shorter staggered request. Reasoning is disabled; temperature 0,
seed 42. Server/model-card sampling defaults otherwise apply. These API fields
are not identical to the historical benchmark script, so no historical speedup
is computed here.

## Same-day measurements

Rates below are **HTTP end-to-end output tokens/s**, using server usage, with
the median across the stated warm runs. The first cold request is retained in
each raw file and excluded from warm statistics. Solo-after is measured after
the simultaneous and staggered sequences on the same service.

| Layout | Warm solo before | Warm solo after switching | Two requests, aggregate |
| --- | ---: | ---: | ---: |
| Dual GPU, opt-ins off, piped MTP on | 51.69 (2 runs) | 56.34 (3 runs) | 134.05 (3 runs) |
| Same dual GPU, hybrid + penalty histories on | 89.60 (2 runs) | 93.01 (3 runs) | 136.18 (3 runs) |
| Physical GPU 0, existing solo MTP | 69.54 (2 runs) | not run | not run |
| Physical GPU 1, existing solo MTP | 60.08 (2 runs) | not run | not run |

Each dual arm runs 14 requests: three initial solo requests, three simultaneous
pairs, a staggered long/short pair, then three solo requests. Each physical
single-card arm runs three requests, including its cold run. The staggered
observation records that streaming began before the newcomer and continued
after it finished. HTTP overlap alone does not prove concurrent GPU execution;
keep the engine path evidence with the API records.

The only settings changed between dual arms are `STRATA_HYBRID_WINDOWS` and
`STRATA_PIPELINE_PENALTIES`. Automatic expert-cache sizing can differ because
the solo pipeline reserves verifier/snapshot memory; the sanitized configs
retain that setting. The single-card controls need a 4096 MiB conversation-cache
budget instead of 16384 MiB and cap host registration at 12 GiB; they are
capacity-adjusted controls, not the same memory layout as the dual service.
Vision is disabled in every isolated arm. Hardware power/fan policies are
unchanged throughout; see `environment.json`.

## Validation and limits

The complete CUDA 13.1 / sm_75 engine also built successfully after correcting
the HIP-only MMQ option in its build configuration. NVIDIA GPU execution was
not run. Initial configuration/permission failures are retained separately in
`cuda-preparation-attempts.json`; they precede the successful build. No SYCL
toolchain was configured, so that backend's build/runtime remain untested.

**Rendered-text equality is not established: only 3 of the 14 paired dual-arm
request hashes match.** The opt-out control itself also varies across repeated
greedy requests before the final solo phase. The cause has not been isolated.
This is a correctness investigation, not a demonstrated lossless acceleration;
different generated content can also change MTP acceptance and measured speed.
The complete per-request comparison is in `summary.json`.

Runtime passed: `True`. Original production restored:
`True`. Original config unchanged:
`True`. Hardware policies unchanged:
`True`. Guard issues: `[]`.
Here "runtime passed" means HTTP completion and the request-overlap observation;
it does not mean output parity or a quality check passed. Failures, if any,
remain in `runtime.json` and the raw request records.

`summary.json` includes the rendered-text hash comparison between paired dual
arms. It covers one repeated greedy synthetic prompt, not a quality suite,
token/logit parity, default-output parity against an unpatched main binary,
sampled decoding, or other GPUs. Full CUDA/SYCL validation is reported separately
in the PR; a HIP run does not establish those backends' runtime behavior.

## Reproduce

Use `docs/MTP_SWITCHING.md` and `tools/bench_mtp_switch.py` from the engine source
commit above. `environment.json` links the immutable public prompt fixture;
extract its first message content into a text file and verify its SHA256. Use
the model/drafter and a working two-stage split, then merge the sanitized config
fragments into an isolated run config with your own paths. Do not replace a
working service before backing it up. Keep the same power limits and all raw
failures; do not compare the cold request against a warm one.
