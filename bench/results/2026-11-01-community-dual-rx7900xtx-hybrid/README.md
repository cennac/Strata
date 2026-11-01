# Two RX 7900 XTX: solo pipeline and concurrent MTP experiment

Measured on 2026-11-01 by Cennac. These are measurements of a **custom 0.1.43-based RDNA3 engine**, not the current upstream engine or the new mainline PR. This archive accompanies the penalty-history fix; it is not a standard or approved benchmark-protocol submission.

## Environment and workload

- Sapphire RX 7900 XTX 24GB Platinum and Sapphire RX 7900 XTX 24GB BLOWER, gfx1100; power caps 303 W / 231 W, automatic fans.
- Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic, ROCm headers/runtime under the existing ROCm 10 installation, GCC 15.2.0. Intel engineering-sample CPU identifies as `Genuine Intel(R) 0000`, family 6 / model 183, 20 online CPUs. 61.5 GiB usable RAM.
- Qwen3.8-Flash-Next GSQ-RCO IQ3_XXS; layer split 26/22; int8 KV, 32768 resident KV cells; maximum context 131072 per active request; HTTP active limit 2, 4 internal slots / 2 groups; spec 3 / min-p 0.5; 16 GiB RAM conversation-cache budget.
- Historical custom base already includes DIRECT slot copying and deferred-commit pipelined MTP backported from upstream `99866a47a2b8`. The GPU kernel archive was not recompiled in the hybrid experiment. `source-provenance.json` records hashes and limits.
- Long-context speed requests: 130037 input tokens, 512 output tokens, 130030 input tokens reused in the representative warm requests. **These are largely cached prompts, not 130K-token fresh-prefill benchmarks.**
- Solo end-to-end throughput = completion tokens / request-batch wall time, including first-token wait; reported solo average = arithmetic mean of the requests' throughputs. Concurrent throughput = all completion tokens / entire repeated two-client interval, including request gaps. It is the sum across clients, not a per-client speed.

## Results

| Custom-tree run | Solo mean end-to-end tok/s | Two-client aggregate tok/s | Concurrent interval | Outcome |
| --- | ---: | ---: | ---: | --- |
| Original MTP control A | 70.94 | 161.68 | 95.00 s | Functional checks passed |
| New binary, hybrid flags off | 72.61 | 159.79 | 96.13 s | Functional checks passed |
| Coexistence only, no penalty-aware staging | 73.11 | 159.53 | 96.28 s | Still took serial solo decode with inherited penalties |
| Original MTP control B | 71.53 | 163.27 | 94.08 s | Functional checks passed |
| Penalty-aware hybrid screening | 120.53 | 157.19 | 91.20 s | Functional checks passed; exact-output diagnostic inconclusive |
| Penalty-aware confirmation | 130.01 | 159.27 | 604.34 s, 94 batches | Speed interval completed; later probe-length assertion failed, whole case remains failed |
| Same binary, follow-up functional acceptance | 128.59 | 159.98 | 32.00 s | Switching, history, cancellation, queueing and vision passed |
| Adopted service, original API port | **116.48** | **161.08** | 184.35 s, 29 batches | Functional acceptance passed |

Adopted-service solo requests: 106.9729, 119.9849 and 122.4952 tok/s. The varying throughput includes different first-token waits; the archive retains the individual timings rather than reporting only the best request. The 604-second run reached blower hotspot 92 C / memory 86 C and minimum available RAM 17.38 GiB across that case including loading and checks. Adoption reached hotspot 85 C on both cards, blower memory 80 C, minimum available RAM 13.13 GiB. `power_w` contains GPU sensor readings, including instantaneous peaks; it is not wall-socket power, and the configured caps were checked separately.

The server already migrates GEN -> STOP/DONE -> BGEN and back. Our historical experiment first allowed a narrowly restricted coexistence of solo windows and grouped MTP. That alone did not restore solo speed: the server's inherited `presence_penalty=1.5` still triggered the engine's serial fallback. Staging accepted per-window penalty histories before final-stage sampling enabled the pipelined solo path while retaining that penalty. Related upstream work: [#3095](https://github.com/Niko1221/Strata/pull/3095) and [#2940](https://github.com/Niko1221/Strata/issues/2940).

The historical switches were `STRATA_HYBRID_WINDOWS=1` and `STRATA_HYBRID_PENALTIES=1`, together with `--pipeline-windows 2`, `STRATA_BATCH_MTP=1` and `STRATA_PIPED_MTP=2`. `tested-hybrid.patch` is their combined diff against the hash recorded in `source-provenance.json`. It requires the custom backported base; it does not apply directly to current main. The new focused PR ports only penalty-history staging behind `STRATA_PIPELINE_PENALTIES=1`; it leaves coexistence to the existing upstream proposal. **No speed or GPU-correctness claim is made for that new port.**

## Validation limits and failures

- The custom binary passed 128K single/dual requests, three mid-generation 1 -> 2 -> 1 cycles, history growth/branching, cancellation/recovery, third-client queueing, GPU image recognition, Chinese output and return to idle GPU utilization. A separate sampled two-client check used temperature 0.7 / presence penalty 1.5. Marker checks establish basic request separation, not model quality.
- A strong-penalty numeric-list probe did not provide a stable serial reference: two serial requests differed, and pipeline outputs differed too. Exact output equivalence was **not established**. The 604-second case was later marked failed by an inappropriate natural-short-answer length assertion; that case remains failed in `measurements.json`.
- Earlier attempts: requested output length exceeded the 128K context boundary (stopped during loading); test directory ownership prevented log creation; the batch HTTP API rejected a `logit_bias` probe with HTTP 400; a natural 57-token EOS failed a test that wrongly required 96 output tokens. These are preserved as failed/no-performance attempts, not GPU failures.
- First adoption attempt failed before inference because a test script's text replacement broke its Python initialization. Automatic rollback restored the prior configuration. A corrected second attempt succeeded. No claim is made that the initial attempt passed.
- Default-path bitwise output parity and quality equivalence across all prompts have not been demonstrated. No full current-main HIP/CUDA engine build, GPU execution of the new port, SYCL build or Metal build was performed for the contribution. Read the PR's current validation list for subsequent checks.

## Files

- `measurements.json`: individual experiment summaries including failed cases.
- `production-summary.json`: adopted-service measurements and accepted configuration hash.
- `request-measurements.jsonl.gz`: compressed numeric per-request measurements, usages, input hashes and SSE arrival timestamps from run3/run6/run7/adoption2. Each line names its original run and file. Prompt/response text is excluded. `quality_pass` in historical summaries denotes marker/sanity checks only.
- `hardware-stats.json`: sensor maxima and minimum available RAM per case.
- `engine-excerpts.log`: selected real pipeline/MTP log lines.
- `historical-config.sanitized.json`: measured settings with local paths replaced by placeholders; not a ready-to-launch config.
- `tested-hybrid.patch` and `source-provenance.json`: historical diff and exact file/binary hashes. Model weights, compiled binaries and unpublished custom base libraries are not included, so this archive alone does not reproduce the exact engine build.
- `mainline-checks.json`, `mainline-cmake-checks.json`: commands/results for the focused port, with machine-local paths replaced. Translation-unit compilation is not full backend compilation or hardware execution. Existing source warnings remain in the logs.
- `SHA256SUMS`: hashes for the archive's files.

The original detailed local archive remains intact. Private machine paths, identifiers and application configuration are not published here. This export is a selected evidence archive, not a copy of the entire operations workspace.
