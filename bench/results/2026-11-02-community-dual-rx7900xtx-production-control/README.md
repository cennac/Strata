# Optimized RDNA3 production control versus the mainline MTP integration

The deployed custom RDNA3 build remains faster than the mainline integration in
PR #4966. The PR covers request-path switching; it does not contain the custom
backend's complete performance patch set. Its opt-in/off comparison must not be
presented as reproducing or superseding the optimized custom build.

## Measurements

Same machine, dual RX 7900 XTX, Qwen3.8-Flash-Next IQ3_XXS, 130,020-token synthetic
input, 512-token output cap except the 128-token staggered request. Rates are HTTP
end-to-end completion-token rates. Dual means aggregate throughput, not per client.

| Stack and workload | Warm solo t/s | Two requests, aggregate t/s | Statistic |
| --- | ---: | ---: | --- |
| Optimized production, original benchmark | 114.58 | 153.63 | 3 solo runs, mean; 5 dual pairs over 33.33 seconds |
| Optimized production, PR switching benchmark | 134.11 | 164.35 | 3 solo-after runs, median; 3 dual pairs, median |
| Mainline integration opt-in, earlier saved run | 93.01 | 136.18 | Same switching script and median definitions |

The switching workload's production solo-before runs were 135.18, 133.79 and
135.54 t/s. Solo-after runs were 114.48, 134.11 and 134.27 t/s. All runs, including
the lower first solo-after result, are retained. Dual pairs were 164.31, 168.33 and
164.35 t/s. The original benchmark's warmup is retained in original-workload.json.

The switching script's two clients share one prompt. The original benchmark uses
two distinct prompts and starts with a simultaneous warmup. These are different
workloads, so use the two switching rows for the direct script comparison. The
production control ran after the saved candidate experiment, on the same day;
these were not randomized/interleaved measurements.

Production retained its original configuration, including GPU vision. The earlier
isolated candidate dual arms disabled vision. The two stacks also have different
server/engine revisions and backend implementations. This is not a one-variable
kernel ablation or proof of equal generated output. Different output can change
MTP acceptance. No per-kernel contribution percentage is established.

## Code and build identity

- Production binary SHA256:
  `3029ce1dcbcf402be45b24cc4fac5ace2b8a4aa58da8555d073951d45eb85710`.
- Production config SHA256:
  `85cc928fdbe4515013549a692562ddd479d4e3c85a9b501e3cd46069c22eee54`.
  production-config.json replaces absolute paths with placeholders.
- [Complete custom code and synthetic fixtures](https://github.com/cennac/Strata/tree/3eb5752b18c30e633e144511d919b95bd64e1959).
  Keep its existing upstream history and attribution; the backend work is not
  introduced by the new switching PR.
- [Mainline integration code](https://github.com/cennac/Strata/tree/4398ff2e8ee44768104587585a5ce6a7e77e99d3).
- [Earlier candidate raw data, config and full build records](https://github.com/cennac/Strata/tree/078146edb81c809722884e8e21dcc98e41fba4ed/bench/results/2026-11-02-community-dual-rx7900xtx-mtp-switch).

## Source inspection: paths not ported

The old tree compiles src/kernels/cuda/rdna_gr.cu. The submitted mainline tree has
neither that file nor its CMake entry. The old RDNA GR/Q8/A8/inject/fused controls,
STRATA_RDNA_MMVQ_WAVE selection in native_mmvq.cu, and STRATA_WAIT_FUSE selection
in verify.cpp are not implemented under those controls in the new tree. The
candidate config still contains the variables; that does not restore their code.

This is a source-inspection inventory. Attribution of the measured performance
gap to any individual path remains **unconfirmed static analysis** pending
isolated ports and controlled ablations. It is not a claim that all mainline
RDNA support or all old optimizations are absent.

## Request commands

On the already-running unchanged production service, using the saved scripts:

```sh
python community/benchmark.py --model Qwen3.8-Flash-Next-IQ3_XXS \
  --singles 3 --dual-seconds 30 --output original-workload.json

python tools/bench_mtp_switch.py --endpoint http://127.0.0.1:8100 \
  --model Qwen3.8-Flash-Next-IQ3_XXS \
  --build-id 3029ce1dcbcf402be45b24cc4fac5ace2b8a4aa58da8555d073951d45eb85710 \
  --topology dual-gpu --scenario switch --prompt-file prompt.txt \
  --tokens 512 --short-tokens 128 --repeats 3 --timeout 600 \
  --output new-workload.json
```

prompt.txt is the user content from community/fixtures/ctx131072-route0.json.gz
in the fixed complete-source commit above. Its SHA256 is recorded by the
switching script. Scripts refuse to overwrite existing output files.

## Checks and limits

Both commands returned zero. The original benchmark's marker checks passed.
The switching workload completed 14 requests and observed the 1-to-2-to-1 stream
sequence. It saves hashes rather than response text and does not establish
semantic quality or token/logit parity. The candidate's earlier output-equality
failure remains unresolved; this production control does not clear it.

Hotspot peaks in sysfs enumeration order were 77 and 82 C; minimum available RAM
was 16.42 GiB. No thermal/memory guard fired. Production was not restarted or
reconfigured. Config, power caps and fan modes were unchanged; loaded=true and
in-flight=0 at completion. result.json and telemetry.json retain the checks.

The old evidence directory is preserved. This appended control does not change
which engine was tested in the submitted PR. The mainline PR stays draft and is
not the recommended replacement for the deployed optimized build.
