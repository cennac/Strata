# 双 7900 XTX：完整实验源码与复现说明

这个分支保存本机测出**单请求 116.48 token/s、双请求合计 161.08 token/s**那套自定义引擎的完整源码组合，并提供独立构建、配置生成和公开合成输入的测速脚本。它与只处理惩罚历史的上游草稿 PR #4909 不同；下载这个分支无需再手工叠加 #4909 或 #3095。

模型：Qwen3.8-Flash-Next GSQ-RCO **IQ3_XXS**，不是 27B 或 IQ3_S。每请求上限 131072，2 个活跃 HTTP 请求；内部 4 槽、2 组。单请求已使用两张卡。

**116/161 是原已部署二进制的历史实测值，不能保证换机器、换驱动或重新编译后完全相同。**本次完整源码重建的验证结果另见 [validation.json](validation.json)，不要把历史二进制与本次重建二进制混为一谈。

单请求、双请求每路速度，以及独立双副本的历史实测另见[同模型对照说明](comparisons/2026-11-01-single-dual/README.zh-CN.md)，其中明确区分同轮对照和不同日期的历史记录。

## 源码组合与归属

| 部分 | 来源 / 本分支内容 |
| --- | --- |
| RDNA3 基础与内核 | [nicejunjie/Strata-rdna3](https://github.com/nicejunjie/Strata-rdna3/tree/fc8f82334239199cd3cfcfa5ab5b8ac09827d19b)，固定 fc8f82334239199cd3cfcfa5ab5b8ac09827d19b |
| AVX2 初始化兼容 | 该仓库 PR #1 的 iq_avx512.cpp 修正 |
| HIP 锁页复制、DIRECT 槽位复制 | 已实测版本的 conversation_copy / snapshot / state 完整实现 |
| 双流流水 MTP | 从 [Strata 99866a47a2b8](https://github.com/Niko1221/Strata/commit/99866a47a2b8)移植的 deferred commit 与分组调度；保留原作者归属 |
| 单流与 batch 共存 | 本机历史实验版的受限开关 STRATA_HYBRID_WINDOWS |
| 带惩罚参数的单流流水 | 本机实测 STRATA_HYBRID_PENALTIES；主线移植另见 [PR #4909](https://github.com/Niko1221/Strata/pull/4909) |
| HTTP 服务 | 本机实测时使用的 serve、strata_control、strata_cli 和 tools 源码；版本较引擎基础更新 |

引擎版本字符串仍为 0.1.43，身份请按提交、源码清单和二进制 SHA256 判断。上游原有许可证保留。[source-provenance.json](source-provenance.json)记录关键文件、依赖和二进制的哈希。

## 适用环境

- 当前验证范围：Linux x86_64、AVX2 CPU、两张 RX 7900 XTX 24GB / gfx1100。
- 本机约 61.5 GiB 可用总内存（64GB），模型及 PLE 在 SSD；原测试白金上限 303W、涡轮 231W、自动风扇。脚本不修改功耗、风扇、PCIe、驱动或 systemd。
- 本机完整重建使用既有 ROCm 10 开发环境，AMD clang 23.0.0git（llvm-project 8f497e0992fb7513f7f78a6f6b6f1056c375e961）、GCC 15.2、Python 3.14.4。构建脚本需要 Python 3.12+；其他 ROCm/编译器版本尚未验证。
- 依赖 llama.cpp 固定 3cf03257f219afbe7334045ff7c6a06ac68c627d、rocWMMA 固定归档及 SHA256。构建脚本自动下载并核验；不需要本机旧 build 目录或私有静态库。
- CUDA、SYCL、Windows、其他显卡没有做这套完整组合的构建和运行验证。

建议为模型、打包产物、依赖和构建目录预留足够 SSD 空间。`--experts-bin` 会额外生成专家文件；16GiB 会话缓存是预算，并不在启动时立即占满。

## 1. 下载与完整构建

```bash
git clone --single-branch --branch community/dual-7900xtx-repro-20261101 \
  https://github.com/cennac/Strata.git strata-dual-xtx
cd strata-dual-xtx

python3 -m venv .venv
.venv/bin/pip install -r community/requirements.lock.txt

# 需要已安装的 ROCm 开发环境、CMake >=3.24、GNU C++ 编译器、make。
# 按实际安装目录修改 /opt/rocm；首次运行会下载固定依赖。
.venv/bin/python community/build.py --rocm /opt/rocm --jobs 4
export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH:-}
build-community/strata --help

# 调度/边界测试；加 --gpus 0,1 才运行两卡逐字节复制测试。
.venv/bin/python community/check_build.py --rocm /opt/rocm --gpus 0,1
```

本分支不是官方通用安装包。**不要运行 setup.sh、START-HERE 或自动升级来替代上述步骤**，它们可能下载另一版官方引擎；也不要拿 .1.43 的字符串判断自己运行的就是这个分支。

## 2. 准备模型与 MTP

本仓库不含主模型、MTP 或 mmproj 权重。准备原版 GSQ-RCO IQ3_XXS 两个分片：[ISTA-DASLab 固定模型版本](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09)。其他量化可能改变内存需求与速度。

以下 `/models`、`/model-data` 为示例路径，请换成自己的目录。第一条打包命令读取两个相邻 GGUF 分片，不能只下载其中一个。

```bash
export STRATA_GGUF_PY="$PWD/.community-deps/llama.cpp/gguf-py"

.venv/bin/python tools/iq_pack.py \
  --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --out /model-data/packs/iq3_xxs --experts-bin

# 只获取原始检查点中 MTP 所需张量；工具有固定版本及哈希校验。
.venv/bin/python tools/mtp_fetch.py fetch --out /model-data/mtp-bf16
.venv/bin/python tools/mtp_fetch.py verify --out /model-data/mtp-bf16
.venv/bin/python tools/mtp_pack.py --src /model-data/mtp-bf16 \
  --experts q2_0 --out /model-data/mtp-q2_0.gguf
.venv/bin/python tools/mtp_rt.py --gguf /model-data/mtp-q2_0.gguf \
  --out /model-data/mtp/rt
cp data/draft_vocab.bin /model-data/mtp/rt/draft_vocab.bin
```

`draft_vocab.bin`是上游公开的 CJK 草稿词表，已保留，且与原实测运行时文件哈希相同。不要漏掉它，否则草稿候选和显存占用可能不同。

## 3. 生成两路各 128K 的配置

```bash
.venv/bin/python community/make_config.py \
  --gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /models/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf \
  --pack /model-data/packs/iq3_xxs --mtp /model-data/mtp/rt \
  --rocm /opt/rocm --gpus 0,1 --output strata-community.json

.venv/bin/python -m serve.server --engine strata \
  --config strata-community.json --host 127.0.0.1 --port 8100 \
  --offline --no-fit-max-tokens
```

配置生成器检查必需文件，拒绝覆盖已有配置。它复制随包提供的合成基准专家热度种子到独立 `community-run/`，配置使用该副本；原模型文件不改动。

生成的配置保留：26/22 层、INT8 KV、32768 resident KV、prefill 16384、16GiB 会话缓存预算、spec 3 / min-p 0.5、4 内部槽 / 2 组、最多 2 活跃请求，以及全部原实测环境开关。其中关键组合为：

```text
--pipeline-windows 2 --batch-groups 2
STRATA_BATCH_MTP=1
STRATA_PIPED_MTP=2
STRATA_SLOT_COPY_DIRECT=1
STRATA_CONVERSATION_PINNED_COPY=1
STRATA_HYBRID_WINDOWS=1
STRATA_HYBRID_PENALTIES=1
```

`parallel=4` 由服务生成 `--batch 4`；`max_active_requests=2` 才是 HTTP 活跃上限。引擎仍为识图预留相关缓冲。未提供图像编码器时，文本服务可用，但不能据此宣称识图已启用。

### 可选：GPU 识图

原测量使用独立的官方 GPU 图像编码器组件 0.1.54（`f-amd-vision 8c1aea4bca1b`，gfx1100，hipBLASLt 100200）及对应 BF16 mmproj；不是把图像识别交给主模型文本进程。

本分支包含 `tools/vision/` 的源代码和 CMake 工程，可独立构建；这个组件的**确切已发行版本**及获取说明见 [vision-component.md](vision-component.md)。不要把读取器使用的 100200 调优表与主引擎 100401 表混用。

已有可用 GPU reader 时，在配置生成命令末尾增加：

```text
--vision-exe /path/to/strata-vision-hip-wrapper \
--mmproj /models/mmproj-Qwen3.8-Flash-Next-BF16.gguf --vision-gpu 0
```

生成器只引用给定文件，不会代装驱动或下载另一版引擎。读图组件可保持自己的运行库包装脚本。是否已启用以 `/v1/status` 的 vision.available 和实际图片请求为准。

## 4. 用公开输入测速

两个压缩 fixture 是程序生成的公开合成文档，不含真实对话或私有材料。每篇约130K输入token，含各自三个独立标记；脚本核验输入哈希和输出标记。模板版本不同，实际计数可能略变，以响应 usage 为准。

本次原版/重建版同日对照均使用这里的公开输入，实际每请求130020输入token。历史116.48/161.08那轮另加了固定实验前缀，实际130037输入token；因此这里只主张本次对照输入一致，不主张与历史轮次逐字相同。原模型卡惩罚默认值继续由服务处理，脚本没有把presence penalty强制清零。

```bash
.venv/bin/python community/benchmark.py \
  --endpoint http://127.0.0.1:8100 --singles 3 --dual-seconds 180 \
  --output benchmark-result.json
```

脚本先发两路文档进行预热，再测三次单请求和持续双请求；每次最多输出512token。结果保留每请求 usage、缓存命中、首字时间、墙钟速度及标记检查。输出文件存在就拒绝覆盖。

- 单请求：整批输出token / 整批墙钟，包含首字等待，三次取算术平均。
- 双请求：整段输出token总和 / 整段墙钟，含批次间开销；是**两路合计**。
- **主要是128K缓存命中后的生成速度，不是完整128K冷预填充速度。**第一次文档请求另存 `cold`，不并入热吞吐；该字段仅表示首次请求，已有服务可能已缓存输入，是否真正冷启动请看 `cached_tokens`。
- MTP接受率依赖文本；不应把这些合成文档的速度当作所有业务保证值，也不能从标记通过推导语义质量等价。

## 验证边界与回退

原部署版此前通过单/双128K、1→2→1切换、历史增长/分支、取消、排队和识图等功能检查。强惩罚逐字一致性仍无结论，604秒历史长测后的短答长度断言失败仍保留。历史数据见[完整证据归档](https://github.com/cennac/Strata/tree/0c6fd25c2eb256e366ba6305cf19396e98c75f39/bench/results/2026-11-01-community-dual-rx7900xtx-hybrid)。

本次完整重建的单请求三次均值为 **113.29 t/s**，双请求持续123.07秒、19批、合计 **158.09 t/s**；原部署版同日对照为 **122.68 / 156.43 t/s**，双路持续32.73秒、5批。持续时间不同，且为一次对照，不能据此判断重建使性能变快或变慢。逐请求数字见 [validation.json](validation.json) 和 [results/](results/)。

本分支的重建检查与历史验收分开列在 validation.json。没有做全面模型质量评测或默认路径逐字一致性证明。源码同时含其他后端文件不代表其他后端已验证。操作与验证记录见 [OPERATIONS.md](OPERATIONS.md)。

采用独立目录、独立配置和独立端口，退出本服务即可回到自己的原部署。不要覆盖现役可执行文件、旧配置或模型；脚本不会替你修改开机服务。服务器版若自己增加远程访问，应使用该服务已有的 API key 选项。

问题反馈建议附：本分支commit、ROCm/编译器、GPU拓扑与功耗、完整启动参数，以及 `community-build.json` 和测速结果；不要上传自己的密钥或真实请求正文。
