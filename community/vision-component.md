# 可选的GPU识图组件

文本引擎已经从本分支完整重建并测速。识图使用另一个独立进程；本次文本重建验证复用了已安装的官方reader，没有声称这个reader也由本分支重新编译验证。

原实际使用的官方组件来源：

- [Strata v0.2.8 发布页](https://github.com/Niko1221/Strata/releases/tag/v0.2.8)
- [rx7900 0.1.54 完整组件包](https://github.com/Niko1221/Strata/releases/download/v0.2.8/strata-engine-0.1.54-linux-x86_64-hip-rx7900.tar.gz)
- 包大小：250580785 bytes。
- SHA256：`4d2833cd95a4912e7e7794f32baa7a00a69830a6980d4ac8a28ef57707b12ee1`。
- BUILD.json：`f-amd-vision 8c1aea4bca1b`；gfx1100/gfx1101/gfx1102；hipBLASLt 100200。

解压到自己的独立目录；只使用其中的 `strata-vision-hip` 及配套库，不用包内的主引擎替换本分支构建出的主引擎。先运行 `sha256sum` 核对上面的归档哈希。

在组件目录创建包装脚本（`/opt/rocm`按自己的安装路径修改）：

```bash
#!/usr/bin/env bash
set -euo pipefail
reader_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
export LD_LIBRARY_PATH="$reader_dir:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"
export HIPBLASLT_TENSILE_LIBPATH="$reader_dir/hipblaslt/library"
exec "$reader_dir/strata-vision-hip" "$@"
```

保存为 `strata-vision-hip-wrapper`，赋予执行权限后，将其路径和对应BF16 mmproj传给 `community/make_config.py --vision-exe ... --mmproj ... --vision-gpu 0`。mmproj取自主模型仓库，权重不包含在这里。主引擎使用本分支提供的100401调优表；不要把它交给使用100200库的reader。

本分支也保留了 `tools/vision/` 的完整上游源码与CMake入口。可按该目录CMake文件说明，以固定llama.cpp依赖、`STRATA_VISION_HIP=ON` 和 `STRATA_VISION_HIP_ARCHS=gfx1100`独立构建；**这条reader源码构建路径尚未在本轮验证**。组件包的构建身份与本分支的reader源码快照也不能视为逐字相同。
