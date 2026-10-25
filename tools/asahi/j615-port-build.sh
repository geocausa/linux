#!/usr/bin/env bash
# Build only; never install a kernel, modify boot files, or reboot.
set -euo pipefail
source_tree=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
build_dir=${1:?Usage: j615-port-build.sh /absolute/output/directory}
[[ $build_dir = /* ]] || { echo "Use an absolute output directory." >&2; exit 2; }
build_dir=$(realpath -m -- "$build_dir")
[[ $build_dir != "$source_tree" && $build_dir != "$source_tree/"* ]] || {
    echo 'Use an absolute output directory outside the source tree.' >&2
    exit 2
}
mkdir -p -- "$build_dir"
if [[ ! -e $build_dir/.config ]]; then
    cp -- "$source_tree/arch/arm64/configs/j615_lab.config" "$build_dir/.config"
fi
make -C "$source_tree" O="$build_dir" olddefconfig
for expected in CONFIG_RUST=y CONFIG_PAGE_SIZE_16KB=y CONFIG_DRM_ASAHI=m CONFIG_DRM_ASAHI_M3_T8122=y CONFIG_DRM_APPLE=m; do
    grep -Fxq -- "$expected" "$build_dir/.config" || {
        echo "Configuration does not retain the required setting: $expected" >&2
        exit 2
    }
done
make -C "$source_tree" O="$build_dir" -j"${JOBS:-4}" Image modules apple/t8122-j615.dtb
