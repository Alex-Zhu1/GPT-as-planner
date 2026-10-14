#!/usr/bin/env bash
# LIBERO-MAX 官方纯 π 配对评测入口（不是 Jev 适配器）。
# 数据：固定清单 + Plus/PRO 场景；Base/Dynamic 由官方运行器执行前缀回放与扰动。
# Jev 的 MAX 适配尚未实现；原始 LIBERO + Jev 回归使用 start_libero_smoke.sh。
# 不自动安装/升级依赖，不修改已经验证过的原始 LIBERO 仿真环境。
set -euo pipefail

# 1. 目录与独立环境。可在命令前用同名环境变量覆盖，全部使用绝对路径。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JEV_MODEL_ROOT="${JEV_MODEL_ROOT:-$(dirname -- "$REPO_ROOT")/model_}"
JEV_MAX_ROOT="${JEV_MAX_ROOT:-$JEV_MODEL_ROOT/datasets/LIBERO-MAX}"
JEV_PI_PYTHON="${JEV_PI_PYTHON:-$JEV_MODEL_ROOT/.openpi311-venv/bin/python}"
JEV_MAX_PYTHON="${JEV_MAX_PYTHON:-$JEV_MODEL_ROOT/.libero-max-venv/bin/python}"
JEV_CHECK_PYTHON="${JEV_CHECK_PYTHON:-$JEV_MODEL_ROOT/.libero-sim-venv/bin/python}"
JEV_OPENPI_ROOT="${JEV_OPENPI_ROOT:-$JEV_MODEL_ROOT/sources/openpi}"
# 官方 OpenPI 运行器导入 cosmos_policy 中的 LIBERO 工具函数；只需依赖，不加载 Cosmos 权重。
# 该目录内应有 cosmos-policy/ 源码；运行检查会真实 import，缺失时明确报错。
JEV_MAX_COSMOS_DEPS="${JEV_MAX_COSMOS_DEPS:-$JEV_MAX_ROOT/.deps}"
# 本机 NVIDIA 535 EGL 读回已复现 SIGABRT；OSMesa 只替换渲染，物理版本不变。
JEV_MAX_GL="${JEV_MAX_GL:-osmesa}"
[[ "$JEV_MAX_GL" == osmesa || "$JEV_MAX_GL" == egl ]] || { echo 'JEV_MAX_GL 必须是 osmesa 或 egl' >&2; exit 2; }
export JEV_MAX_GL LP_NUM_THREADS=2

usage() {
    cat <<'EOF'
LIBERO-MAX：官方纯 π Base/Dynamic 配对评测。

用法：
  bash start_libero_max.sh --show-config
  bash start_libero_max.sh --check-data                   # 不加载模型，只校验数据
  bash start_libero_max.sh --check                        # 数据 + 环境 + 端口检查
  bash start_libero_max.sh --run --source plus --limit 1  # 一对样本的集成检查
  bash start_libero_max.sh --run --source pro --limit 1
  bash start_libero_max.sh --run --track lite --source plus --limit 0
  bash start_libero_max.sh --run --track lite --source pro --limit 0

参数：
  --track lite|max       默认 lite；官方分别为 800 / 8000 对案例
  --source plus|pro      默认 plus；两个来源必须分进程运行
  --limit N             默认 1；0 跑所选来源全部案例，正数取清单前 N 个
  --query-interval N    默认 5；每次 π 推理后执行 N 步，不能超过模型动作 horizon
  --port N             默认 18840；单 GPU 0
  --output ABS_PATH    新建输出目录；默认 runtime/libero-max-pi-时间戳-进程号

环境变量：JEV_MODEL_ROOT、JEV_MAX_ROOT、JEV_PI_PYTHON、JEV_MAX_PYTHON、
JEV_CHECK_PYTHON、JEV_OPENPI_ROOT、JEV_MAX_COSMOS_DEPS、JEV_MAX_GL（默认 osmesa）。
MAX 仿真要求 MuJoCo 3.2.6 / robosuite 1.4.0，请单独准备 .libero-max-venv。
不要将当前 MuJoCo 2.3.7 的 .libero-sim-venv 当作 MAX 运行环境。
π 使用已有 pi05_libero 权重；动作 horizon 取实际 OpenPI 配置并打印，不硬编码论文设置。
不需要 conda activate；不需要下载新权重；本入口不启动 Valen/Astra。

注意：退出 0 只表示官方运行器完成，不保证机器人任务成功。
查看 cases/*/paired_summary.json；子集检查不能报告为完整 Lite/Max 成绩。
Jev + MAX 仍需适配动态扰动与配对接口，不能靠更换数据路径完成。
EOF
}

# 2. 默认仅显示帮助，避免误启动完整数据集。--run 是执行评测的明确开关。
MODE=help TRACK=lite SOURCE=plus LIMIT=1 QUERY_INTERVAL=5 PORT=18840 OUTPUT=""
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --show-config) MODE=config; shift ;;
        --check-data) MODE=data; shift ;;
        --check) MODE=check; shift ;;
        --run) MODE=run; shift ;;
        --track|--source|--limit|--query-interval|--port|--output)
            if (($# < 2)); then printf '缺少参数值：%s\n' "$1" >&2; exit 2; fi
            case "$1" in
                --track) TRACK="$2" ;; --source) SOURCE="$2" ;; --limit) LIMIT="$2" ;;
                --query-interval) QUERY_INTERVAL="$2" ;; --port) PORT="$2" ;; --output) OUTPUT="$2" ;;
            esac
            shift 2 ;;
        *) printf '未知参数：%s\n' "$1" >&2; exit 2 ;;
    esac
done
if [[ "$MODE" == help ]]; then usage; exit 0; fi
[[ "$TRACK" == lite || "$TRACK" == max ]] || { echo 'track 必须为 lite 或 max' >&2; exit 2; }
[[ "$SOURCE" == plus || "$SOURCE" == pro ]] || { echo 'source 必须为 plus 或 pro' >&2; exit 2; }
for value in "$LIMIT" "$QUERY_INTERVAL" "$PORT"; do
    [[ "$value" =~ ^(0|[1-9][0-9]{0,6})$ ]] || { echo '数值参数必须为非负十进制整数' >&2; exit 2; }
done
(( QUERY_INTERVAL >= 1 && PORT >= 1 && PORT <= 65535 )) || { echo 'query-interval 或 port 超出范围' >&2; exit 2; }
for path in "$JEV_MODEL_ROOT" "$JEV_MAX_ROOT" "$JEV_PI_PYTHON" "$JEV_MAX_PYTHON" "$JEV_CHECK_PYTHON" "$JEV_OPENPI_ROOT" "$JEV_MAX_COSMOS_DEPS"; do
    [[ "$path" == /* ]] || { printf '需要绝对路径：%s\n' "$path" >&2; exit 2; }
done
[[ -z "$OUTPUT" || "$OUTPUT" == /* ]] || { echo 'output 需要绝对路径' >&2; exit 2; }

MANIFEST="$JEV_MAX_ROOT/benchmark/lite/libero_max_lite.json"
[[ "$TRACK" == lite ]] || MANIFEST="$JEV_MAX_ROOT/benchmark/max8000/libero_max_8000.json"
LIBERO_SOURCE="$JEV_MAX_ROOT/.deps/LIBERO-plus"
[[ "$SOURCE" == plus ]] || LIBERO_SOURCE="$JEV_MAX_ROOT/.deps/LIBERO-PRO"
SOURCE_CONFIG="$JEV_MAX_ROOT/.deps/libero-$SOURCE-config"
cat <<EOF
Mode:             $MODE (pi only; Jev MAX adapter not implemented)
Dataset:          $JEV_MAX_ROOT
Manifest:         $MANIFEST
Source/config:    $SOURCE / $SOURCE_CONFIG
Case limit:       $LIMIT (0 = all selected-source cases)
MAX simulation:   $JEV_MAX_PYTHON
Pi/OpenPI:        $JEV_PI_PYTHON
Pi checkpoint:    $JEV_MODEL_ROOT/models/pi05_libero
Cosmos helpers:   $JEV_MAX_COSMOS_DEPS/cosmos-policy
Execution:        $JEV_MAX_GL rendering; Pi GPU 0; query interval $QUERY_INTERVAL; port $PORT
EOF
[[ "$MODE" != config ]] || exit 0

# 3. 数据校验使用现有轻量 Python，不要求 MAX 运行环境已安装。
[[ -x "$JEV_CHECK_PYTHON" ]] || { echo "缺少数据检查 Python：$JEV_CHECK_PYTHON" >&2; exit 1; }
PYTHONPATH="$JEV_MAX_ROOT/src" "$JEV_CHECK_PYTHON" -m libero_max validate-manifest "$MANIFEST"
"$JEV_CHECK_PYTHON" - "$SOURCE_CONFIG/config.yaml" "$JEV_MAX_ROOT/download-manifest.json" <<'PY'
import json, pathlib, sys
config = json.loads(pathlib.Path(sys.argv[1]).read_text())
for key in ('benchmark_root', 'bddl_files', 'init_states', 'assets'):
    if not pathlib.Path(config[key]).is_dir():
        raise SystemExit('Missing source resource: '+key+' = '+config[key])
lock = json.loads(pathlib.Path(sys.argv[2]).read_text())
print('Data paths ready; recorded MAX revision:', lock['repositories']['LIBERO-MAX']['revision'])
print('This path check does not rehash every asset; download verification files retain the original checks.')
PY
[[ "$MODE" != data ]] || exit 0

# 4. 运行检查：不将原始 LIBERO 环境自动升级，也不忽略缺失依赖。
[[ -x "$JEV_MAX_PYTHON" ]] || { echo "缺少独立 MAX 仿真环境：$JEV_MAX_PYTHON；数据已就绪，但运行尚未就绪。" >&2; exit 1; }
[[ -x "$JEV_PI_PYTHON" ]] || { echo "缺少 Pi Python：$JEV_PI_PYTHON" >&2; exit 1; }
export MUJOCO_GL="$JEV_MAX_GL" PYOPENGL_PLATFORM="$JEV_MAX_GL" MUJOCO_EGL_DEVICE_ID=0
# Plus 的 Wand 图像扰动需要 ImageMagick；服务器原生库解包在用户目录。
# 原始 smoke 不使用这些覆盖；MAX Python 另有只读基础依赖路径，不升级原环境。
if [[ -d "$JEV_MODEL_ROOT/.libero-max-native/lib" ]]; then
    export MAGICK_HOME="$JEV_MODEL_ROOT/.libero-max-native"
    export LD_LIBRARY_PATH="$MAGICK_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export MAGICK_CONFIGURE_PATH="$MAGICK_HOME/etc/ImageMagick-6"
    export MAGICK_CODER_MODULE_PATH="$MAGICK_HOME/lib/ImageMagick-6.9.11/modules-Q16/coders"
fi
export LIBERO_SOURCE_PACKAGE_ROOT="$LIBERO_SOURCE/libero" LIBERO_CONFIG_PATH="$SOURCE_CONFIG"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
# 原生 EGL / MuJoCo 崩溃时尽量输出 Python 调用栈；及时刷新诊断日志。
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
CLIENT_PATH="$JEV_MAX_ROOT/scripts/libero_source_overlay:$JEV_MAX_ROOT/src:$JEV_OPENPI_ROOT/packages/openpi-client/src:$JEV_MAX_COSMOS_DEPS/cosmos-policy"
PYTHONPATH="$CLIENT_PATH" "$JEV_MAX_PYTHON" - "$PORT" <<'PY'
import importlib.metadata as m
import socket, sys
for package, expected in [('mujoco', '3.2.6'), ('robosuite', '1.4.0')]:
    actual = m.version(package)
    if actual != expected:
        raise SystemExit(f'MAX requires {package}=={expected}, found {actual}')
from cosmos_policy.experiments.robot.libero.libero_utils import get_libero_env, get_libero_dummy_action
from libero.libero import benchmark
from openpi_client.websocket_client_policy import WebsocketClientPolicy
with socket.socket() as sock:
    sock.bind(('127.0.0.1', int(sys.argv[1])))
print('MAX runtime imports and port check passed.')
PY
PYTHONPATH="$JEV_OPENPI_ROOT/src:$JEV_OPENPI_ROOT/packages/openpi-client/src" "$JEV_PI_PYTHON" - "$QUERY_INTERVAL" "$JEV_MODEL_ROOT/models/pi05_libero" <<'PY'
import pathlib, sys
from openpi.training import config
from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
horizon = config.get_config('pi05_libero').model.action_horizon
if int(sys.argv[1]) > horizon:
    raise SystemExit(f'Query interval exceeds action horizon {horizon}')
root = pathlib.Path(sys.argv[2])
assert (root/'params').is_dir() and (root/'assets').is_dir(), 'Missing checkpoint'
print('OpenPI imports ready; actual action horizon:', horizon)
PY
[[ "$MODE" != check ]] || exit 0

# 5. 固定来源和样本子集；不改变每个 case 的扰动参数、初态或 policy seed。
OUTPUT="${OUTPUT:-$REPO_ROOT/runtime/libero-max-pi-$(date +%Y%m%d-%H%M%S)-$$}"
[[ ! -e "$OUTPUT" ]] || { echo "输出目录必须是新目录：$OUTPUT" >&2; exit 1; }
mkdir -p "$OUTPUT"
"$JEV_CHECK_PYTHON" - "$MANIFEST" "$SOURCE" "$LIMIT" "$OUTPUT/selected-manifest.json" <<'PY'
import json, pathlib, sys
manifest = json.loads(pathlib.Path(sys.argv[1]).read_text())
selected = [case for case in manifest['cases'] if
            (case.get('substrate_variant', {}).get('benchmark') == 'LIBERO-PRO') == (sys.argv[2] == 'pro')]
if int(sys.argv[3]):
    selected = selected[:int(sys.argv[3])]
assert selected, 'No selected cases'
pathlib.Path(sys.argv[4]).write_text(json.dumps({**manifest, 'cases': selected}, indent=2)+'\n')
print('Selected pairs:', len(selected), '; source:', sys.argv[2])
PY

# 6. 使用官方配对运行器：确定性 π 噪声、Base 前缀回放、原生任务判定。
# 官方脚本负责服务与 worker 的退出回收；当前限定一张 GPU，避免编号重映射歧义。
# 官方服务默认监听 0.0.0.0；应在可信服务器网络使用，切勿当作认证 API 暴露。
export PROJECT_DIR="$JEV_MAX_ROOT" DEPS_DIR="$JEV_MAX_ROOT/.deps"
export OPENPI_DIR="$JEV_OPENPI_ROOT" OPENPI_PYTHON="$JEV_PI_PYTHON" OPENPI_FALLBACK_PYTHON="$JEV_PI_PYTHON"
export CLIENT_PYTHON="$JEV_MAX_PYTHON" CHECKPOINT="$JEV_MODEL_ROOT/models/pi05_libero"
export COSMOS_DEPS="$JEV_MAX_COSMOS_DEPS" LIBERO_IMPL_DIR="$LIBERO_SOURCE"
export LIBERO_OVERLAY="$JEV_MAX_ROOT/scripts/libero_source_overlay"
# 首次 JAX 编译会阻塞服务心跳；仅客户端将心跳等待放宽到 600 秒。
# 使用官方 overlay 接口，不修改下载的 MAX/OpenPI 源码或评测语义。
export OPENPI_CLIENT_OVERLAY="$REPO_ROOT/hybrid_rollout/libero/max_client_overlay"
export JEV_MAX_WS_PING_TIMEOUT=600
printf 'WebSocket client ping timeout: %s seconds\n' "$JEV_MAX_WS_PING_TIMEOUT"
printf '%s\n' "$JEV_MAX_WS_PING_TIMEOUT" > "$OUTPUT/client-ping-timeout-seconds.txt"
printf '%s\n' "$JEV_MAX_GL" > "$OUTPUT/render-backend.txt"
export GPUS=0 PORT_BASE="$PORT" QUERY_INTERVAL RESUME=0
export OPENPI_DATA_HOME="$JEV_MODEL_ROOT/.cache/openpi" XLA_PYTHON_CLIENT_PREALLOCATE=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset MAX_CASES_PER_SHARD POLICY_NOTIFICATION
printf 'Artifacts: %s\n' "$OUTPUT"
exec bash "$JEV_MAX_ROOT/scripts/run_openpi_persistent_benchmark.sh" "$OUTPUT/selected-manifest.json" "$OUTPUT"
