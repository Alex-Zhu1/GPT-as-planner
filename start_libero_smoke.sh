#!/usr/bin/env bash
# 用途：原始 LIBERO 单场景回归，启动 MuJoCo/EGL + pi05_libero + Valen + Astra。
# 这不是 LIBERO-MAX 评测入口：MAX 需要动态扰动和 Base/Dynamic 精确前缀配对。
# 环境隔离：不需要 conda activate，每个服务使用自己的虚拟环境 Python。
# 成功标准：完整 instruction 的原生 success=true；不能把服务启动成功算作任务成功。
set -euo pipefail

# 1. 从脚本位置推导服务器目录，允许从任意工作目录调用；含空格路径必须加引号。
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
JEV_ROOT="$(dirname -- "$REPO_ROOT")"
USER_ROOT="$(dirname -- "$JEV_ROOT")"

# 2. 环境与模型位置，可通过同名环境变量覆盖。
# SIM：MuJoCo 2.3.7 / robosuite 1.4.1；不要直接替换成 MAX 要求的环境。
# PI：Python 3.11 + OpenPI/JAX，加载 models/pi05_libero，输出 10×7 动作。
# VALEN：Python 3.12，加载 models/Valen-2B，处理已执行动作后的双相机观测。
# ASTRA：正常账号授权包装器；无需提取、硬编码模型 URL 或复制登录凭据。
JEV_MODEL_ROOT="${JEV_MODEL_ROOT:-$JEV_ROOT/model_}"
JEV_ASTRA_RUNTIME="${JEV_ASTRA_RUNTIME:-$USER_ROOT/astra-runtime}"
JEV_SIM_PYTHON="${JEV_SIM_PYTHON:-$JEV_MODEL_ROOT/.libero-sim-venv/bin/python}"
JEV_PI_PYTHON="${JEV_PI_PYTHON:-$JEV_MODEL_ROOT/.openpi311-venv/bin/python}"
JEV_VALEN_PYTHON="${JEV_VALEN_PYTHON:-$USER_ROOT/envs/jev-valen/bin/python}"

usage() {
    cat <<'EOF'
用法（可从任意目录运行本脚本）：
  bash start_libero_smoke.sh --show-config     # 只显示环境与权重路径
  bash start_libero_smoke.sh --check           # 检查文件、配置、端口，不加载模型
  bash start_libero_smoke.sh                   # 完整任务；仅原生任务成功才算通过
  bash start_libero_smoke.sh --max-decisions 5 # 0..50；0 为原生 episode 时限
  bash start_libero_smoke.sh --max-uncertainty-reviews 0 # 禁用不确定性复核
  bash start_libero_smoke.sh --uncertainty-policy pause # 复现旧版不确定暂停策略
  bash start_libero_smoke.sh --chunk-steps 1    # 1..5，默认 5；常规监控窗口控制步数
  bash start_libero_smoke.sh --output /absolute/path/to/new-run
  bash start_libero_smoke.sh --planner-backend api --api-base-url http://127.0.0.1:8317/v1
  --api-key-file PATH 指定本地模型 API 访问密钥；默认仍使用 Codex agent

环境：仿真 Python 3.10；Pi/OpenPI Python 3.11；Valen Python 3.12。
无需切换 conda，也无需 source activate；直接使用各环境的 Python。
可覆盖变量（请使用绝对路径）：JEV_MODEL_ROOT、JEV_ASTRA_RUNTIME、
JEV_SIM_PYTHON、JEV_PI_PYTHON、JEV_VALEN_PYTHON。

Astra 由已有 run-with-astra 包装器提供登录、代理和账号锁，固定
gpt-6-astra / xhigh。无需提取服务 URL；不要再用 run-with-astra 包裹此脚本。
本地权重直接加载，不下载；服务只绑定 127.0.0.1，端口 8001/8003/18830。
Smoke 默认 continue：有效但不确定的检测继续执行受限动作段，持续观测；确认失败时调用 Astra。
不会降低 Valen 阈值或把不确定标成安全。监控错误、过期证据仍暂停。
pause 模式最多 10 次 Astra 不确定性复核，每次只能批准 1 步学生动作或停止。
结束或 Ctrl-C 后监督器回收本次服务。退出码 2 表示 smoke 未完成，
例如 Valen 不确定暂停；查看输出目录 controller/smoke_report.json。
EOF
}

ARGS=()
SHOW_CONFIG=0
PLANNER_BACKEND=codex
# 3. 只转发明确支持的参数；Python 入口进一步验证数值范围。
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --show-config) SHOW_CONFIG=1; shift ;;
        --check) ARGS+=("$1"); shift ;;
        --max-decisions|--max-uncertainty-reviews|--uncertainty-policy|--chunk-steps|--output|--planner-backend|--api-base-url|--api-key-file)
            if (($# < 2)); then printf '缺少参数值: %s\n' "$1" >&2; exit 2; fi
            [[ "$1" != --planner-backend ]] || PLANNER_BACKEND="$2"
            ARGS+=("$1" "$2"); shift 2 ;;
        *) printf '未知参数: %s（使用 --help）\n' "$1" >&2; exit 2 ;;
    esac
done

for path in "$JEV_MODEL_ROOT" "$JEV_ASTRA_RUNTIME" "$JEV_SIM_PYTHON" "$JEV_PI_PYTHON" "$JEV_VALEN_PYTHON"; do
    if [[ "$path" != /* ]]; then printf '配置路径必须为绝对路径: %s\n' "$path" >&2; exit 2; fi
done

cat <<EOF
Repository:      $REPO_ROOT
Simulation:      $JEV_SIM_PYTHON
Pi/OpenPI:       $JEV_PI_PYTHON
Pi checkpoint:   $JEV_MODEL_ROOT/models/pi05_libero
Valen:           $JEV_VALEN_PYTHON
Valen weights:   $JEV_MODEL_ROOT/models/Valen-2B
LIBERO assets:   $JEV_MODEL_ROOT/sources/LIBERO
Astra wrapper:   $JEV_ASTRA_RUNTIME/run-with-astra
Planner backend: $PLANNER_BACKEND (api does not start the Astra wrapper)
Astra model:     gpt-6-astra / xhigh
Rendering:       EGL, GPU 0
Uncertainty:     continue by default (--uncertainty-policy pause restores legacy behavior)
Normal chunk:    5 controls by default (--chunk-steps overrides); reviewed continuation: 1 control
EOF
if ((SHOW_CONFIG)); then exit 0; fi

[[ "$PLANNER_BACKEND" == codex || "$PLANNER_BACKEND" == api ]] || { printf '无效 planner backend\n' >&2; exit 2; }
EXECUTABLES=("$JEV_SIM_PYTHON" "$JEV_PI_PYTHON" "$JEV_VALEN_PYTHON")
if [[ "$PLANNER_BACKEND" == codex ]]; then
    EXECUTABLES+=("$JEV_ASTRA_RUNTIME/run-with-astra" "$JEV_ASTRA_RUNTIME/codex")
fi
for executable in "${EXECUTABLES[@]}"; do
    if [[ ! -x "$executable" ]]; then printf '缺少可执行文件: %s\n' "$executable" >&2; exit 1; fi
done
for directory in "$JEV_MODEL_ROOT/models/pi05_libero/params" \
    "$JEV_MODEL_ROOT/models/pi05_libero/assets" "$JEV_MODEL_ROOT/models/Valen-2B" \
    "$JEV_MODEL_ROOT/sources/LIBERO/libero/libero/assets"; do
    if [[ ! -d "$directory" ]]; then printf '缺少模型或仿真资源目录: %s\n' "$directory" >&2; exit 1; fi
done
if [[ ! -f "$JEV_MODEL_ROOT/.cache/openpi/big_vision/paligemma_tokenizer.model" ]]; then
    printf '缺少 OpenPI PaliGemma tokenizer 缓存，请先准备模型资源。\n' >&2
    exit 1
fi

# 4. EGL 是无窗口渲染后端，MuJoCo 负责物理仿真。
# 本机服务通信不经过 HTTP 代理；Astra 的外网代理/登录由包装器管理。
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=0
export PYTHONPATH="$REPO_ROOT:$JEV_MODEL_ROOT/sources/LIBERO"
export NO_PROXY=localhost,127.0.0.1 no_proxy=localhost,127.0.0.1
cd -- "$REPO_ROOT"
# 5. Python 监督器按顺序启动并等待服务就绪，随后执行控制器。
# 默认任务：libero_spatial / task 0 / initial state 0 / seed 7。
# 每个 π chunk 提供 10 步，普通执行前 5 步后重新观测，直到成功或原生时限。
# continue 仅放行有效的未知证据；确认失败调用 Astra，监控异常仍暂停。
# 输出 runtime/libero-smoke-*/controller/smoke_report.json；Ctrl-C 回收本次进程组。
exec "$JEV_SIM_PYTHON" -m hybrid_rollout.libero.launch_smoke \
    --model-root "$JEV_MODEL_ROOT" --astra-runtime "$JEV_ASTRA_RUNTIME" \
    --controller-python "$JEV_SIM_PYTHON" --pi-python "$JEV_PI_PYTHON" \
    --valen-python "$JEV_VALEN_PYTHON" ${ARGS[@]+"${ARGS[@]}"}
