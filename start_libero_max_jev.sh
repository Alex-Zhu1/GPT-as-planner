#!/usr/bin/env bash
# MAX/JEV launcher: current asynchronous instruction planning and explicit legacy modes.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODEL_ROOT="${JEV_MODEL_ROOT:-$(dirname -- "$REPO_ROOT")/model_}"
export JEV_MAX_GL="${JEV_MAX_GL:-osmesa}"

usage() {
    cat <<'EOF'
用法：bash start_libero_max_jev.sh 模式 [新绝对输出路径] [参数]
  --continuous       新版：从初始状态运行异步监测，比较无/有 Astra 的 Dynamic
  --continuous-fork  新版：同状态分支；输出路径后必须给 controls.jsonl 绝对路径
  --run              旧版：同步监测，Base/Dynamic × 无/有 Astra 四组
  --dynamic          旧版：Dynamic 对比；加 --continuous 可切换新版
  --fork             旧版：同状态分支；加 --continuous 可切换新版
参数：
  --output PATH       替代位置输出参数（不能重复提供）
  --model-root PATH   模型根目录；默认使用 JEV_MODEL_ROOT
  --max-root PATH     LIBERO-MAX 根目录
  --port-offset N     端口偏移；默认使用 JEV_MAX_PORT_OFFSET
  --planner-url URL   显式复用共享 Astra /replan；不启动或停止该服务
  --planner-backend codex|api  默认保留 Codex agent；api 使用无会话图片模型接口
  --api-base-url URL  模型 API 地址；默认 http://127.0.0.1:8317/v1
  --api-key-file PATH 模型 API 的本地访问密钥文件（不是 Codex auth.json）
  --astra-call-budget N  --progress-budget N  --subgoal-budget N
  --monitor-max-lag N    --max-review-interval N
  --stall-steps N        --stall-min-reports N  --planner-timeout SECONDS
所有未知、多余或缺值参数均报错；不支持 --check，不会把它当作成功校验。
每组记录实际配置、逐步日志和视频。只有原生任务成功才算成功。
EOF
}
fail() { printf '%s\n' "$*" >&2; exit 2; }
if (($# == 0)); then usage; exit 0; fi
MODE="$1"
shift
case "$MODE" in
    --help|-h) usage; exit 0 ;;
    --run|--dynamic|--fork|--continuous|--continuous-fork) ;;
    *) fail "未知模式：$MODE（使用 --help）" ;;
esac
OUTPUT="" PREFIX="" CONTINUOUS=0
EXTRA=()
[[ "$MODE" != --continuous && "$MODE" != --continuous-fork ]] || CONTINUOUS=1
while (($#)); do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --continuous) CONTINUOUS=1; shift ;;
        --output|--model-root|--max-root|--port-offset|--planner-url|--planner-backend|--api-base-url|--api-key-file|--astra-call-budget|--progress-budget|--subgoal-budget|--monitor-max-lag|--max-review-interval|--stall-steps|--stall-min-reports|--planner-timeout)
            (($# >= 2)) || fail "缺少参数值：$1"
            [[ "$2" != --* ]] || fail "缺少参数值：$1"
            case "$1" in
                --output) [[ -z "$OUTPUT" ]] || fail '输出路径重复'; OUTPUT="$2" ;;
                --model-root) MODEL_ROOT="$2" ;;
                --max-root) [[ "$2" == /* ]] || fail 'max-root 需要绝对路径'; EXTRA+=("$1" "$2") ;;
                *) EXTRA+=("$1" "$2") ;;
            esac
            shift 2 ;;
        -*) fail "未知参数：$1" ;;
        *)
            if [[ -z "$OUTPUT" ]]; then OUTPUT="$1"
            elif [[ ( "$MODE" == --fork || "$MODE" == --continuous-fork ) && -z "$PREFIX" ]]; then PREFIX="$1"
            else fail "多余的位置参数：$1"
            fi
            shift ;;
    esac
done
[[ "$JEV_MAX_GL" == osmesa || "$JEV_MAX_GL" == egl ]] || fail '无效渲染后端'
[[ "$MODEL_ROOT" == /* ]] || fail 'model-root 需要绝对路径'
if [[ "$MODE" == --fork || "$MODE" == --continuous-fork ]]; then
    [[ "$PREFIX" == /* && -f "$PREFIX" ]] || fail '需要 controls.jsonl 的绝对路径'
    EXTRA+=(--fork-prefix "$PREFIX")
fi
[[ "$MODE" != --run || "$CONTINUOUS" == 0 ]] || fail '新版初始状态入口请使用 --continuous；--run 保留旧版四组协议'
OUTPUT="${OUTPUT:-$REPO_ROOT/runtime/max-jev-${MODE#--}-$(date +%Y%m%d-%H%M%S)-$$}"
[[ "$OUTPUT" == /* ]] || fail '输出需要绝对路径'
[[ ! -e "$OUTPUT" ]] || fail "输出目录必须是新目录：$OUTPUT"
[[ "$MODE" == --run ]] || EXTRA+=(--dynamic-only --progress-review)
((CONTINUOUS == 0)) || EXTRA+=(--continuous)
cd "$REPO_ROOT"
exec "$MODEL_ROOT/.libero-max-venv/bin/python" -m hybrid_rollout.libero.max_compare \
    --model-root "$MODEL_ROOT" --output "$OUTPUT" "${EXTRA[@]}"
