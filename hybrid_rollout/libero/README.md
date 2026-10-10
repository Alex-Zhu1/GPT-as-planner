# LIBERO + MuJoCo + Pi 0.5 + Valen

An independent experimental entry point for the same asynchronous monitoring and
failure-triggered recovery method. It does not change the RoboDojo or RoboLab
evaluation entry points, results or published scores.

The simulator and controller share one owning thread. Pi inference and Valen
monitoring use workers, with at most one unreviewed chunk. Confirmed failures
pause simulation, discard speculative inference, capture the same scene again,
and call an explicitly configured HTTP recovery Planner. Replanning never resets
or restores the episode. Only native LIBERO success counts as success.

## Compatibility

Use **`pi05_libero`**, available at `gs://openpi-assets/checkpoints/pi05_libero`.
The RoboDojo ARX X5 checkpoint and its normalization assets cannot be reused.
Download the complete checkpoint, including `params/` and `assets/`, on a host
with access to the upstream bucket. The PaliGemma tokenizer must also be available
in OpenPI's cache; set `OPENPI_DATA_HOME` for an offline cache.

The implementation follows these upstream interfaces:

- [OpenPI LIBERO evaluation](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)
- [OpenPI LIBERO input/output transforms](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py)
- [OpenPI LIBERO setup and checkpoint](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)
- [LIBERO environment wrapper](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/env_wrapper.py)

Use an OpenPI revision with `pi05_libero`: 10-step horizon, seven output actions.
This integration's policy server identifies its config, checkpoint digest,
robot profile and inference sequence. The controller rejects a generic unlabelled
server and stale sessions. Start a fresh policy server for each trial.

Observation contract: `agentview` and `robot0_eye_in_hand` RGB, both rotated 180
degrees to match OpenPI evaluation; 224x224 padded policy images; eight measured
state values (EEF xyz, axis-angle xyz, two gripper joint positions). Valen sees
the two actual views, not a fabricated second wrist camera. Actions are normalized
Panda OSC_POSE inputs: translation xyz, rotation xyz, gripper. They are not joint
positions or absolute EEF poses. Values sent to the native controller are clipped
to [-1,1], with raw proposals and executed controls saved separately. Negative
gripper control opens; positive closes. No extra inversion or binarization is used.

## Environments

Keep model environments separate:

1. **Simulation/controller**: Python 3.10, a working LIBERO/robosuite/MuJoCo install,
   the packages in `requirements-libero.txt`, and `openpi-client` from the same
   OpenPI checkout as the policy server. `requirements-libero.txt` is an initial
   inference-only recipe, not a GPU-validated environment lock. It does not install
   the LIBERO source/assets. Install that checkout editable with pip after its
   dependencies. The upstream Python 3.8 training lock is not suitable for this
   Python 3.10+ controller. Validate imports on the deployment host.
2. **Pi**: the official compatible OpenPI/JAX environment and LIBERO weights.
3. **Valen**: its own CUDA/PyTorch environment, Transformers 5.4.0 as required by
   Valen-2B, torchvision, Pillow and av. The supplied HTTP adapter uses the model's
   native `predict(..., execution="shared_state")`, retaining FP32 parameters and
   the model's BF16 backbone compute. Only use trusted local checkpoint code.

Example simulator environment preparation (Linux, after obtaining LIBERO/OpenPI):

```sh
conda create -n jev-libero python=3.10 pip
conda activate jev-libero
# CPU torch is sufficient for LIBERO initial-state loading; Pi/Valen run separately.
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-libero.txt
python -m pip install -e /absolute/path/to/LIBERO
python -m pip install -e /absolute/path/to/openpi/packages/openpi-client
python -m pip check
```

MuJoCo rendering is separate from CUDA model inference. Use `--gl egl` on a Linux
host with working EGL, or `--gl osmesa` with the system OSMesa library installed
for CPU rendering. H100 availability alone does not establish EGL support; verify
rendering on the target container. macOS uses its native OpenGL backend: omit
`--gl`. `glfw` is intended for a Linux display-backed fallback. No Isaac Sim is used.

## Start one bounded trial

Run all three commands from this repository root, with each process's environment
activated. Valen must be able to read the controller's images at the same absolute
paths. Default service bindings are loopback. The new entry point currently uses
**HTTP Planner only**; the RoboDojo-specific Codex tools and EEF schema are not
silently reused. Provide an Astra-backed endpoint implementing the contract below.
There is no fake/default Planner, credential, or automatic paid-model call in setup.

```sh
# Terminal 1: OpenPI environment
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
python -m hybrid_rollout.libero.pi05_server \
  --checkpoint /absolute/path/to/pi05_libero \
  --identity-output runtime/libero-trial-001-policy.json --port 18830

# Terminal 2: Valen environment (local model and processor files must be complete)
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python -m hybrid_rollout.libero.valen_server \
  --model /absolute/path/to/Valen-2B --port 8001

# Terminal 3: simulation/controller environment, after both services are ready
python -m hybrid_rollout.libero.run \
  --libero-root /absolute/path/to/LIBERO \
  --checkpoint /absolute/path/to/pi05_libero \
  --output runtime/libero-trial-001 \
  --suite libero_spatial --task-id 0 --initial-state-id 0 --seed 7 \
  --student-port 18830 --valen-url http://127.0.0.1:8001/predict \
  --planner-url http://127.0.0.1:8003/replan \
  --chunk-steps 1 --max-decisions 10 --max-replans 3 --gl osmesa
```

Ten startup settling controls are recorded as initialization, excluded from the
policy budget, and executed once before monitoring. Initial-state ID is an index
in the selected task's official initial states, distinct from the RNG seed.
An already-successful/ended initialization is rejected. Use `--max-decisions 0`
for the suite's evaluation horizon after the bounded trial works. A ten-decision
trial will generally be incomplete, which is expected. First JAX compilation
can be slow; timeouts and evidence age are explicit CLI settings.

For a single command, copy `launch.example.json` to `private/libero-launch.json`,
set environment/checkpoint/LIBERO/Planner paths, then run in the controller env:

```sh
python -m hybrid_rollout.robodojo.jev.launch \
  --config private/libero-launch.json --output runtime/libero-managed-001 --check
python -m hybrid_rollout.robodojo.jev.launch \
  --config private/libero-launch.json --output runtime/libero-managed-001
```

This reuses the existing process supervisor, not the RoboDojo simulator. LIBERO
runs inside the controller process. `--check` checks executable paths/config/ports;
it does not load models or prove renderer compatibility. Stop standalone servers
before using the supervisor. All models share GPU 0 in the example; measure actual
peak memory before increasing image history or concurrency.

## Recovery endpoint

The request has the existing Jev envelope: `version`, `request_id`, `run_id`,
`episode_id`, `plan_revision`, `subgoal_id`, `program_id`, `current_scene`,
`failure`, `evidence_frames`, `confirmation_reports`, `previous_attempts`,
`remaining_plan`, `native_request` and `max_chunk_steps`. The `contract` field
specifies LIBERO semantics. `native_request` contains measured EEF/state, two image
paths, a fresh proposed action array, and an `assessment_example`. No reward,
object ground truth, or simulator success is sent to the models.

Return exactly the six identity fields (`request_id` plus the five generation
fields) and `decision`. Outer identities echo the outer request. The inner
`decision.request_id` echoes `native_request.request_id`.

Supported decisions:

- `student`: retry the fresh Pi proposal for `steps` controls.
- `edit`: add six bounded normalized offsets to the proposal and optionally change
  the gripper. Requires visible failure/misalignment evidence. Example edit field:
  `{"delta": [0.1, 0, 0, 0, 0, 0], "gripper": "open"}`. Each offset is at most 0.25
  in magnitude. This is an OSC action correction, not an IK or collision guarantee.
- `replan`: install 1–8 declarative subgoals with `subgoal_id`, `instruction` and
  observable `success_condition`. The instructions really reach Pi; the original
  task remains the final obligation. No action is executed by plan installation.
- `stop`: end incomplete without resetting.

For student/edit/stop provide `request_id`, `mode`, `steps`, `reason`, `assessment`
(matching `assessment_example`, but with the Planner's actual evidence). `edit`
also includes the edit object. For replan provide only `request_id`, `mode`,
`reason`, `subgoals`. Absolute `eef`, joint-space and dual-arm targets are rejected.

Confirmed failure thresholds, uncertain-observation pauses, same-scene recovery,
one unreviewed chunk, and post-correction verification reuse the existing scheduler.
The monitor's confidence thresholds remain uncalibrated for LIBERO.

## Validation and artifacts

```sh
python -m pytest hybrid_rollout/libero/test_integration.py \
  hybrid_rollout/robodojo/test_jev.py hybrid_rollout/robodojo/test_jev_launch.py -q
```

These CPU fixtures test protocol mapping and scheduler invariants, not real
LIBERO rendering, GPU inference, or task success. No end-to-end LIBERO GPU result
is claimed by this local adaptation.

Inspect `selection.json`, `run.json`, `observations/`, `proposal_*.npz`,
`execution_*.npz`, `jev_events.jsonl`, `jev_summary.json`, and `result.json`.
`result.success` comes only from the native task check. Native horizon exhaustion
can produce a completed episode with success false; CLI exit 0 requires success.
Interrupted or uncertain execution is incomplete. Output directories must be new.
