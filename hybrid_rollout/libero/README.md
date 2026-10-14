# LIBERO + MuJoCo + Pi 0.5 + Valen

An independent experimental entry point for the same asynchronous monitoring and
failure-triggered recovery method. It does not change the RoboDojo or RoboLab
evaluation entry points, results or published scores.

The simulator and controller share one owning thread. Pi inference and Valen
monitoring use workers, with at most one unreviewed chunk. Confirmed failures
pause simulation, discard speculative inference, capture the same scene again,
and call an explicitly configured HTTP recovery Planner. LIBERO now defaults to
`--uncertainty-policy continue`: valid unknown evidence is logged without blocking
the next bounded student chunk. This is not a clearance or safety prediction.
Monitor errors/stale evidence still stop, as do unverified recovery corrections.
Replanning never resets
or restores the episode. Only native LIBERO success counts as success.

## Compatibility

### LIBERO-MAX launcher

`start_libero_smoke.sh` remains the original LIBERO + Jev regression entry point.
The separately documented `start_libero_max.sh` wraps the official **pure-Pi**
Base/Dynamic launcher. The separate Jev entry point below runs the four-arm
camera-shift smoke. Run from the repository:

```sh
bash start_libero_max.sh --show-config
bash start_libero_max.sh --check-data --source plus
bash start_libero_max.sh --check-data --source pro
bash start_libero_max.sh --check
bash start_libero_max.sh --run --source plus --limit 1
```

Both data checks were verified on the prepared server. MAX uses MuJoCo 3.2.6 /
robosuite 1.4.0 in `model_/.libero-max-venv/bin/python`, selected by
`JEV_MAX_PYTHON`. This environment shares unchanged base Python dependencies
through a `.pth` path to `.libero-sim-venv`; keep that base environment available.
The original simulator retains MuJoCo 2.3.7 / robosuite 1.4.1. Cosmos Policy
provides LIBERO helper functions only; no Cosmos weights are loaded.
ImageMagick libraries are unpacked under `model_/.libero-max-native`, without
system installation. Versions are recorded in the dataset's
`max-runtime-environment.json`. `--check` verifies actual imports and versions.
The MAX-only client overlay sets a 600-second WebSocket heartbeat timeout to
allow the first JAX compilation; the default timeout interrupted the first run
before a task result. It changes transport timing, not actions or task scoring.
Use `--help` for environment overrides and track/source selection. The default
case limit is one pair; `--limit 0` selects all cases of one source. Plus and PRO
must run separately. A completed launcher is not proof of task success; inspect
the paired summaries.

The server's NVIDIA 535.129.03 EGL path reproduced a native abort inside
`mjr_readPixels`; a separate blocked-render stack entered `libnvidia-eglcore`.
MAX launchers now default to `JEV_MAX_GL=osmesa`, keeping MuJoCo 3.2.6 physics
and robosuite 1.4.0 while using user-local `libOSMesa.so.8` (Mesa 23.2.1).
`JEV_MAX_GL=egl` is available for explicit driver diagnosis, not a validated
default on this server. Pi and Valen still run on GPU. Two-control preflight
state hashes matched exactly between backends; rendered pixels differed by
0.708/255 on average, so comparisons must use one consistent backend.

Server trial on 2026-10-11 (`libero-max-pi-20261011-153227-115779`):
the first Plus Lite case, `libero_10-t0009-camera_shift-d0-p195`, completed the
Base instruction in 244 controls / 49 queries (`success=true`). Dynamic aborted
with SIGABRT (exit 134), leaving an empty terminal trace; the outer launcher
returned 1. This is an incomplete paired smoke, not a measured Dynamic task
failure. No readable core or Python traceback was available. The launcher now
enables Python faulthandler for future diagnosis; this has not resolved the crash.
The OpenPI trace records action hashes but not per-query image/state hashes, so
even completed pairing cannot establish exact input identity from these traces.

### Optional stateless image API backend

The LIBERO planner now supports two selectable backends. Omitting the option
preserves the existing Codex agent and per-episode thread behavior. Add
`--planner-backend api` to use independent model requests with attached camera
images through the local gateway in `tools/codex_api`.

```sh
# New experiment, existing Pi / Valen / simulator setup:
bash start_libero_max_jev.sh --continuous /absolute/new/api-output \
  --planner-backend api \
  --api-base-url http://127.0.0.1:8317/v1 \
  --api-key-file /absolute/path/to/local-proxy-key

# Original behavior remains available and is still the default:
bash start_libero_max_jev.sh --continuous /absolute/new/codex-output \
  --planner-backend codex
```

The default API URL is `http://127.0.0.1:8317/v1`; the default key file is
`runtime/local-codex-api/api-key` under this repository. The gateway must be
reachable from the machine running the planner. The rollout launcher owns its
planner adapter, but does not start or stop the model gateway. API mode does
not require `astra-runtime/codex` or `run-with-astra`. It adds no new Python
dependency: the adapter uses the standard-library HTTP client.

The original LIBERO smoke launcher also accepts these three flags. To run
only the adapter for existing controllers, start it with the controller Python:

```sh
python -m hybrid_rollout.libero.planner_server \
  --planner-backend api --api-base-url http://127.0.0.1:8317/v1 \
  --api-key-file /absolute/path/to/local-proxy-key \
  --output /absolute/new/planner-audit \
  --evidence-root /absolute/path/to/experiment-evidence \
  --port 8104 --timeout 540
```

The adapter keeps the existing `/replan` request/response contract. Controllers
can use `--planner-url http://127.0.0.1:8104/replan`. In the MAX launcher,
`--planner-backend api` together with `--planner-url` verifies that the shared
adapter actually reports the API backend. Configure API URL/key on the shared
adapter rather than passing them to its controller. `shared_astra_service` also
accepts the backend flags; use a separate `--service-root` and port for an API
service so it cannot replace an existing Codex service.

API input is explicitly versioned as `stateless_libero_review_v1`. Every request
includes the current two cameras, selected historical keyframes, their indexed
manifest, and the complete evidence snapshot supplied by the controller.
There is no thread, implicit conversation history, or cross-request image
deduplication. Historical images needed by a new request are attached again.
Instruction-only reviews retain their public-evidence boundary. Bounded-EEF
reviews inline the validated fresh control context that the agent previously
read with a tool, including the measured pose and explicitly unexecuted Pi
proposal. No tool definitions are sent and no tools are executed.

The API backend has an explicit rollout task prompt and the existing structured
decision schema. It does not load AGENTS.md or skills, or start Codex/MCP.
This is not a prompt-identical ablation of the Codex agent: the task rules,
context packaging, and control-context delivery are recorded in the audit.
Model and reasoning remain `gpt-6-astra` / `xhigh`.

Both backends use the same image-root checks, decision identity checks,
subgoal/board validation, and bounded action checks before the controller can
execute anything. Invalid/incomplete/refused responses fail closed; there is
no automatic API retry. Unsupported sampling/token-limit options are not added.
Image format and the gateway's 16 MiB request limit are checked without silently
resizing or discarding evidence.

Host experiment audits remain saved under `planner/api_requests/<identity-hash>`:
input snapshot, image manifest/hashes, explicit prompt, output schema, model
response, validated decision, token usage, and timings. The API key and base64
images are not written to these audit files. Identical completed requests can
reuse their validated result; changing evidence under the same request ID is
rejected. An unresolved/failed request requires reconciliation instead of an
automatic second model call. These audit records are not implicit model history.
`protocol.json`, planner health, and invocation records identify the backend.

### MAX / Jev initial-state asynchronous comparison (current)

```sh
bash start_libero_max_jev.sh --continuous /absolute/new/output \
  --astra-call-budget 12 --progress-budget 80 --subgoal-budget 80 \
  --monitor-max-lag 40 --max-review-interval 160 \
  --stall-steps 30 --stall-min-reports 4 --planner-timeout 600
```

This starts the current asynchronous Pi/Jev controller at the frozen initial
state of `libero_10-t0009-camera_shift-d0-p195`, with monitoring active before
the camera perturbation. It requires no previous run or recorded controls.
It compares two Dynamic episodes: `astra_off/intervention` records monitor
reports without intervention (`continuous_shadow`), and `astra_on/intervention`
pauses on review triggers and asks Astra to judge/revise the instruction or
propose a bounded EEF correction.
Both use the same case and policy seed, but infer Pi actions independently;
this is not a same-state fork comparison. Native original-task success remains
the only success criterion. Astra uses `episode_instruction_eef_v2` and the
instruction planner skill. Action arrays are available only through read-only
diagnostic tools; free action editing and direct Pi-candidate execution are excluded.

All eight controller options shown above are forwarded through the comparison
and worker into `max_run`. `--model-root`, `--max-root`, and `--port-offset` are
also supported. Unknown, extra, or missing arguments fail explicitly; `--check`
is not a validation mode. Output must be a new absolute path.
Use `--planner-url http://127.0.0.1:8103/replan` to reuse an upgraded shared Astra
service. The launcher checks its protocol before starting either arm, records
its health and ownership, and does not launch or clean up another planner. This
avoids competing for the authenticated account lease. Client request/response
audits remain under the Astra-on arm; full tool/prompt audits belong to the
shared service. Without this option the launcher owns an isolated planner.
`--dynamic --continuous` is equivalent to the current initial-state entry; `--run`,
`--dynamic`, and `--fork` without `--continuous` retain the legacy controller.

`protocol.json` records the actual controller settings, planner role/input
format, monitor profile/start point, per-group shadow status/effective Astra
budget, and source hashes including the skill. Each arm's `selection.json`
records the parsed settings and verified planner health. Historical experiment
records are unchanged; the first measured initial-state run is recorded below.

Before an Astra-enabled controller initializes the simulator, it checks
`/health` for readiness, the required conversation format, and
`gpt-6-astra` / `xhigh`. `shared_astra_service` performs the same check before
reusing a live PID, along with the configured port and evidence root. Its
default `--required-protocol` is `episode_instruction_eef_v2`; legacy callers can
explicitly request `episode_instruction_v1` or `episode_thread_delta_v2`. An incompatible existing service
is reported and left running, with no automatic restart. If a newly launched
service fails this check, only that invocation's new process is cleaned up.

Measured on 2026-10-12 in `runtime/max-initial-async-20261012-122324`:
both arms reached 520 controls without native task success. No-Astra retained
103 Jev reports; Astra-on retained 98 (including terminal drains), with maximum
delivered observation lag of five controls in both arms. All seven Astra pause
intervals contained zero executed controls, verified against per-control wall
timestamps. Detection completion preceded the chunk-boundary pause by three or
four controls. Astra supplied explicit grasp/alignment instructions, rejected
three completion-score triggers unsupported by images, and finally restored
the original instruction. These instructions reached Pi but did not produce a
verified lift or task completion in this case.

The first Astra attempt failed before generation because the structured-output
API rejected regex lookaround in the recovery-ID schema. Removing that
lookaround preserves host rejection of the reserved `original-task` ID; 327
offline tests passed. Only Astra-on was rerun from the initial state with the
same controller settings. `comparison-final.json` links the valid no-Astra arm
to `retry-schema-compatible/astra_on` and explicitly excludes the failed
attempt; `report.txt` and `audit.json` retain the analysis and evidence.
The shared Astra service was restored and its new protocol verified afterward.

### MAX / Jev four-arm smoke (legacy)

```sh
bash start_libero_max_jev.sh --run /absolute/new/output
# For a long server run independent of the interactive SSH connection:
nohup bash start_libero_max_jev.sh --run /absolute/new/output > /absolute/new.log 2>&1 < /dev/null &
```

This entry point fixes `libero_10-t0009-camera_shift-d0-p195` and runs
`astra_off/control`, `astra_off/intervention`, `astra_on/control`, and
`astra_on/intervention`. Each arm owns a fresh MuJoCo environment, preserves
the official frozen initial state and camera-change trigger, and has 520
control steps. Pi returns 10 actions and Jev commits at most 5 per monitor
boundary. Dynamic replays its corresponding Base proposals before the change;
Valen remains active. Matching action and image/state hashes are audited after
the run; a mismatch is reported, not silently treated as a valid pair.

Both ablations use the current `uncertainty_policy=continue`, with no uncertainty
review calls. Valid unknown evidence permits another bounded student chunk;
errors and stale evidence still pause. The no-Astra arm uses a fixed stop on
confirmed failure and starts no planner service. The Astra arm uses the
configured `--astra-call-budget` (now default 12) through the existing authenticated
`astra-runtime`; actual invocation counts are reported, including zero.
This isolates Astra availability, not the historical pause-on-unknown policy.

Only native original-instruction success counts as task success. Completing
the process, reaching the horizon, clearing monitor checks, and installing a
recovery plan do not count. No recovery resets or simulator truth are sent to
Valen/Astra. The adapter currently supports `shift_camera`, not arbitrary
instruction replacement/cancellation scenarios.

Each episode stores `controls.jsonl`, `queries.jsonl`, `max-trace.jsonl`,
`jev_events.jsonl`, `result.json`, `frames/`, `execution.mp4`, and `video.json`.
The video is captured live after every control, at 1x simulation speed, with
model waiting time omitted. Empty camera buffers are rejected even if the
upstream stable-render check accepts them. `comparison.json` reports native
outcomes separately from process errors and actual Astra invocations.

Observed on 2026-10-11 in `runtime/max-jev-four-20261011-osmesa`:

| Configuration | Native task success | Controls | Valen reports | Actual Astra calls |
| --- | --- | ---: | ---: | ---: |
| No Astra / Base | true | 234 | 47 | 0 |
| No Astra / Dynamic | false | 520 | 104 | 0 |
| Astra enabled / Base | true | 236 | 48 | 0 |
| Astra enabled / Dynamic | false | 520 | 104 | 0 |

Both Dynamic runs exhausted the budget with the moka pot still beside the lit
stove. All 303 reports were `UNCERTAIN`; no confirmed failure triggered recovery.
Thus this experiment demonstrates no recovery benefit and does not test Astra's
ability to recover after an actual invocation. Both Base/Dynamic pairs passed
the official pre-change action and measured image/state matching checks. Camera
changes occurred at controls 147 and 146 respectively. Independent policy-server
starts differed slightly despite identical initial images/state and fixed noise
seeds (first proposal maximum absolute difference 0.002563); the two-step Base
difference must not be attributed to Astra. This is one case, not a benchmark
success-rate estimate, and uses the current continue-on-uncertainty policy.

All four videos contain every executed control plus the initial frame, verified
by decoding: 235, 521, 237 and 521 frames at 20 fps. Runtime services were cleaned
up. The run's original `launcher_exit=0` records infrastructure completion; the
task results remain false for both Dynamic arms. The launcher has subsequently
been corrected to exit 2 for valid task failures, 1 for infrastructure errors,
and 0 only when every task succeeds. Recorded protocol hashes retain the exact
code used for the measured run; this exit-status correction changes no actions.

### Dynamic progress-review comparison

```sh
bash start_libero_max_jev.sh --dynamic /absolute/new/output
```

This runs live Dynamic episodes from the frozen initial state: the unchanged
no-Astra controller followed by Astra with the progress watchdog enabled. It
does not replay a Base trajectory. The task, seed, checkpoint and 520-control
budget stay fixed; native original-instruction success is still the only score.

Diagnosis: Valen's `noul` is the softmax probability of candidate `true`, read
correctly by the adapter. The old history contains only two endpoints, normally
0.25 simulation seconds apart. Scores between 0.2 and 0.8 are deliberately unknown;
the 0.8 threshold and applicability guard prevent sustained midrange stuck
scores from ever confirming failure. These probabilities are not calibrated for
LIBERO. Repeated `UNCERTAIN` alone previously had no liveness escalation.

The independent watchdog asks for a **progress review**, not a confirmed failure:
over an 80-control window, at least 75% of reports must have attempted-progress
score >=0.8 and stuck score >=0.55. As a backstop, after 160 controls without a
review, an entirely uncertain recent window also requests review. The latter
means progress remains unverified, not that motion or task progress is absent.
Duplicate observations cannot vote; generation changes and evidence gaps reset
the window. A review clears the window, and the configured Astra call budget limits
calls. These are explicit heuristic review settings, not calibrated safety gates.

Astra receives current cameras plus widely spaced historical images from the
window, may install actionable Pi subgoals, and must preserve the original task.
Original Valen reports and thresholds remain unchanged; neither ground-truth
object state nor the intervention flag enters the watchdog or model request.
A progress-driven local retry that remains uncertain is recorded as such and
returns to bounded monitoring; confirmed-failure recovery verification retains
its stricter behavior. `progress_review_trigger`, planner requests/responses,
subgoal instruction records and native results make actual intervention auditable.

Measured run `runtime/max-dynamic-progress-20261011`: no-Astra Dynamic failed at
520 controls (104 uncertain reports); progress-review Astra Dynamic succeeded at
295 controls (49 uncertain and 10 running reports). Exactly one Astra invocation
occurred at control 85, before the camera change at control 138 (baseline: 146).
Initial state/images and all pre-review action chunks matched exactly. The native
original goal, not a planner statement, determined success. Videos decode to
521 and 296 frames, including initial observations, at 20 fps.

Astra proposed three subgoals, but only the first instruction, "Open the gripper
and lift clear of the stove control.", reached Pi: 42 queries and 210 controls.
No `subgoal_completed` event occurred before native task success. Therefore this
run verifies actual intervention and original-task completion, not sequential
verification of the three-subgoal plan or post-perturbation-only recovery.
One fixed case/seed cannot establish a general success-rate improvement.
The run's `plan_installed_without_execution` attempt status describes plan
installation; query/execution logs establish later execution. Subsequent logging
now records executed subgoal controls explicitly, without modifying the measured
run. Runtime services were stopped after both trials.

### Same-state post-change branches

To compare branches after replaying a recorded prefix, use the current
continuous fork entry (the initial-state entry above needs no prefix):

```sh
JEV_MAX_PORT_OFFSET=100 bash start_libero_max_jev.sh --continuous-fork \
  "$PWD/runtime/new-continuous-fork" /absolute/recorded/controls.jsonl
```

Pi commits at most five controls without waiting for Valen. A single detector
worker consumes the latest endpoint and up to six recently captured endpoints;
pending endpoints are coalesced instead of forming an unbounded stale queue.
The comparison's no-Astra arm uses explicit `--continuous-shadow`: all valid
Valen reports are recorded, but do not stop Pi, so it can use the same full
original task budget. A separate no-Astra stop-on-confirmed-failure diagnostic
is not the main completion-rate comparator. Monitor transport errors still
stop and are classified as infrastructure failures, never benchmark successes.
Reports retain their original observation identity and timestamp. A report can
trigger a chunk-boundary pause, but cannot authorize an action in a newer scene.
The host drains old work, obtains a fresh non-stepping observation, and submits
that current scene to Astra. In current task-progress mode no Pi candidate is
generated for Astra: Pi infers only after the instruction decision. No controls execute while
planning. Under the task-progress profile, completion scores request Astra board review;
the constraint-only profile retains fresh paused-scene completion verification.

Progress review and recovery-subgoal execution budgets default to 80 controls.
Constraint state `RUNNING` does not reset either budget. A high advancement
score refreshes the fallback scheduling reference without establishing task progress. Subgoal timeout requests
another review rather than claiming completion. The current task-progress path
stops at a required review if the configured Astra call budget is exhausted
(default 12). Astra receives the original task and remaining episode/subgoal
budgets. Pi receives only the active instruction; Astra retains responsibility
for returning to the original task. Historical measurements below predate this
instruction-only protocol and do not validate its real-model effectiveness.

`controls.jsonl` includes per-control wall timestamps in continuous mode.
`monitor_report` and `pause_confirmed` events distinguish source-observation lag,
detector completion latency, controller delivery latency, and controls after
the controller trigger. Compare detector completion time against control
timestamps to count controls executed after a report became ready. Software
chunk-boundary stopping does not simulate physical actuator braking or inertia;
videos show simulation time and omit model waiting time.

Measured continuous run `runtime/max-continuous-fork-20261011` used case
`libero_10-t0009-camera_shift-d0-p195` and the original instruction
`turn on the stove and put the moka pot on it table 19`. The camera change and
branch occurred at control 146. Both arms had 374 remaining controls; branch
integration/controller/camera/RNG/image checks matched exactly. Actions and
simulation states matched through the first intervention at control 226.

| Arm | Original-task success | Controls | Astra calls |
| --- | --- | ---: | ---: |
| Pi with asynchronous shadow Valen, no Astra | false | 520 | 0 |
| Continuous Jev with Astra recovery | false | 520 | 3 |

All three interventions were execution-budget reviews, not confirmed Valen
failures: original-task review at 226, then recovery-subgoal timeouts at 306 and
386. The active corrections were side-handle grasp/lift, body grasp/lift, then
explicit visual approach followed by body grasp/lift. Each ran for 80 controls;
none produced a verified subgoal transition. At 466 the teacher budget was
exhausted and Pi resumed the original instruction for the final 54 controls.

At the three reviews, the most recent Valen report described controls 221, 301
and 381. Capture-to-report times were 1.003, 0.981 and 0.987 seconds. Three more
controls executed after each report completed, reaching the next chunk boundary.
After the controller trigger, and throughout fresh-scene/Astra planning, zero
controls executed. These are timing measurements of the latest reports, not
latency measurements of a Valen-confirmed failure. The wall pauses were recorded
separately; videos omit those waits and each decode to 521 actual frames at 20 fps.

The original no-Astra stop diagnostic ended at 291 and is excluded from the
full-budget comparison. An earlier recovery trial lost its shared service after
stage1 completed and is classified as infrastructure-interrupted, not a benchmark
failure. Final calls use the standalone-lifecycle shared Astra service; only this
experiment's request IDs and attached images are copied into `audit/`. Both final
arms use OSMesa. This single case establishes the revised execution protocol,
not a recovery success-rate improvement. See `audit/comparison.json`,
`audit/interventions.jsonl`, and each episode's full `execution.mp4`.

```sh
JEV_MAX_PORT_OFFSET=100 bash start_libero_max_jev.sh --fork \
  "$PWD/runtime/new-postchange-fork" \
  "$PWD/runtime/max-dynamic-progress-20261011/astra_off/episodes/intervention/controls.jsonl"
```

The supplied file is an actual recorded no-Astra Pi control sequence. Both arms
execute its identical prefix from the frozen initial state, stopping immediately
after the first camera change (control 146 for this prefix). This reconstructs
controller history and solver warm starts rather than restoring qpos/qvel alone.
The remaining budget is 520-146=374 controls in each arm, and the next policy
noise index is 30. Both discard the unexecuted tail of the old chunk and query
Pi from the same post-change observation. No monitor evaluation or Astra request
is made during the prefix; each monitor/watchdog starts fresh at the branch.

`branch-state.npz` saves MuJoCo `mjSTATE_INTEGRATION` and camera model arrays;
`branch-controllers.json` and `branch-rng.json` save additional numeric controller
and random-generator state. `branch-state.json` compares their hashes, prefix
actions, per-control integration/image hashes, event state, remaining budget and
next policy index. The Astra branch refuses to run on any reference mismatch.
`prefix-checks.json` provides each checkpoint, including the initial observation.
Videos contain both the actually executed shared prefix and subsequent branch.
The original task's native predicate remains the only success criterion.

Port offset 100 uses 18940/8101/8103 so unrelated experiments on default ports
are left running. These trials assess post-change recovery on this one case;
they do not change monitoring thresholds to obtain a desired result.

For an already-running Astra service, the controller can use `--planner-url`,
`--planner-timeout 1200` (including serial queue time), and `--planner-audit`
to record only its own request IDs and responses. Place the episode output in a
new isolated subdirectory of that service's configured evidence root. Exclude
the shared planner from the launcher's managed services and reserved ports;
cleanup must never terminate it. The stage1 batch enumerates its manifest, so
an external episode directory is not a new stage1 case. Shared calls can add
queue latency; this is not an isolated wall-clock performance comparison.

Start the independent shared service on the prepared server from the repo root:

```bash
../model_/.libero-max-venv/bin/python -m hybrid_rollout.libero.shared_astra_service
source runtime/shared-astra/client.env
# Pass explicitly to each experiment controller:
# --planner-url "$JEV_PLANNER_URL" --planner-timeout "$JEV_PLANNER_TIMEOUT"
```

The default endpoint is `http://127.0.0.1:8103/replan` on that server; `/health`
checks the HTTP adapter. The detached service survives SSH disconnects and
experiment completion. It is not a boot service: rerun the command after a host
restart. It retains the authenticated account lease and serializes Astra calls.
All clients must store their image evidence under this repository's `runtime/`.
Service metadata, client environment, logs and model-call audits live in
`runtime/shared-astra/`, independently of any episode directory. Do not add this
service or port to an experiment supervisor's services or reserved ports, and
do not terminate it from experiment cleanup. Controllers must receive the URL
explicitly; sourcing the environment alone does not change existing launchers.
The loopback URL is shared by processes on this host; remote clients require
SSH forwarding and evidence files available on the server.

Independent Pi processes showed small numerical action differences despite
identical inputs and seeded flow noise. For strict action pairing, add
`--matched-policy-trace /absolute/baseline/queries.jsonl` with the fork reference
and planner audit. Until the first teacher request, each query requires exact
instruction, simulator-state hash and image-hash matches, then uses the recorded
baseline Pi chunk. A mismatch stops the trial. The first recovery review releases
this replay before paused-scene refresh, after which all new queries (including
the fresh candidate submitted to Astra) use the live Pi model. This is a paired
intervention experiment, not a second independent policy sample. Neither the
trace nor future baseline outcomes are included in Astra's request.

The strict post-change trial on 2026-10-11 is archived under
`runtime/max-postchange-fork-isolated-20261011/paired-audit`. Both arms reached
520 controls and failed the original native goal. State, inputs and executed
controls matched exactly through control 311, including the camera change at
146. Astra was called once at 311 via the existing stage1-36 HTTP service.
The trigger was the unverified-progress budget (7/17 elevated reports, not a
confirmed stuck violation). Astra returned three subgoals, but only recovery-1
was executed: 42 chunks / 209 controls, with no subgoal completion event.
Thus this trial establishes post-change intervention, not successful recovery
or improved task completion. Two infrastructure/pairing trials were retained
and excluded; neither is counted as a formal task result. The shared service's
manifest and launch configuration stayed unchanged, and its results contained
no external episode. Inspect `comparison.json`, `interventions.jsonl`, and
`recovery-diagnosis.json` in the audit directory for the evidence.

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
The bundled `planner_server` can provide this HTTP endpoint using an explicitly
authenticated Codex runtime (see the smoke launcher below). There is no
fake/default Planner, credential, or automatic model call in environment setup.

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

### Prepared-server full smoke

With the prepared sibling `../model_` directory, the server's
`../../astra-runtime` authenticated runtime, and `../../envs/jev-valen`:

```sh
bash start_libero_smoke.sh --show-config
bash start_libero_smoke.sh --check
bash start_libero_smoke.sh
```

The shell entry point lives in the repository root and works from any current
directory when invoked by its absolute path. No `conda activate` or prior `source`
is needed: it selects each service's existing Python executable directly.

| Service | Prepared environment | Model/assets |
| --- | --- | --- |
| MuJoCo/controller | `../model_/.libero-sim-venv/bin/python` (3.10) | `../model_/sources/LIBERO` |
| Pi/OpenPI | `../model_/.openpi311-venv/bin/python` (3.11) | `../model_/models/pi05_libero` |
| Valen | `../../envs/jev-valen/bin/python` (3.12) | `../model_/models/Valen-2B` |
| Astra HTTP adapter | controller Python, wrapped by `../../astra-runtime/run-with-astra` | authenticated `astra-runtime/codex`, `gpt-6-astra / xhigh` |

The Pi service loads the existing complete local checkpoint, normalization assets,
and cached tokenizer; this script does not download weights. Astra needs no URL
extraction script and no credentials embedded in application code. The existing
runtime supplies its login, proxy and account lock. The controller talks to the
local planner at `http://127.0.0.1:8003/replan`, which invokes Codex.

For different environment locations, set absolute paths in `JEV_MODEL_ROOT`,
`JEV_ASTRA_RUNTIME`, `JEV_SIM_PYTHON`, `JEV_PI_PYTHON`, `JEV_VALEN_PYTHON`.
For example, `JEV_PI_PYTHON=/path/to/conda/envs/openpi/bin/python bash
start_libero_smoke.sh --check` selects an independently prepared compatible conda
environment. Shell options are `--check`, `--show-config`, `--max-decisions`,
`--output`, `--uncertainty-policy`, `--max-uncertainty-reviews`, `--chunk-steps` and `--help`; the Python entry point below
has additional path options.

This starts Valen-2B, the official JAX `pi05_libero` policy, the loopback Astra
HTTP planner, and an EGL controller. The task smoke now defaults to
`--max-decisions 0`: run toward native task success up to the suite horizon,
subject to explicit monitoring/recovery stops. Each run gets a new
`runtime/libero-smoke-*` directory. The supervisor stops only its own services
when the controller exits. Ports 8001, 8003 and 18830 must be free. `--check`
checks paths and ports; only the actual smoke loads models and runs inference.
Use `--model-root`, `--astra-runtime`, `--valen-python`, `--pi-python`, and
`--controller-python` to override the server layout. Pi defaults to the separate
`.openpi311-venv`; Valen and the Python 3.10 simulator keep separate environments.
`OPENPI_DATA_HOME` points to `model_/.cache/openpi`, including the PaliGemma tokenizer.

The planner uses the runtime's normal authenticated `codex exec`, fixed to
`gpt-6-astra` / `xhigh`, with actual camera attachments, read-only permissions,
and structured output. It implements LIBERO single-arm normalized actions;
it does not reuse RoboDojo's dual-arm decision tools. Decisions are checked by
the same LIBERO validator before reaching the scheduler. The runtime owns the
account lock; do not wrap the entire smoke launcher in `run-with-astra` as well.

`controller/smoke_report.json` requires native instruction success for `passed`.
Budget or horizon exhaustion with `success=false` is never task-smoke success.
The separate `component_check` field records the old connectivity-check criterion.
The report also records a separate Astra HTTP probe using a recorded real
observation and its matching Pi proposal. This probe is explicitly labelled
`SMOKE_PROBE`: it asserts no confirmed failure and never executes its decision.
`recovery_calls` records actual scheduler recovery calls separately. A smoke
cannot establish recovery execution if Valen never confirms a failure.
An unavailable model or an early uncertainty pause remains an incomplete smoke;
the launcher does not lower monitor thresholds to make a run pass.

The default `continue` policy preserves temporal history and the pending Pi
proposal when valid evidence is uncertain. It records `uncertain_continuation`
events and `RUNNING_UNCERTAIN`, then monitors the next chunk. It does not consume
Astra uncertainty-review budget. Confirmed violations still trigger Astra;
native task success or the native horizon terminates the episode. This change
addresses premature stopping, not Valen calibration or proven recovery efficacy.
The shared scheduler and other environment entry points retain their `pause` default.

Validated on the same task/state with real Pi and Valen in
`libero-smoke-20261010-223701-30eea8`: the continue policy completed the original
instruction at control 101 (`success=true`, `complete=true`). Twenty valid
uncertain reports permitted another bounded chunk; uncertainty reviews and
confirmed-failure recovery calls were both zero. This establishes task completion
without uncertainty deadlock on this scene, not superior success rate or verified
recovery efficacy. The previous pure-Pi run completed at control 104; rendering
and policy trajectories are not identical, so the three-control difference is
not evidence of an improvement. Original task completion remains the obligation.

The legacy `--uncertainty-policy pause` mode uses `--max-uncertainty-reviews 10`.
After the ordinary
same-scene supplemental observation remains uncertain, the controller holds
physics, discards speculative inference, and requests a fresh proposal and an
Astra review of the real camera evidence. This request is labelled
`review_kind=uncertainty`, retains the actual Valen report, and asserts no confirmed
failure. Only `student` for exactly one control with an aligned intent assessment,
or `stop`, is accepted; edits, high-level replans, stale responses and unavailable
reviewers cannot resume simulation. Every approved control is monitored again.
Unverified corrections from the confirmed-failure path still stop; this option
does not bypass their verification. Exhausting the separate review budget stops.

This is a deliberate scheduling-policy extension, not a calibration fix or an
equivalent result for the original confirmed-failure-only method. The normal
`hybrid_rollout.libero.run` default remains zero reviews. Use
`bash start_libero_smoke.sh --uncertainty-policy pause --max-uncertainty-reviews 0 --chunk-steps 1` for the
original scheduling policy (with the added initial visual context).
`run.json` records the policy and budget; `jev_uncertainty_*.json` records real
requests, and `uncertainty_reviews` is separate from `recovery_calls` in the smoke
report. Initial pre-action views supply temporal context but cast no confirmation
votes. Valen's thresholds and original question definitions remain unchanged.

The smoke defaults to `--chunk-steps 5`, an ordinary observation interval of five
simulator controls (the contract allows 1..5). Uncertainty-authorized continuations remain
exactly one control, even when the ordinary interval is five. Ten decisions are
not necessarily ten simulator controls; use `executed_steps` from the report.
Do not interpret an uncertainty review as calibrated Valen confidence or a
confirmed-failure recovery. A reviewer may legitimately stop either experiment.

The revised-policy run `libero-smoke-20261010-203200-f00b6b` passed the former
bounded component-check criterion on 2026-10-10, NOT task smoke: ten decisions
executed fourteen simulator controls, including
nine actual Astra uncertainty reviews authorizing one student control each. Each
control endpoint was monitored again; the run ended at its decision budget. The
separate Astra protocol probe also passed and the launcher exited 0. Native task
success was false and confirmed-failure recovery calls were zero. This verifies
the uncertainty-review loop, not a task-success score or an executed corrective
recovery. The one-control-window comparison run stopped after Astra declined
continuation; prompt rewrites and adding pre-action images alone did not reliably
resolve Valen's uncertainty. No monitor thresholds or questions were changed.

Original-policy server smoke on 2026-10-10 (`libero-smoke-20261010-200929-166f44`): real
EGL simulation, JAX Pi inference and Valen HTTP monitoring ran successfully.
After one student control, Valen remained uncertain even after supplemental
observation, so the controller ended `PAUSED_UNCERTAIN` (exit 2). The separate
`gpt-6-astra` / `xhigh` two-camera HTTP probe returned a validated two-subgoal
replan. Its decision was not executed; actual recovery calls were zero and
native task success was false. The overall smoke is **incomplete**, not passed.
LIBERO monitor calibration/temporal evidence remains to be evaluated before
claiming a complete recovery loop. See that run's `controller/smoke_report.json`.
Weight loading and the first JAX inference can take several minutes per launch.

### Matched Pi / Jev comparison

From the prepared server repository root:

```sh
../model_/.libero-sim-venv/bin/python -m hybrid_rollout.libero.compare
```

This runs fresh Pi servers for three arms on the same `libero_spatial` task 0,
initial state 0, simulator seed 7 and Pi initial JAX seed 0: pure Pi, original
confirmed-failure-only Jev, and legacy Jev with a ten-call uncertainty-review
budget. This historical comparison explicitly retains `pause`; use the smoke
launcher for the new `continue` policy. All arms use five-control ordinary chunks and no decision truncation;
the native horizon is 220 controls. The current Jev's uncertainty continuations
remain one control. Task success comes only from the native result. A Jev policy
stop is a failed task attempt, never converted into a successful smoke.

The comparison stores a protocol, per-arm configs/artifacts and incremental
`comparison.json` in `runtime/libero-comparison-*`. Report startup-inclusive wall
time separately from controller/inference timings. Different speculative and
discarded inference counts consume the Pi RNG differently after initialization.
Identical seeds also do not establish identical rendered inputs: the first real
comparison had equal initial proprioception but small camera-image and initial
action differences. Compare the saved initial proposals and images when auditing
comparability; this is a system-level comparison, not identical-noise replay.
One scene is diagnostic evidence, not a general success-rate benchmark.
`hybrid_rollout.libero.shadow` can replay the pure-Pi recorded frames through
Valen offline without affecting execution; successful task completion alone
does not label every intermediate state failure-free or establish FPR/TPR.

Observed on 2026-10-10 (`libero-comparison-20261010-215403-6df640`):

| Arm | Native task success | Controls | Wall time including startup | Outcome |
| --- | --- | ---: | ---: | --- |
| Pure Pi | true | 104 | 339.4 s | Completed the instruction |
| Original Jev | false | 5 | 150.3 s | Uncertainty unresolved after supplemental observation |
| Current Jev | false | 15 | 915.2 s | Ten uncertainty reviews exhausted |

Current Jev produced 22 uncertain monitor reports, zero confirmed-failure reports,
and zero confirmed-failure recovery calls. Its ten Astra reviews permitted one
student control each. This scene shows no task-completion benefit from Jev;
uncertainty handling blocked an instruction that pure Pi completed. This is not
a general benchmark or evidence that every intermediate alarm is false.
The three arms have identical task/initial-state/checkpoint provenance and
initial proprioception. Initial images differ (mean absolute channel difference
below 0.16/255), and initial action maximum absolute difference is below 0.011.
These differences and subsequent RNG consumption limit causal attribution.
Offline Valen replay of the pure-Pi trajectory produced `UNCERTAIN` for all 21
windows (first at control 5), with zero confirmed failures. Thus the current
monitoring program did not establish clear continuation anywhere on this
successful trajectory. This replay is not a closed-loop recovery experiment.

### HTTP contract

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
- `eef`: replace a short Pi action prefix with a single-arm geometric target, only
  when `native_request.eef_control.supported` is true. `target` contains `position`
  (three world coordinates in meters), `quaternion_xyzw` (unit quaternion),
  `frame: "MuJoCo world"`, and `gripper: "keep" | "open" | "close"`. Relative to
  the fresh measured EEF, translation is bounded by 5 cm and rotation by 0.35 rad.
  Like edit, this requires an assessed visible failure or misaligned intent.
  Example target: `{"position": [0.13, 0.2, 0.3], "quaternion_xyzw": [0, 0, 0, 1],
  "frame": "MuJoCo world", "gripper": "close"}` (valid only near that pose).
- `replan`: install 1–8 declarative subgoals with `subgoal_id`, `instruction` and
  observable `success_condition`. The instructions really reach Pi; the original
  task remains the final obligation. No action is executed by plan installation.
- `stop`: end incomplete without resetting.

For student/edit/eef/stop provide `request_id`, `mode`, `steps`, `reason`, `assessment`
(matching `assessment_example`, but with the Planner's actual evidence). `edit`
also includes the edit object. For replan provide only `request_id`, `mode`,
`reason`, `subgoals`. EEF additionally requires `target`; joint-space and dual-arm
targets remain unsupported. Each response chooses one mode. A later review can
choose EEF under an already-installed replan, using a newly generated Pi proposal.

EEF conversion reads the live fixed-impedance, delta OSC_POSE controller's actual
scaling (not a hard-coded meter/radian multiplier). Unsupported configurations
fail closed for EEF while leaving student/edit/replan available. This adapter
matches robosuite 1.4 `scale_action` and `set_goal_orientation`: world translation,
`R_goal = Exp(delta_world) @ R_measured`. No IK layer is needed for an OSC controller.
The host reads the controlled site before **each** of at most five controls,
converts the remaining pose error into bounded normalized input, and records the
target, before/after measured poses and sent action in `eef_NNN.json` (including
completed controls when a later step fails). `keep` preserves the selected Pi
prefix's gripper command for each row. Target bounds are command limits, not a
guarantee of tracking, collision avoidance, or successful grasping.

The request also includes `student_eef_command_preview`: nominal integration of
the clipped Pi controls using the OSC configuration. It is explicitly **not FK,
achieved robot motion, or an object prediction**. No simulation is advanced to
construct it, and Valen's post-action input remains unchanged.

In continuous mode, local edit/EEF execution waits for its new Valen report before
any Pi handback. RUNNING allows fresh Pi inference from the corrected state;
FAILED enters the usual recovery gate; SUSPECTED/UNCERTAIN pauses as
`PAUSED_RECOVERY_UNVERIFIED` and requests Astra with
`local_correction_unverified`, the raw evidence board, fresh cameras and a new
Pi proposal. No controls execute during this review. Astra may authorize bounded
controls, change the plan or stop; authorized controls are observed again before
automatic handback. Persistent uncertainty consumes the finite teacher-call
budget and ends explicitly as STOPPED when exhausted, rather than reporting a
terminal episode as merely paused. A clear report does not establish target or task
success. Corrections do not persist as offsets over future chunks or train Pi.
Replan still changes Pi's instruction, and original-task success remains native.

Confirmed failure thresholds, optional uncertain-observation pauses, same-scene recovery,
one unreviewed chunk, and post-correction verification reuse the existing scheduler.
The monitor's confidence thresholds remain uncalibrated for LIBERO.

## Validation and artifacts

### Continuous status board: evidence versus Astra interpretation

`max_run --continuous` selects the versioned `task_progress` Valen question
profile. The ten extra `noul` answers remain **raw evidence**: advancement,
original-task satisfaction, visibility/holding/support/destination relations,
scene comparability, and approach/transport/release observations. Neither Valen
nor Jev writes the semantic robot/task/scene status of the board.

Jev maintains three explicitly attributed streams:

- `host_measurement` / `host_execution_log`: EEF and robot measurements,
  instructions, installed plans, executed controls and scheduler transitions.
- `valen`: timestamped raw answers with image/observation references and subgoal
  scope. Conflicting or uncertain scores remain verbatim; they are not rewritten
  as scene facts or natural-language conclusions.
- `astra_interpretations`: Astra's robot status, task status
  (`pending/in_progress/completed/uncertain`), task rationale, scene context,
  uncertainties and cited observation IDs. Each entry retains the reviewing
  request, source step/observation and plan revision. Later Valen rows do not
  overwrite it; only a later Astra review supplies a new interpretation.

The first board has no semantic status until Astra reviews it. Historical Astra
judgments are not current truth, and `completed` never sets native simulator
success. The board goes only to Astra, not back to Valen as pseudo-evidence.

Valen receives up to six chronological dual-camera endpoints (12 images). Five
questions use explicit `choice` candidates: comparability, current-instruction
progress, repeated ineffective attempts, grasp outcome, and recovery completion.
Each includes an insufficient-evidence candidate (and non-applicable/no-attempt
choices where relevant). The remaining questions retain their binary score heads.
All candidate probabilities are preserved as `raw_answers`; scalar projections
exist only for numerical compatibility. They are not calibrated scene facts.

Jev requests interpretation using these initial, uncalibrated numeric rules:

- `anomaly_scores_review`: two consecutive applicable high collision, target
  mismatch or grasp-failure scores within the bounded advisory sampling gap
  (`--monitor-max-lag`, default 40 controls). Coalesced observations may request
  review; they do not relax the separate strict constraint-confirmation gap.
- `low_progress_scores_review`: six time-sampled valid observations spanning 30 executed
  controls by default, at least four low-progress/attempt candidates. Visibility
  and comparability must be reliable; explicit unknown evidence cannot count.
  A control-span buffer supports short chunks and longer configured spans; the
  shortest eligible suffix is sampled evenly into six distinct observations.
  `--stall-steps` sets the span; `--stall-min-reports` can raise the four-vote floor.
- `ambiguous_observations_review`: three consecutive observations with target
  visibility or scene comparability at or below the low threshold (0.2).
  The first review requires 30 executed controls; subsequent visual reviews
  require 80 controls since the last Astra review, including across replans.
  Diffuse progress candidates and intermediate visual scores remain unknown:
  they neither vote for stagnation nor cause this rapid visual-review path.
  The existing explicitly labelled execution-budget fallback covers prolonged
  unknowns. Anomaly and completion candidates bypass the visual-review interval.
  This is a visibility/evidence review, not a declaration of task stagnation.
- `completion_scores_review`: two consecutive high completion candidates request
  Astra interpretation for the SAME condition (original task or recovery goal);
  the same bounded advisory sampling gap applies. Jev never merges different
  conditions or advances the task from these scores.

One high progress score cannot clear the window. Two consecutive reliable high
progress scores clear it and refresh the fallback timer. Reviews and new plans
reset trigger evidence, while preserving board history. Late pre-review reports
cannot vote again. Duplicate evidence and long gaps cannot create confirmations.

Board mode polls Valen without blocking ordinary Pi chunks. At most one monitor
evaluation is in flight; new endpoints coalesce into the latest short history
window rather than queueing stale work. `monitor_submitted.skipped_endpoints`
and report lag make skipped endpoints explicit. Only at the configured maximum
observation lag does the host apply backpressure and wait for the worker, with
its existing timeout. Fallback review uses
executed controls since activation, last review or sustained progress, capped by
`--max-review-interval`. It is labelled as an execution budget, not Valen-detected
anomaly. Steady progress does not cause periodic Astra calls. Without Astra,
non-shadow mode stops at a required review; only explicit shadow mode logs and
continues. Exhausted Astra
budget stops execution when another review is required.

In continuous `task_progress` mode, `recovery_policy=single_instruction` replaces
future recovery queues. Replan accepts exactly **one** current recovery instruction
(the wire format retains a one-element `subgoals` array). Jev holds only that active
instruction; the original task stays separately recorded as the final obligation.
Pi receives the current instruction and fresh observations, without concatenating
a second task into its prompt. Old instructions and Astra assessments remain in
history, not an executable backlog.

The recovery instruction persists across multiple monitored chunks. Five controls
is an observation/control bound, not an instruction expiry. Only new evidence or
the execution-budget fallback requests another Astra decision. Astra cannot
return `student`, `edit` or `eef` in this mode; both planner and host validators
reject them. `keep_instruction` resumes fresh Pi inference without changing the
instruction or its start step. Old candidates and pre-review votes are not reused. New anomaly,
completion or visibility evidence can then request Astra again.

Astra may explicitly return `resume_original` when the current scene supports the
original instruction. This requires the current request ID, plan revision, active
subgoal ID, reason, and a reference to the fresh paused observation. It replaces
the active instruction, discards the old candidate and requests new Pi inference;
it does not assert task success. `advance_subgoal` is not offered in this mode.
Legacy constraint-only plan protocols remain compatible.

`--astra-call-budget` configures the total call cap (default **12**, positive
integer). Every replan, keep, stop and resume decision consumes
one call. Required review after exhaustion stops explicitly; it never executes an
instruction after a required review cannot be obtained. The selected cap is logged in run/selection metadata. More
frequent scene-grounded review increases Astra cost and latency; these defaults
have only CPU fixture validation, not real-model calibration.

Both planner implementations share visual evidence selection and guidance in
`visual_evidence.py`. Attachments begin with the freshly captured two-camera scene,
then prioritize the largest observed public-image change pair, the previous Astra
decision scene and the latest EEF before/after pair. These event roles survive
replans and trigger-window filtering; ordinary historical samples and episode
context follow. Roles retain observation ID, step, capture time and plan revision.
Duplicate paths are attached once while retaining all references in
`image_manifest.json`, in the same order as the CLI `--image` arguments.

Change candidates use the maximum camera-wise mean RGB difference of 32×32
thumbnails between executed chunk boundaries. This is a numeric sampling heuristic,
not perturbation detection: robot motion, camera motion or lighting can dominate;
changes inside a chunk can be missed. No hidden simulator perturbation labels or
object state are exposed. Only the strongest pair is pinned, keeping context bounded.
Jev also pins known execution events without interpreting their scene contents.

The shared planning prompt explicitly requires comparing current views, historical
changes and the status board, checking whether the current instruction remains
valid, citing observation IDs and stating contradictions or uncertainty. Previous
Astra judgments and Valen scores are evidence to recheck, not current facts. Only
actually attached frames may support visual claims; a path in board history alone
does not imply the model saw those pixels. The guidance remains part of the CLI
planning prompt; this does not introduce a separate system-role API message.

Both planner implementations require a structured `board_assessment` alongside
`decision` when `board_review_required=true`, for all modes including replan.
The service and host validate its shape and supplied observation references
before accepting the decision. Reference validation is not semantic verification
of the model's claim. Requests include the board, prior attempts, fresh paused
cameras and a new Pi candidate; old images keep their source-step labels.

Replans retain evidence and Astra interpretations across generations. A review
clears the short sampling window, not the episode record. `task_context.json`
holds the latest board; `task_context_updated` events and replan responses retain
raw evidence and interpretations on disk. Request memory is bounded to 12 recent
observations, 32 plan/sampling records, and 16 Astra interpretations.

This is a relational evidence board, not object tracking, a generated scene
narrative, or a world-state oracle. Questions and thresholds have not been
calibrated on live Valen. This candidate/trigger revision has only CPU fixture validation; no GPU experiments
or shared services were run or changed for this revision. Earlier smoke results
used the prior five-control instruction-expiry policy.

```sh
python -m pytest hybrid_rollout/libero/test_integration.py \
  hybrid_rollout/libero/test_eef.py hybrid_rollout/libero/test_continuous.py \
  hybrid_rollout/libero/test_planner.py \
  hybrid_rollout/libero/test_task_context.py hybrid_rollout/libero/test_recovery_protocol.py \
  hybrid_rollout/libero/test_visual_evidence.py \
  hybrid_rollout/robodojo/test_jev.py hybrid_rollout/robodojo/test_jev_launch.py -q
```

These CPU fixtures test protocol mapping and scheduler invariants, not real
LIBERO rendering, GPU inference, or task success. Use the prepared-server smoke
above for real models, and inspect its report before interpreting the outcome.
On DolphinFS, proposal archival falls back from unsupported hard links to an
exclusive byte-for-byte copy, preserving existing artifacts.

Inspect `selection.json`, `run.json`, `observations/`, `proposal_*.npz`,
`execution_*.npz`, `jev_events.jsonl`, `jev_summary.json`, and `result.json`.
`result.success` comes only from the native task check. Native horizon exhaustion
can produce a completed episode with success false; CLI exit 0 requires success.
Interrupted or uncertain execution is incomplete. Output directories must be new.

Trigger-window observation IDs remain citable even after recent-history eviction.
Holding an object while it rests on a support surface does not invalidate visual
comparability; only Astra interprets physical relations.

### Episode-scoped Astra conversation

Jev/Valen model requests use `jev_scene_evidence_v1`: chronological camera images,
original task, current instruction, success condition, and observation/time/step
identities only. A shared allowlist at `ValenMonitor.evaluate` covers both monitor
profiles, asynchronous monitoring, paused inspection and completion verification.
Pi action arrays, EEF poses, robot state vectors, action source and native outcomes
are excluded, including extra image metadata. Host audit records retain these
fields for offline diagnosis. The input contract participates in `program_id`,
so evidence from the earlier numeric-input protocol belongs to another generation.
The 2026-10-12 live comparison below validates this interface, but visual
completion false positives remain unresolved.

Current continuous task-progress requests use `planner_role=instruction_eef`
and `input_format=episode_instruction_eef_v2`. The fixed developer instructions come
from [the Astra instruction planner skill](skills/astra-instruction-planner/SKILL.md).
Every turn begins with `review_brief`: the trigger-specific question, original
task, current instruction/success condition, execution start and elapsed controls,
and required decision. Completion asks whether the subgoal was actually reached;
low progress or budget asks whether to continue or replace it; ambiguity asks
which facts remain unobservable; anomaly asks whether the images support it.
A replacement uses LIBERO benchmark task language: action, object, and destination
or relation, with the observable outcome in a separate `success_condition`.
Natural tasks such as pick-up-and-place can contain multiple verbs. Pi instructions
must not prescribe finger alignment, gripper controls, grasp contact points or
EEF motions. For an already lit stove, `put the moka pot on the stove` is a
benchmark-style remaining task; success still requires visual/native verification.
The skill examples were checked against `:language` in local LIBERO BDDL files:
`libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl`,
`libero_90/KITCHEN_SCENE3_put_the_moka_pot_on_the_stove.bddl`,
`libero_10/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy.bddl`,
and `libero_90/KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet.bddl`.

The default prompt contains current/historical images, compact Valen evidence,
instruction history, execution receipts and a diagnostic descriptor, without Pi
action arrays, EEF poses or state vectors. Board/history reads omit those fields too.
At a review pause, the host captures fresh images and infers one fresh Pi candidate
for diagnosis; physics stays paused throughout. The original `proposal_*.npz`
remains an audit artifact. The planner archives an immutable JSON control record
under its episode directory with review/observation/prediction IDs and a SHA-256.
`read_control_diagnostics` reads bounded slices of that current prediction or its
measured EEF/controller context; it rejects other reviews and modified records.
`read_observation_actions` separately reads historical executed controls. Neither
tool writes files or edits actions; `read_episode_record` supplies task evidence.

The bounded `eef` decision requires the current review/observation IDs, a visual
reason, an explicit failed-execution or misaligned-intent diagnosis, 1..5 controls
(within both chunk bounds and remaining episode controls), and a single world-frame target. The host
rejects displacement over 5 cm, rotation over 0.35 rad, non-unit quaternions,
nonfinite values and unsupported controllers. Gripper state must explicitly be
`open` or `close`, so correction never inherits an unseen Pi gripper command.
The existing EEF executor recomputes OSC inputs from measured pose each control
using the actual controlled grip-site position and orientation. Robosuite's
`robot0_eef_quat` describes the hand body and can differ from that site by 90
degrees; it remains in the original Pi state representation, not the EEF target
reference. This distinction has a regression test and a real simulator check.
The executor checks actual displacement/rotation before and after each step, including
the final step. An observed envelope breach stops execution and retains the trace;
the controller cannot guarantee that the physical plant never overshoots.
Afterward, the scheduler waits for visual evidence from the new endpoint before
fresh Pi inference; it does not treat that report as proof of task success.
`keep_instruction`, `replan` and `resume_original` also discard the diagnostic
prediction and infer afresh. `student`, `edit` and arbitrary action-array output
remain unavailable in this protocol. Jev retains its scene-only model inputs.

The original `episode_instruction_v1` remains instruction-only, without action
tools or pre-review Pi inference. The `episode_thread_delta_v2` action protocol
described below remains available for legacy constraint-only baselines. Existing
conversations cannot change formats or acquire new tools on resume; a mismatch fails before
submitting a turn. Offline regression checks do not restart shared services or
launch experiments. The explicitly authorized live deployment is recorded below.

Both planner entry points now use the same `EpisodeThreads` implementation and
Codex app-server transport. `(run_id, episode_id)` identifies a persistent Astra
thread; replan and EEF transitions do not start another conversation. Fixed
instructions are installed once, and per-turn output schemas still bind the
fresh proposal identity and permitted recovery operations. The model remains
`gpt-6-astra` with `xhigh` effort.

Each turn supplies current dual-camera pixels, previously unsent selected
keyframes, new Valen evidence, a diagnostic descriptor (the fresh candidate
itself only in the legacy action protocol), and the last actual host
execution receipt in `previous_result` (not the previous Astra attempt).
`gate_instruction` is excluded; static action-contract fields
are supplied once. Full board snapshots and append-only request/response history
remain under `planner/episodes/<identity-hash>/`. `read_episode_record` provides
bounded read-only slices of this episode's board/history; it cannot execute
controls or read another episode. This is structural projection, not host-written
semantic summarization. Thread history can still consume tokens or be compacted;
smaller appended input is not proof of lower latency.

`thread.json` persists the thread ID, delivered evidence/image keys and validated
response cache. After a service restart it resumes the same thread. A repeated
identical request returns its validated cached response without another model
turn. A timeout, invalid response or crash during a pending turn leaves an
unresolved journal and fails closed: it never silently starts a fresh thread or
replays a decision. Operator reconciliation is required for that episode. Calls
remain serialized and each episode is isolated. Shutdown closes only the owned
app-server process; durable conversation files remain available for resume.

Per-call audit directories retain the original full `request.json`, actual
`model_input.json`, attached `image_manifest.json`, thread/turn IDs, RPC events,
validated response and measured turn duration. Health reports
`conversation_mode=episode_instruction_eef_v2` with all three supported formats listed;
each invocation records its actual format. Health alone does not demonstrate a
successful model turn. Regression tests use a fake RPC transport and never call
a GPU/model service.

The v2 prompt factors evidence identities and instruction scopes into context
tables. Each observation refers to a schema defining the question/candidate
order. All original probability values, `choice` and `confidence` remain at full
precision. Scores are removed only when exactly recoverable from the declared
projection; absent, unfamiliar or contradictory answers/scores are retained as
raw answers or score overrides. No host-written semantic summary replaces the
evidence. The schema is included once per turn so a compacted conversation can
still interpret new rows. All new observations remain present, including those
outside the board's 12-row recent window.

Historical executed-action arrays are archived under the owning episode's
`actions/`, keyed by a hash of the observation ID. `read_observation_actions`
accepts `observation_id`, a zero-based control `offset`, and `limit` (1–20
controls); it returns the exact recorded arrays and pagination metadata. This
index survives board eviction and service restart. Unknown IDs, another episode,
symlinks and conflicting arrays for the same ID are rejected. Full original
requests and append-only history still retain every action array. Only the legacy
action protocol includes current Pi candidate arrays and command previews in the
default prompt. Executed commands never establish achieved motion or task success.

`previous_result` contains the latest execution's start/end ticks, count and
control source (and the execution decision index when available). It references
the current observation's action record and does not repeat prior Astra prose,
old trigger frames or intended results. For older requests lacking the native
receipt, only explicit current-scene execution metadata may supply a receipt;
otherwise it is null. Current state/EEF fields identical in `native_request` and
`current_scene` use an explicit scene reference.

The compact image manifest uses zero-based attachment indices, observation IDs,
steps, cameras and role labels. Alias observations remain explicit. Full paths
and original references remain in the separate audit manifest and actual RPC
image attachments; attachment selection/order and source IDs are unchanged.

This source change requires deliberate deployment after active v1 episodes
finish. Do not restart or reload a running shared service as part of offline
validation. An existing thread with a different/missing `input_format` fails
before a new turn instead of silently replacing its conversation or using a
tool that was never registered. New v2 episodes resume their original thread
and tool registrations normally. Deployment must retain the original v1 service
for unfinished v1 episodes.

CPU regression checks for this format:

```sh
python -m pytest hybrid_rollout/libero/test_compact_input.py \
  hybrid_rollout/libero/test_episode_thread.py \
  hybrid_rollout/libero/test_visual_evidence.py -q
```

### Authorized live comparison, 2026-10-12

Run record: `runtime/max-instruction-eef-20261012-133056/report.txt`, with exact
configuration, source hashes, prompts/tool responses, images, EEF traces and
`comparison-final.json`. Shared Astra on port 8103 was explicitly upgraded to
`episode_instruction_eef_v2` and remains running. The launcher now supports
`--planner-url` without taking ownership of the shared service and validates its
protocol before starting model services. Final offline regression: 354 passed.

Case `libero_10-t0009-camera_shift-d0-p195`, horizon 520, Pi seed 195,
progress/subgoal budget 80, Astra budget 12, monitor lag limit 40, maximum review
interval 160, stall window 30 with four reports, and planner timeout 600 seconds.
Both arms start from the same state and identical dual-camera images. The camera
perturbation is proximity-triggered (off step 148, on step 153), not synchronized
by control step. Each arm performs its own Pi inference.

The initial Astra-on attempt stopped before executing its first EEF correction:
the live pose check correctly rejected the hand-body/grip-site orientation
mismatch. Its artifacts remain under `comparison/astra_on` and are excluded.
After correcting only the EEF observation reference, a real single-control
simulator check passed with Pi states unchanged. The final Astra-on arm was rerun
from the initial state under `retry-site-fix/astra_on`; the unaffected baseline
under `comparison/astra_off` was retained.

The baseline exhausted 520 controls with native success false. Astra-on stopped
at 375 controls with native success false and an incomplete episode because all
12 review calls were spent. This is not an equal-horizon efficacy comparison.
Pi actually executed `put the moka pot on the stove` for 200 controls and
`pick up the moka pot and place it on the stove` for 85 controls. Across five
reviews, 19 read-only tool calls succeeded; no action/state arrays appeared in
the default model inputs. Both bounded EEF corrections executed five controls:
the 3 cm upward target achieved 2.50 cm vertically, and the 4 cm horizontal target
achieved 2.78 cm horizontally. Neither recovered a successful grasp; the pot
remained on the table. Endpoint monitoring preceded fresh Pi execution.

Monitoring continued during execution: baseline 103 reports, Astra-on 70 ordinary
or drained reports plus two EEF endpoint reviews, with maximum received lag five
controls. All 12 Astra pauses contained zero control execution. Eleven reviews
were triggered by false completion signals that Astra rejected; these consumed
the budget. The interfaces worked after the coordinate fix, but this single run
does not establish improved task success or that task-level wording alone fixes
Pi's grasping behavior.
