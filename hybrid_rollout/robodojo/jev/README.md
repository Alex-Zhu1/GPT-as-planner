# Astra subgoal planning, Pi execution and Valen monitoring

This is an **opt-in experimental RoboDojo entry point**, independent of the
unchanged `skill.run` full-time Astra reviewer and GPT-only baseline. It uses
Valen (the multimodal Jev-inspired model), not the text-only TypeSafe Jev API.
No dependency on the sibling `Jev` checkout is needed.

## Subgoal policy (default)

`jev.run` defaults to `--monitor-policy subgoals`. Astra produces **one to
three language stages**, Pi executes the active instruction, and Valen detects
anomalies. **Only Astra assesses completion and authorizes stage transitions.**
The previous background recovery experiment requires `--monitor-policy temporal`.

```text
initial observation -> Pi original-task inference (speculative, no execution)
                    -> Astra initial language plan (in parallel)
validated plan -> Pi active-stage instruction -> bounded execution
                    ^                              |
                    |                       Valen anomaly evidence
                    +--- no review due ------------+
                    +--- anomaly or periodic review -> paused Astra assessment
                                                      | keep current stage
                                                      | advance completed stage
                                                      | plan missing/failed work
native termination -> record success/failure separately from visual completion
```

The initial Pi proposal is reused only if its instruction exactly matches the
first stage and the observation has not changed. Otherwise it is discarded.
Every stage change or review discards unexecuted old proposals. There is one
simulator owner, one Valen request at a time, and at most one unreviewed executed
chunk. Pi inference overlaps the preceding Valen report; execution waits for
that report. Astra reviews use a fresh paused capture without advancing physics.
They do not reset the episode or produce numerical robot actions.

The host derives requirements from the original instruction and public task
context. The installed `organize_table` scene has four movable items and a static
drawer: mouse on pad, keyboard aligned face up in its frame, figurine upright on
its stand, alarm clock upright on the cabinet, then both arms return. Its source
instruction's drawer-opening/miscellaneous-storage clause does not apply to this
scene and is explicitly explained in `task_context`; the original stays in the
audit. These public semantics contain no per-episode success or object truth.
Other known task templates preserve their required order; unknown instructions
remain whole. Placement requirements in this scene may be reordered.

The shared [subgoal_planner.md](subgoal_planner.md) uses a **version-4** schema in
both API and compact transports. Astra returns `plan` (1–3 replacement goals),
`keep_plan`, or `advance`. Every review assesses **all** requirements as satisfied,
pending, or uncertain, with visual evidence. A replacement plan covers exactly
the unfinished, uncertain, or reopened requirements, each once. Coverage, checks,
ordering constraints and request identity are validated. Astra can write concise
RoboDojo-style repair instructions instead of copying original clauses verbatim.
It can recognize work done outside the current stage and reopen damaged results;
Valen cannot mark requirements complete or reopen them. Arm return remains last.
External `--planner-url` services must implement this version-4 envelope.
The older full-agent skill is not loaded; the model remains `gpt-6-astra/high`.
No performance gain is assumed without evaluation.

Pi instructions use short action/target/destination descriptions, for example
`Pick up the fallen figurine and put it upright on the stand.` The host validates
at most **24 words, 160 characters and one sentence per covered requirement**
(600 characters total for grouped stages), and rejects common verification or
preservation appendices such as `ensure ...` or `without disturbing ...`.
These bounds are a language-interface policy, not an experimentally optimal
token limit. Stability, support, release and preservation requirements remain
in the exact completion checks and Astra's visual assessments. Necessary release
actions are still allowed. Invalid text is rejected before activation, never
silently truncated; a rejected planner response stops the scheduler as before.

Every review includes the last six activated instruction versions, their covered
requirements, actual executed control counts and the reasons for changing them.
When fresh execution evidence shows repeated failed grasps or ineffective
attempts, Astra should return `plan` with a different short description of the
same unfinished goal, preserving completion checks and future work. A periodic
review may do this even when Valen found no anomaly. Normal approach, occlusion
or elapsed time alone do not justify a rewrite. The existing review cooldown and
planner-call budget still apply; no automatic synonym cycling is introduced.
`subgoal_instruction_rewritten` records old/new wording and Astra's reason;
`instruction_history` in the summary retains execution counts across `keep_plan`.

Valen receives plan context, task requirements, recent three-camera endpoints
and labeled reference images. Its only questions are:

| Question | Candidates |
| --- | --- |
| Observability | usable, occluded, ambiguous |
| Each independent anomaly | present, clear, insufficient_evidence |

Anomalies are wrong object, bad placement/orientation/support, wrong order,
collision, grasp failure, regression of an achieved result, and stalled attempts.
Questions distinguish an unfinished placement from an attempted but bad result.
There are no completion votes or phase/progress classifiers. Neither model
receives predicted Pi actions, rewards or simulator object truth.

Scheduling defaults (experimental thresholds, not calibrated probabilities):

- 25 controls per chunk at 0.04 s/control, 6 recent Valen endpoints, threshold 0.8.
- 3 simulated seconds of the same visible anomaly request Astra review;
  collision probability >=0.9 requests review immediately on a new detection.
- 6 simulated seconds of confidently occluded/ambiguous observations request
  review. Diffuse low-confidence answers alone do not repeatedly trigger Astra.
- Reviews have a 6-second execution cooldown (`--subgoal-review-cooldown-seconds`).
  The first anomaly review and a new acute collision bypass it; persistent acute
  detections do not bypass it repeatedly.
- Every 12 simulated seconds since the last review, Astra checks progress even
  without an anomaly (`--subgoal-review-seconds`). This allows completion and
  stage switching without making Valen a completion judge. A 40-second episode
  with no anomalies uses four Astra calls including the initial plan.
- A final visual confirmation retains the last instruction for settling until
  native termination. Later visual assessments can revoke it. Only native
  success counts as task success.
- `--max-replans 12` includes the initial call; zero removes this cap.
  Exhausted budgets, malformed plans or unavailable models stop incomplete.
  `--max-decisions 0` uses native termination. Older action-tuning flags require
  an explicit older monitor policy.

```sh
python -m hybrid_rollout.robodojo.jev.run \
  --output /absolute/new/subgoal-run --task organize_table \
  --checkpoint /absolute/path/to/pi05/checkpoint \
  --sim-port 19113 --student-port 18830 \
  --valen-url http://127.0.0.1:8001/predict \
  --planner-backend api --api-base-url http://127.0.0.1:8317/v1 \
  --api-key-file /absolute/path/to/local-api-key \
  --monitor-policy subgoals --chunk-steps 25 --max-replans 12
```

Alternatively use `--codex /absolute/path/to/codex`. Valen's `/predict` must
support `choice` questions; malformed or unsupported responses fail closed.
`jev_plan_*_<goal>.json`, `jev_events.jsonl`, `jev_summary.json` and the planner
request/input/schema/prompt/response artifacts record the actual activated
instructions and stage evidence. `result.json` remains the native outcome;
exit code 0 means a native episode finished, including native task failure.

```sh
python -m pytest hybrid_rollout/robodojo/test_subgoals.py \
  hybrid_rollout/robodojo/test_subgoal_backends.py -q
```

## Temporal policy (explicit comparison)

Select `--monitor-policy temporal` to run this prior protocol:

```text
pi05: predict 50 controls -> execute up to 25 -> fresh camera endpoint
        ^                                           |
        |                  Valen: recent 6 endpoints, about 5 simulated seconds
        |                                           |
        +---------- continue while risk / uncertainty accumulates
        |                                           |
        +---------- ordinary Astra review in background (bounded continuation)
                                                    |
                         continue, or request fresh paused recovery
                                                    |
                           fresh scene + proposal -> student/edit/EEF/replan
```

- `--chunk-steps 25` executes a prefix of 25 controls from each 50-step pi05
  prediction. At RoboDojo's 0.04-second control interval this is one simulated
  second. `--chunk-steps 15` is also supported (0.6 seconds). The unused suffix
  is discarded before replanning from a new observation. Correction actions
  remain limited to five controls. Monitoring occurs at chunk endpoints; an
  event within a chunk can only be noticed at its next endpoint.
- For 15-step chunks, `--history 10` gives about 5.4 simulated seconds of
  visual context; uncertainty and cooldown budgets remain specified in seconds.
- Valen sees the three cameras at the latest six endpoints, including initial
  context. With 25-step chunks a full window spans five simulated seconds.
  Rewards and hidden simulator state are excluded from its visual input.
  Pi inference overlaps Valen inference; simulator RPC remains on one thread.
- Each constraint integrates applicability-weighted evidence above baseline
  0.5, with exponential retention 0.85 per simulated second, floored at zero.
  The threshold is 0.6 probability-seconds and the minimum evidence duration
  is three simulated seconds. Exact exponential integration avoids changing
  the evidence scale when the chunk length changes. These are experimental
  heuristics, not calibrated failure probabilities.
- Six simulated seconds of continuous uncertainty **for the same constraint**
  also request review. Neither uncertainty nor a review request ends an episode.
  Fresh executed intervals contribute once; inspection images do not add time.
  Gaps reset accumulation. The simulator control interval is validated before
  execution. `--min-evidence-seconds` and `--uncertainty-seconds` set the windows.
- Ordinary Astra reviews run in the background. Pi can advance at most another
  six simulated seconds (`--max-async-seconds 6`, 150 controls), with Valen at
  each endpoint, before waiting at a boundary. Only one Astra call is active.
  A collision score of at least 0.9 pauses at the next endpoint immediately,
  bypassing the ordinary evidence window and cooldown.
- A background answer can only be `continue` or `request_recovery`: it cannot
  execute an action or declare task success. A recovery request first pauses,
  captures the **current** scene and obtains a fresh proposal, then asks Astra
  for a validated student/edit/EEF/replan decision. Historical numeric actions
  are never applied after the robot has moved. Wrong-generation answers are
  discarded; pending answers cannot actuate after native termination.
- An accepted continuation clears accumulated evidence while retaining visual
  history. Reviews have a four-simulated-second cooldown, except acute risks.
  Local recovery installs a new plan revision and fresh evidence runtime.
  Only native simulator termination determines completion and native success.
- The default compact planner uses **gpt-6-astra/high** with attached images and
  a single structured answer, without shell/file/notes tool loops. One local
  app-server process is reused; review threads are isolated. At most three
  timestamps from three cameras are attached. Full source evidence is retained
  in the audit; prompts use bounded state, recent review summaries and sampled
  FK for paused recovery. This reduces avoidable overhead, but model latency is
  measured, not guaranteed. `astra_*/timing.json` records each completed call.

Paused compact Astra reviews support selectable action input with `--codex`
and the compact backend (the default for temporal mode), or with the `api` backend:

| Flag | Predicted trajectory supplied to Astra |
| --- | --- |
| `--astra-action-chunk none` | No action chunk or predicted FK points; retain current scene and decision identity/budgets. Next intent must remain uncertain. |
| `--astra-action-chunk sampled` | Default: existing four FK samples at indices 0, 12, 25, 49 for a 50-step proposal. |
| `--astra-action-chunk recent --astra-action-steps 5` | First five consecutive upcoming steps of the latest proposal, capped by the host and student execution budgets. N supports 1–25. |

`recent` means the next N steps, not the tail of the 50-step prediction or
previously executed actions. These options change only Astra input; full proposals
remain in local audit files, pi execution budgets remain unchanged, and background
reviews never receive predicted action chunks in any mode. Each completed call's
`timing.json` records mode, configured N, supplied point count, text size and latency.
Custom options are rejected for agent/HTTP backends rather than silently ignored.

`--astra-output-format mode-specific` omits unused numeric payloads: student/stop
responses omit edit and target, edit omits target, and EEF omits edit. Native
assessment and geometry validation remain active. The default `full` format is
available for comparisons. Timing logs additionally record local preparation,
connection, thread creation, turn submission, first visible output, output-stream
duration and token usage. The interval before first output includes network,
queueing, image processing and reasoning; it is not a pure reasoning timer.

### Task-template language replanning (compact and API)

Both compact transports now restrict `mode=replan` to **exactly one reviewed
task-template goal**. Astra selects one `replan_candidates` entry and copies its
`subgoal_id`, `instruction`, and `success_condition` exactly. The structured
schema binds those fields together; local validation rejects multi-goal queues,
paraphrases, changed conditions, and stale proposal IDs before installing a plan.
Reasoning stays in `reason` (up to 600 characters), never in the Pi prompt.
The scheduler still retains the original task as the final obligation and
resumes it after the chosen subgoal is verified; a subgoal cannot declare native
task success. Agent/HTTP planner contracts are unchanged.

For example, `organize_table` permits `Place the mouse on the mouse pad.` or
`Push the keyboard into the frame.` It does not permit added colors, arm
assignments, conditional verification steps, or improvised motion procedures.
Language sorting preserves the episode's category names and destination mapping,
e.g. `Put car objects into the left basket.` Number arrangement retains the
original task-level instruction:
`Arrange the numbers from left to right to form the largest possible number, and place them on the pad.`
It does **not** allow `Move 2 ...`, digit-specific pad relocation, or replacing
`Move` with `Place` while retaining the same unsupported per-digit command.
This restricts the language interface; it does not establish that Pi can recover
from every misplaced object.

The reviewed catalogue covers the ten tasks in the local comparison panel:
organize_table, classify_objects_by_language, imitate_sorting_sequence,
arrange_largest_number, pack_objects_into_box, classify_objects, build_tower,
make_kong, fold_clothes, and put_bottles_into_dustbin. Matching uses the original
episode instruction, so variants sharing that template work too. Unknown
templates omit `replan` from the schema; native student/edit/EEF choices remain
available. Add reviewed templates in `task_instructions.py` rather than falling
back to free-form instructions.

This restriction is independent of `--astra-output-format`: keep
`--astra-output-format mode-specific` for concise control fields. Image selection
is unchanged: first/middle/latest available timestamps, three cameras each,
at most nine images (fewer when history is short). Existing action-chunk settings
are unchanged, including `--astra-action-chunk none` in the previous comparison.
No additional launch flag is needed for the new replanning restriction.

Training-format evidence was checked against the local RoboDojo source at
commit `ee67a1468510da7624a089164402359f2afc72c8`:

- `XPolicyLab/policy/Pi_05/openpi/src/openpi/training/config.py` config
  `pi05_base_aloha_full_sim_arx-x5_seed_0` uses `RoboDojo_sim_arx-x5_v30` and
  `prompt_from_task=True`; `training/data_loader.py` applies
  `PromptFromLeRobotTask(dataset_meta.tasks)`.
- `XPolicyLab/scripts/transform_lerobot_v30_format.py` selects an episode's
  `instruction`/`instructions` and writes that same text as `task` on each frame.
- `task/RoboDojo/tasks/<task>.py::gen_instruction` provides the original task
  templates. The catalogue retains those templates, or splits source clauses
  for table organization and language sorting (making the drawer referent
  explicit). Single clauses are a constrained adaptation, **not** a claim that
  standalone subgoal labels were present during training. Full training task
  metadata was not available locally for an exhaustive vocabulary check.

The existing intent/outcome gate and dual-arm geometry checks still validate
corrections. High-level subgoal instructions reach pi05, and visual completion
requires distinct clean endpoints. The original task remains a final obligation.
Invalid evidence, malformed decisions, explicit exhausted budgets and model
timeouts still stop with an incomplete result; they are not native failures.

`--monitor-policy accumulating` preserves the previous observation-count policy
(1-step chunks, 2-frame windows, 12 uncertain observations and blocking review).
`--monitor-policy legacy` preserves discrete confirmations and pause on unresolved
uncertainty. Both retain their previous agent planner unless `--planner-backend
compact` or `--planner-backend api` is selected. The full-time `skill.run`
reviewer is a separate entry point.

Automatic student decisions record uncertain intent and
`decision_author=jev_scheduler`; background advice does not manufacture a fresh
predictive action assessment. Runs use `evaluation_method=pi05_valen_replan` and
must be distinguished from published baselines.

## Run the temporal comparison

For a paired pure-pi05 control, start only the same simulator and pi05 services,
then run `python -m hybrid_rollout.robodojo.baseline` with `--output`, `--task`,
`--checkpoint`, `--sim-port`, `--student-port`, `--seed` and `--chunk-steps 25`.
Use the identical frozen case manifest/layout and checkpoint as the harness run.
This entry point uses the original task instruction, makes no Valen/Astra calls,
and waits for native termination. It retains the native recorder and FK audit;
legacy decision-log filenames do not indicate a model call. `baseline_summary.json`
and `run.json` explicitly identify the pure-pi05 condition.

For the Conda controller environment, dependency installation, and separated
service setup, see [Environment setup](ENVIRONMENT.md).

Use the repository's existing simulator and fresh pi05 services, checkpoint and
authorized Codex backend configuration. Valen must expose its Noul `/predict`
service and share the controller's absolute image filesystem. A tunnel alone
does not transfer image files. No services, credentials, models or simulator
assets are installed or launched by this entry point.

From the `GPT-as-Policy 2` repository root, in the controller environment:

```sh
python -m hybrid_rollout.robodojo.jev.run \
  --output /absolute/path/to/new-controller-run \
  --task arrange_largest_number \
  --checkpoint /absolute/path/to/your/pi05/checkpoint \
  --sim-port 19113 --student-port 18830 \
  --valen-url http://127.0.0.1:8001/predict \
  --codex /absolute/path/to/codex \
  --chunk-steps 25 --max-replans 0 --max-decisions 0 \
  --monitor-policy temporal --uncertainty-seconds 6
```

`--codex` selects the compact Astra backend for temporal mode. `--max-replans 0`
(the temporal and accumulating default) removes the call-count cap; a positive
cap stops when exhausted. Legacy defaults to three calls. CPU tests make no
paid calls; real experiments use the configured model account.

Alternatively use `--planner-url http://127.0.0.1:8003/replan` instead of
`--codex` for a separately provided HTTP Planner implementing both the version-2
background advisory envelope and version-1 paused recovery envelope in temporal mode. No default Planner is silently
selected. The new `refresh_observation` RPC requires the simulator server from
this updated source; an old server fails closed instead of relabeling cached
images as fresh captures.

### Direct image model API backend

Replace `--codex /path/to/codex` with `--planner-backend api` to use the local
image model gateway. Existing Codex `compact` and `agent` backends and external
`--planner-url` calls remain available; their defaults are unchanged.

```sh
python -m hybrid_rollout.robodojo.jev.run \
  --output /absolute/path/to/new-api-controller-run \
  --task arrange_largest_number \
  --checkpoint /absolute/path/to/your/pi05/checkpoint \
  --sim-port 19113 --student-port 18830 \
  --valen-url http://127.0.0.1:8001/predict \
  --planner-backend api \
  --api-base-url http://127.0.0.1:8317/v1 \
  --api-key-file /absolute/path/to/local-proxy-key \
  --monitor-policy temporal --chunk-steps 25 --max-replans 0 --max-decisions 0
```

The URL and key flags are optional: defaults are `http://127.0.0.1:8317/v1`
and this repository's `runtime/local-codex-api/api-key`. This is the gateway's
local access key, not Codex `auth.json`. The gateway in `tools/codex_api` must
already be running and reachable from the controller. The API adapter uses
Python's standard-library HTTP client and adds no SDK dependency.

The API backend calls `/v1/responses` directly with `gpt-6-astra` / `high`,
independent input, `store=false` and `tools=[]`. It creates no Codex process or
thread and does not load AGENTS.md, skills or MCP. It sends the same explicit
compact task instructions, output schema, selected images (up to three
timestamps from three cameras), state, and recent review summaries as the Codex
compact backend. Each selected image is sent as base64 bytes every time.
There is no implicit conversation or cross-call image cache. Codex may add its
own instructions/configuration, so the two backends are not guaranteed to have
identical complete model input.

Both background advice (version 2) and fresh paused recovery (version 1) are
supported. Background output can only continue or request recovery. Paused
responses retain native dual-arm, identity, evidence gate and action-bound
validation; the scheduler still owns execution and native task success.
`--astra-action-chunk none|sampled|recent`, `--astra-action-steps N`, and
`--astra-output-format full|mode-specific` work for API calls as for compact
Codex. Model effort stays `high`, independently of LIBERO's `xhigh` setting.

API mode cannot be combined with `--codex` or `--planner-url`; API URL/key flags
require `--planner-backend api`. `--planner-url` is a rollout `/replan` endpoint
with a different protocol and cannot be set to the model gateway's `/v1` URL.
For the JSON supervisor launcher, make the same replacement in
`controller.argv`; no additional planner service is needed.

`planner_backend.json` records the selected backend. Each API review writes
`astra_api_<identity-hash>/` containing the request, compact input, image paths
and hashes, explicit instructions, schema, model response, validated response,
token usage and timing. The key and image base64 are not logged. Failed or
repeated request IDs are not automatically retried; incomplete/refused/invalid
responses stop through the existing scheduler error handling. Closing the
planner discards in-flight answers, although remote inference may still finish.
PNG/JPEG/WebP bytes and the gateway's 16 MiB body limit are checked; images are
not silently resized or dropped. API timing records total preparation and
request duration; it does not fabricate Codex thread or first-output timings.

`--monitor-timeout`, `--policy-timeout`, `--planner-timeout`,
`--max-evidence-age`, `--max-evidence-gap`, and `--history` control bounded waits
and evidence freshness. The default evidence age is 120 seconds. If planning
outlasts that budget in temporal or accumulating mode, the owner obtains a new non-stepping
observation and requires identical episode/step, measured robot state and EEF
poses before accepting the answer. Native RoboDojo refresh renders without
stepping physics; RTX image bytes may differ from sampling noise. Changed
physical state rejects the old decision. The new capture contributes no accumulation vote. Legacy mode still
rejects expired evidence. Capture times are never fabricated. Timeouts cannot forcibly cancel remote GPU work. Pending worker
flags are recorded; the run stops and is not automatically resumed/retried.
Unresolved `PAUSED_*` results are fail-closed controller exits: the episode is
finalized as incomplete without resetting its scene. There is no automatic
resume command or background retry after exit.

### Operating Order

1. Run the CPU tests below first. They require no model or simulator service.
2. On the configured Linux/GPU host, use three separate environments for
   RoboDojo/Isaac, OpenPI/pi05, and Valen. Use this updated checkout in the
   simulator and controller. Do not substitute the LIBERO/OpenVLA driver.
3. Start a fresh simulator and pi05 server for the episode. Example commands
   below assume their upstream packages/assets are already installed and
   importable. Use an existing writable run directory; `controller` must not
   already exist. Set paths explicitly in each terminal.

```sh
# Simulator environment, from the repository root:
python -m hybrid_rollout.robodojo.robodojo_server.server \
  --task arrange_largest_number --output /runs/jev-check/sim --port 19113

# OpenPI environment, from the same repository root:
python -m hybrid_rollout.robodojo.pi05_server.server \
  --checkpoint /absolute/path/to/your/pi05/checkpoint \
  --port 18830 --identity-output /runs/jev-check/pi05-identity.json
```

4. Start your existing Valen HTTP service on port 8001 and wait for it to load.
   It must support the multimodal Noul payload and absolute shared image paths.
   The repo supplies the client, not Valen weights or a Valen server installer.
   Wait for the simulator's `ready` log and pi05's `loaded` log. **Do not probe
   port 19113 with curl/nc:** the simulator accepts one owning connection, and
   a probe/disconnect can end that server. Do not start the original `skill.run`
   controller alongside the Jev controller.
5. Run the controller command above with
   `--output /runs/jev-check/controller`. For the first live check, use
   `--max-decisions 10` instead of `0` to bound the experiment; this can produce
   an intentionally incomplete episode. Inspect `jev_events.jsonl`,
   `jev_summary.json`, and `result.json` before trying a full native horizon.

This is a manual operating recipe, not a claim that GPU setup has been tested
on this Mac. Configure the repository's authorized Codex profile first; its
checked-in gateway is a placeholder. A completed controller exit is not by
itself task success: inspect the native `success` field.

## HTTP Planner Contract

The request includes `version: 1`, `request_id`, `run_id`, `episode_id`,
`plan_revision`, `subgoal_id`, `program_id`, `current_scene`, `failure`,
`confirmation_reports`, `evidence_frames`, `previous_attempts`,
`original_task`, `remaining_task`, `remaining_plan`, `native_request`, and
`max_chunk_steps`. `native_request.student_instruction` records the actual
active subgoal prompt; `native_request.instruction` preserves the original task.

Return **exactly** the six identity fields plus `decision`:

```json
{
  "request_id": "echo outer request_id",
  "run_id": "echo run_id",
  "episode_id": "echo episode_id",
  "plan_revision": 0,
  "subgoal_id": "echo subgoal_id",
  "program_id": "echo program_id",
  "decision": {
    "request_id": "echo native_request.request_id",
    "mode": "stop",
    "steps": 1,
    "reason": "No supported local recovery.",
    "assessment": {
      "task_progress": {
        "verified_completed": [],
        "currently_attempting": "Retry the current task phase",
        "remaining": ["Finish the original task"]
      },
      "current_subgoal": "Retry the current task phase",
      "execution_status": "failed",
      "execution_evidence": "Describe the actual failure evidence here.",
      "expected_next_intent": "Describe the required next phase here.",
      "predicted_next_intent": "Describe the supplied proposal here.",
      "intent_status": "uncertain",
      "intent_evidence": "Describe the available evidence here."
    }
  }
}
```

The example is a schema illustration, not measured failure evidence. For `eef`,
include explicit `target.left` and `target.right` with `position`,
`quaternion_wxyz`, and boolean `gripper_closed`. Targets are ARX X5 **link6 in
environment-origin coordinates**, not Panda geometry. Existing 5 cm/0.35 rad
limits and native joint/gripper constraints remain unchanged. `edit` uses the
existing dual-arm edit schema. `steps` cannot exceed `max_chunk_steps` (default
one). The inner ID binds the current proposal; the outer ID binds this recovery
event. All identities must match, including revision type.

For a high-level replan, use the same outer envelope and this alternative
`decision` shape (no native action executes when it is installed):

```json
{
  "request_id": "echo native_request.request_id",
  "mode": "replan",
  "reason": "Restore the prerequisite before continuing the original task.",
  "subgoals": [
    {
      "subgoal_id": "reacquire",
      "instruction": "Pick up the target object again.",
      "success_condition": "The target object is visibly lifted and held."
    },
    {
      "subgoal_id": "place",
      "instruction": "Place the target object at its intended destination.",
      "success_condition": "The target rests at the destination after release."
    }
  ]
}
```

There must be 1-8 unique subgoals; `original-task` is reserved for the host's
final obligation. Native decision bounds, monitoring thresholds and global
budgets cannot be changed by a plan. The host validates structure, not semantic
plan quality. Visual completion can be wrong; native success remains separate.
This is a declarative instruction plan, **not geometric path planning or an
arbitrary-code executor**. A learned predictive-intent gate remains future work.

## Validation And Artifacts

```sh
python -m pytest hybrid_rollout/robodojo/test_jev_accumulation.py \
  hybrid_rollout/robodojo/test_jev.py \
  hybrid_rollout/robodojo/test_contract.py -q
```

CPU fixtures cover temporal confirmation, applicability, unknown/replayed/stale
evidence, asynchronous overlap, same-scene recovery through the native EEF
path, one reset, budget persistence, late/invalid Planner replies, model
timeouts, supplemental observations without duplicate votes, post-correction
verification, real subgoal prompts/advancement, and a decision-only Planner
without actuator handles. They do not
establish actual Valen accuracy, Astra task success, GPU runtime compatibility
or physical-robot safety. No physical robot driver is added.

Runs retain normal observation/proposal/execution artifacts and add
`jev_events.jsonl`, `jev_replan_NNN.json`, `jev_summary.json`, discarded
`jev_inference/` proposals, fresh `jev_refresh/` captures, and per-request
`jev_planner_*/` audit workspaces. `jev_plan_NNN.json` records plan activation;
`jev_discarded/` preserves prepared proposals invalidated by high-level plans.
Native outcome remains in `result.json`.

## Flow Review

| Requested branch | Execution behavior | Fixture coverage |
| --- | --- | --- |
| Normal -> VLA | Review last chunk before committing the next | `test_monitor_and_student_overlap_without_execution_leading` |
| Uncertain -> observe/pause | Fresh sensor capture, no physics, bounded retries | `test_supplemental_clear_resumes_with_a_new_proposal` |
| Repeated paused observations | Never accumulate confirmation votes | `test_repeated_paused_high_scores_cannot_confirm_a_failure` |
| Failure -> Astra | Two/three applicable endpoints, one request per event | `test_same_scene_recovery_uses_native_eef_and_returns_to_student` |
| VLA self-recovery / stop | Planner can select either; no forced EEF | `test_planner_can_choose_self_recovery_or_stop` |
| EEF -> verify -> VLA | Inference waits for recovery validation | `test_vla_is_regenerated_only_after_recovery_verification` |
| High-level replan | New instructions reach pi05; confirmed subgoals advance | `test_high_level_replan_reaches_actual_policy_and_retains_original_task` |

Before synchronizing this flow revision into `GPT-as-Policy 2`, a complete
backup was created alongside the repo:
`../GPT-as-Policy-2-backup-20261009-102419.tar.gz` relative to the repository root.
It includes `.git`, uncommitted changes and untracked files. Its extracted files
were compared against the pre-edit checkout with no differences. SHA-256:
`5e80e9a3f260d9017255ce120415dc94d0973c50fb56f57375071cba8302786d`.
For recovery, extract into a **new directory** and compare; do not overwrite a
working checkout blindly.
