# Asynchronous Valen monitoring and Astra recovery

This is an **opt-in experimental RoboDojo entry point**, independent of the
unchanged `skill.run` full-time Astra reviewer and GPT-only baseline. It uses
Valen (the multimodal Jev-inspired model), not the text-only TypeSafe Jev API.
No dependency on the sibling `Jev` checkout is needed.

```text
pi05 inference --------------------------> native student control
      |                                          |
      |                                  RGB x3 + measured state
      |                                          |
      +---- overlaps ---- Valen worker -> constraint runtime
                                          | clear: continue
                                          | uncertain: pause + fresh capture
                                          | first high: SUSPECTED
                                          | confirmed failure
                                          v
                                pause + discard old proposal
                                          |
                            fresh same-scene capture + pi05
                                          |
                                  Astra recovery Planner
                                          |
                       student / edit / EEF / high-level plan / stop
                                          |
                    execute -> verify new observation -> fresh student
```

## What Is Implemented

- Valen reads the high, left-wrist and right-wrist cameras, measured proprio,
  actual executed actions, original task and active recovery context. Rewards,
  simulator success and hidden object state are not sent to it.
- pi05 inference and Valen run concurrently. Simulator RPC, FK and execution
  stay on one owning thread. There is only one call per model in flight.
- The scheduler permits **one unreviewed chunk**, not an unlimited observation
  backlog. Default `--chunk-steps 1` therefore permits at most one unreviewed
  control. An explicit 2-5 step chunk permits that many unreviewed controls.
  Monitoring is at chunk endpoints, not every physics substep. No endpoints
  are dropped; throughput is limited by the slower model.
- Target mismatch, observed collision and grasp failure require two consecutive
  applicable scores >= 0.8. Stuck requires three. Scores <= 0.2 break a chain;
  intermediate/invalid scores, uncertain applicability, repeated captures,
  stale evidence and excessive step/time gaps also break it. These thresholds
  are starting settings, **not calibrated robot failure probabilities**.
- Separate applicability questions prevent an unattempted grasp or intentional
  hold from automatically becoming a grasp/stuck failure. Applicability is
  model evidence, not an oracle; it still needs robot-domain validation.
- Only confirmed failures invoke Planner. Multiple violations at one endpoint
  are delivered together. Unavailable monitoring pauses/stops without invoking
  Planner. A single high score enters `SUSPECTED` and another actual executed
  endpoint is required to confirm it.
- An unknown execution check stops advancement and discards speculation. Up to
  `--max-observation-refreshes` fresh, non-stepping captures are inspected
  (default one). A clear inspection allows a new policy proposal. An unclear
  inspection leaves `PAUSED_UNCERTAIN`; it never triggers Planner. Repeated
  captures at the same control step contribute **zero additional confirmation
  votes**, including for subgoal completion. A normal incomplete subgoal alone
  is not an execution failure or reason to pause.
- During replanning there are zero `chunk_step` calls. The old policy call is
  drained within its original deadline and its result discarded, then a fresh
  sensor capture and fresh proposal are obtained at the same simulator step.
  No reset or rollback occurs. A stuck policy stops the run rather than opening
  another connection or overlapping inference against the same model.
- Astra receives the paused scene separately from versioned failure history,
  previous attempts, original task, native proposal/FK and immutable monitor
  limits. It can choose student self-recovery, `edit`, `eef`, a declarative
  high-level `replan`, or `stop`.
- The existing outcome/intent takeover gate and both-arm position/quaternion/
  gripper checks still validate Planner decisions. EEF execution uses the
  existing local IK controller; the Planner never receives an actuating object.
- Each accepted recovery increments the plan revision and installs a fresh
  confirmation runtime/history. Old Planner identities are rejected. The
  original task, global control history and recovery budget are never reset.
- After a local Planner action, no new policy inference starts until the
  resulting observation passes monitoring. An unclear result gets supplemental
  inspection; if it remains unclear or suspicious, the controller records
  `PAUSED_RECOVERY_UNVERIFIED` without handing back. A clear check is evidence
  for continuation, **not proof of target arrival, grasp success or task success**.
- High-level replans provide ordered subgoals with observable success conditions.
  Their instructions actually reach pi05; two distinct, clean executed
  endpoints are required for visual subgoal completion. Advancing invalidates
  old proposals and monitoring history. The original task is always retained
  as a final obligation, even if the supplied subgoals omit remaining work.
  Only native simulator success counts as task success. Exhausted budgets,
  unavailable models, invalid plans or Planner refusal never resume old actions.

This variant deliberately does **not** claim the original per-chunk Astra
predictive-intent review. Automatic student decisions record `uncertain` intent
and `decision_author=jev_scheduler`, not invented visual/intent assessments.
Existing decision artifacts retain their legacy filenames/schema, but carry
`evaluation_method=pi05_valen_replan`; do not mix these runs into published
baseline results.

## Run

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
  --chunk-steps 1 --max-replans 3 --max-decisions 0 \
  --max-observation-refreshes 1
```

`--codex` reuses the repository's pinned Astra `CodexPolicy` and its normal
analysis tools. It is started **only after confirmed failure**, in a separate
decision-only turn. Its apparent execute tool records the selected decision;
only the scheduler can execute that decision after independent validation.
Each recovery request includes prior same-episode attempts. No real paid model
calls are made by tests. Running this command can incur model costs.

Alternatively use `--planner-url http://127.0.0.1:8003/replan` instead of
`--codex` for a separately provided HTTP Planner. No default Planner is silently
selected. The new `refresh_observation` RPC requires the simulator server from
this updated source; an old server fails closed instead of relabeling cached
images as fresh captures.

`--monitor-timeout`, `--policy-timeout`, `--planner-timeout`,
`--max-evidence-age`, `--max-evidence-gap`, and `--history` control bounded waits
and evidence freshness. The default evidence age is 120 seconds. If planning
outlasts that budget, its answer is rejected rather than executed on expired
evidence. Measure latency and tune these explicitly; do not fabricate new
capture times. Timeouts cannot forcibly cancel remote GPU work. Pending worker
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
python -m pytest hybrid_rollout/robodojo/test_jev.py \
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
