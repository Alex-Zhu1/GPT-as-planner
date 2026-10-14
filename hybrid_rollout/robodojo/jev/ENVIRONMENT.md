# Jev controller environment

This recipe creates the CPU controller environment on a Linux server. It does
not install RoboDojo, Isaac Sim, OpenPI model inference, Valen weights, CUDA, or
an Astra backend. Keep those services in their own upstream environments.
The dependency ranges are a starting configuration, not a Linux/GPU-validated
lockfile. The existing offline test environment is not this Conda environment.

## Create the controller environment

Run from the root of this repository, using the server's actual checkout path:

```sh
conda env create -f environment-jev.yml
conda activate jev-controller

# Use the same RoboDojo-compatible OpenPI checkout as the pi05 server.
# Replace the path; this installs only its lightweight client package.
export OPENPI_ROOT=/absolute/path/to/compatible-openpi
python -m pip install -e "$OPENPI_ROOT/packages/openpi-client"
python -m pip check

python -c 'from openpi_client.websocket_client_policy import WebsocketClientPolicy; from hybrid_rollout.robodojo.robodojo_server.client import RoboDojoTools; from hybrid_rollout.robodojo.jev.planner import CodexPlanner; print("Controller imports OK")'
python -m hybrid_rollout.robodojo.jev.run --help
python -m pytest hybrid_rollout/robodojo/test_jev.py hybrid_rollout/robodojo/test_contract.py -q
python -m pytest hybrid_rollout/robodojo/test_subgoals.py hybrid_rollout/robodojo/test_subgoal_backends.py -q
```

Alternatively, create Python 3.11 with Conda and run
`python -m pip install -r requirements-jev.txt`, then install the client as above.
Do not use `--no-deps` to bypass a conflict. Resolve mismatches against the actual
server checkout. The current public OpenPI client declares NumPy below 2;
this controller recipe respects that constraint:
https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/pyproject.toml

The required OpenPI revision and robot-specific policy configuration are recorded
in `../SOURCE.json`. A generic latest OpenPI installation is not a substitute for
the compatible fork/configuration. Client installation alone does not provide
the model server, checkpoint, or its robot-specific normalization assets.

## External environments

| Process | Required environment | What this repo provides |
| --- | --- | --- |
| Controller | Python 3.11, these requirements, matching openpi-client | Jev scheduler, monitor client, recovery validation |
| Simulator | RoboDojo source/assets, Isaac Sim 5.1 and matching IsaacLab setup | Simulator server adapter |
| pi05 | Compatible OpenPI/JAX environment and checkpoint | Policy server entry point |
| Valen | Your Valen deployment's own dependencies and weights | HTTP Noul client only |
| Astra | Authorized Codex executable, local image model gateway, or external HTTP Planner | Recovery client and decision protocol |

Do not install Isaac, torch, transformers, or jax into the controller merely to
satisfy a different service. Their versions must come from the corresponding
upstream checkout/environment, not from a shared catch-all requirements file.
The simulator environment must be able to import RoboDojo's `env`, `task`,
`utils`, and `src` modules in addition to this repository.

For `--codex`, configure the repository's authorized backend/profile first; the
checked-in gateway is a `.invalid` placeholder. This requirements file does not
install or authenticate the Codex executable. `--planner-url` is the alternative
when a compatible HTTP Planner is already available.
`--planner-backend api` instead uses the running image gateway documented in
`README.md`; it requires no Codex executable in the controller environment and
uses the existing Python standard library for HTTP.

## Start the services

### One-command supervisor

Use separate environments but one launcher. No repeated `conda activate` is
needed: each child uses the absolute Python executable from its own environment.
Environment installation remains a one-time prerequisite, not a launcher action.

```sh
# From the repo root; private/ is ignored by Git.
mkdir -p private
cp hybrid_rollout/robodojo/jev/launch.example.json private/jev-launch.json
# Edit private/jev-launch.json for this server before continuing.
conda activate jev-controller
python -m hybrid_rollout.robodojo.jev.launch \
  --config private/jev-launch.json --output /runs/jev-trial-001 --check
python -m hybrid_rollout.robodojo.jev.launch \
  --config private/jev-launch.json --output /runs/jev-trial-001
```

Set the four executable paths, RoboDojo source directory, checkpoint paths in
both pi05/controller commands, GPU assignments, and authorized Astra backend.
Replace the Valen command with its real argv array, including its model/port
options, and ready_text with a log line emitted only after model AND HTTP
readiness. The example deliberately has no invented Valen entry point.
The example assigns GPUs 0/1/2; it is not a requirement for exactly three GPUs.
Choose assignments based on actual VRAM; sharing GPUs can cause OOM.

Each process inherits the launcher's environment, then applies its own `env`.
The launcher prepends this repository to PYTHONPATH. Use service-specific `env`
for additional library paths, CUDA settings, or Conda activation variables your
upstream setup requires. An absolute Python path does not run activation hooks.
`{repo}` and `{run}` expand to absolute paths. Commands are argv arrays, not
shell strings; no shell expansion, pipes, or implicit command evaluation occur.
Keep credentials outside this file and repository. Only use trusted configs.

The launcher starts Valen, pi05, then the simulator, waits for each log marker,
and starts the controller last. It checks port availability by binding temporary
sockets, never by connecting to the single-owner simulator. This cannot reserve
ports against another process racing to start. Do not run other launchers on the
same ports. Readiness markers indicate startup, not model correctness.

Logs are `valen.log`, `pi05.log`, `simulator.log`, and `controller.log` in the run
directory; controller artifacts are in its `controller/` subdirectory. The
launcher propagates the controller exit code and writes `launcher_result.json`;
exit code 0 alone is not proof of task success. Check native `result.json`.
On startup failure, Ctrl-C/SIGTERM, or controller exit it terminates only the
process groups it started. There is no automatic model or episode retry.
Use tmux for a foreground run that must survive an SSH disconnection. Managed
services must remain in the foreground, not daemonize or detach process groups.
This first launcher targets one Linux host, not a distributed cluster.

### Manual startup

In each terminal, activate its respective environment and change to this repo's
root. Create a writable run parent first. Example paths must be replaced; the
controller output directory must not already exist. Use a fresh simulator and
pi05 server for each trial.

```sh
# Terminal 1: configured RoboDojo environment
python -m hybrid_rollout.robodojo.robodojo_server.server \
  --task arrange_largest_number --output /runs/jev-check/sim --port 19113

# Terminal 2: configured OpenPI environment
python -m hybrid_rollout.robodojo.pi05_server.server \
  --checkpoint /absolute/path/to/checkpoint --port 18830 \
  --identity-output /runs/jev-check/pi05-identity.json

# Terminal 3: start your existing Valen deployment with /predict on port 8001.
# No Valen server launcher is supplied by this repository.

# Terminal 4: after all services are ready
conda activate jev-controller
python -m hybrid_rollout.robodojo.jev.run \
  --output /runs/jev-check/controller --task arrange_largest_number \
  --checkpoint /absolute/path/to/checkpoint \
  --sim-port 19113 --student-port 18830 \
  --valen-url http://127.0.0.1:8001/predict \
  --codex /absolute/path/to/codex \
  --monitor-policy subgoals --chunk-steps 25 --max-decisions 0 --max-replans 12
```

Replace `--codex ...` with `--planner-url http://127.0.0.1:8003/replan` when using
your external Planner implementing the version-5 subgoal-plan envelope. The Valen
server must support choice questions and return every candidate probability.
Use `--monitor-policy temporal` for the previous background-recovery protocol.
Model calls may incur charges.

The simulator and pi05 clients connect to loopback. For the first deployment,
run them on the same host with reachable ports. Valen must see image files at
the same absolute paths as the controller; a network tunnel does not transfer
files. Do not probe the simulator port with curl/nc or run a second controller.

Inspect `jev_events.jsonl`, `jev_summary.json`, and native `result.json` after
the bounded trial. Passing imports/tests does not establish model accuracy,
checkpoint compatibility, or GPU simulation success.
