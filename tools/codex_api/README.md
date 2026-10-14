# Local stateless Codex model API

A minimal Python implementation of the Codex HTTP flow inspected in
[CLIProxyAPI v8.0.24](https://github.com/router-for-me/CLIProxyAPI/tree/v8.0.24),
the core version pinned by EasyCLIProxyAPI during this investigation.
Reference commit: `091ff27f31285ec4fd1dd5823b44f7f51fb8f48b`.

The proxy reads the existing Codex `auth.json` and sends HTTP requests directly
to `https://chatgpt.com/backend-api/codex/responses`. It never starts Codex,
loads AGENTS.md/skills/MCP, executes tools, or maintains conversation history.
The shared HTTP client pools upstream connections. Up to eight inference
requests can run concurrently; account limits still apply.

## Start on this machine

From the repository root:

```bash
python -m pip install -r tools/codex_api/requirements.txt
# Reuse this machine's existing outbound network proxy, without printing it.
. /mnt/data_hdd/mhzhu/tools/codex/proxy-env.sh
python tools/codex_api/server.py \
  --auth-file /mnt/data_hdd/mhzhu/tools/codex/home/auth.json \
  --key-file runtime/local-codex-api/api-key \
  --port 8317
```

Only `127.0.0.1` is bound. A random local API key is created with mode 0600.
`runtime/` is ignored by Git. All endpoints require this local bearer key.
The local key is separate from the Codex access/refresh tokens.

## Call with Python

```python
from pathlib import Path
from openai import OpenAI

repo = Path('/mnt/data_hdd/mhzhu/Jev/GPT-as-Policy/GPT-as-Policy')
client = OpenAI(
    base_url='http://127.0.0.1:8317/v1',
    api_key=(repo / 'runtime/local-codex-api/api-key').read_text().strip(),
    timeout=180,
    max_retries=0,
)

result = client.chat.completions.create(
    model='gpt-6-astra',
    messages=[{'role': 'user', 'content': 'Reply with exactly OK.'}],
)
print(result.choices[0].message.content)

# Responses preserves the explicit instructions and tools you supply.
result = client.responses.create(
    model='gpt-6-astra',
    instructions='',
    tools=[],
    input='Reply with exactly OK.',
    store=False,
)
print(result.output_text)
```

Every call contains only its supplied input. There is no thread ID or automatic
history replay. To provide history, send it explicitly in messages/input.
For images, use Chat Completions `image_url` content parts or Responses
`input_image`. Function calls are returned to the caller and never executed.

## Supported endpoints and explicit transformations

- `GET /health`: process/configuration health, not an upstream inference check.
- `GET /v1/models`: retrieves the current upstream model catalog.
- `POST /v1/responses`: JSON or SSE (`stream=true`).
- `POST /v1/chat/completions`: non-streaming text/images/function calls and
  structured-output mapping. Chat streaming returns an explicit 400.

The upstream requires `stream=true` and `store=false`; non-streaming clients
receive the collected completed response. Missing `instructions` becomes the
empty string; missing `tools` becomes an empty array. Explicit values are
preserved. `system` messages map to `developer` as required by this route.
Chat messages and function calls map to Responses items. No coding prompt,
image-generation tool, reasoning setting, or previous input is injected.

Unsupported fields return 400 instead of being silently discarded. This
includes `temperature`, `top_p`, token limits, and `previous_response_id`.
This is a deliberately limited compatibility layer, not the entire OpenAI API.

## Authentication and operational limits

The service rereads the existing auth file for each request, so credentials
renewed by Codex are picked up without restarting. It does **not** refresh or
rewrite the shared OAuth credentials itself: independently rotating a refresh
token could interfere with the user's running Codex clients. If upstream
returns 401, renew the login through Codex and retry. This initial service is
not a self-contained unattended OAuth lifecycle manager.

There is no inference retry and no request/response body logging. Ordinary
network access uses the process's proxy environment. Upstream redirects are
not followed. The API does not promise that OpenAI retains no records;
`store=false` is a protocol setting, not a zero-retention guarantee.

This uses a Codex subscription backend, not the public API-key contract, so
availability, supported fields, and model permissions can change.

## Verification

```bash
python -m unittest tools.codex_api.test_server -v
```

Tests cover exact outbound prompt/tool content, no history carryover, external
credential renewal, image/tool conversion, unsupported experiment parameters,
SSE framing, and failed or truncated upstream responses.

An explicit real-inference smoke test is available (three short requests):

```bash
python -m tools.codex_api.smoke
```

On 2026-10-13 the live `gpt-6-astra` checks returned `API_OK` (2.23 s),
`NO_HISTORY` in a separate request (3.39 s), and `STREAM_OK` over SSE (2.54 s).
Local authentication, browser-origin rejection, model catalog retrieval, and
unsupported-parameter rejection also passed. Non-secret results are saved in
`runtime/local-codex-api/smoke.json`. The detached process PID for this setup is
recorded in `runtime/local-codex-api/server.pid`; startup output is in
`runtime/local-codex-api/server.log`. No automatic boot startup was installed.

## Transport diagnostics

The gateway now writes JSON metadata to stderr (the running service's
`runtime/local-codex-api/server.log`). `/health` reports `diagnostics_version=1`
when the updated implementation is loaded. Every call has a local 32-hex
`request_id`, accepted from `X-Client-Request-Id` or generated locally, and echoed
as `X-Request-Id`. It is never forwarded upstream. Logs include:

- Local request-body receipt and admission; HTTPX connection/TLS, upload and
  response-header trace stages; upstream HTTP status.
- First/last SSE line and event times, counts, and allowlisted event types.
  Heartbeats every 30 seconds distinguish no network activity from an active
  stream without a terminal event. A closed downstream peer is reported without
  changing request execution or adding inference retries.
- Exception class and error category, including connect/read/write/pool timeout,
  proxy/protocol failure, upstream HTTP error, truncated stream and generation
  failure. Allowlisted upstream failure codes/incomplete reasons are retained;
  unknown values are marked `unknown`. No raw exception messages, proxy URLs,
  auth headers, account IDs, prompts, images or output text enter these logs.

`hybrid_rollout/model_api.py` records client HTTP start, response headers, body
receipt, decoding and failures. RoboDojo API calls write these to
`astra_api_*/client_events.jsonl`; their local request IDs join directly to gateway
logs. Other callers default to stderr. Error bodies are read only to extract
allowlisted codes; raw messages are discarded. Failed RoboDojo timing files
retain `transport_diagnostics`, and paused scheduler timeouts snapshot the
transport state in a `planner_timeout` event even if the worker has not returned.

Timeout meanings are unchanged: HTTPX's 180 seconds limits individual network
waits, not the total generation duration. SSE activity can keep that connection
alive longer. Non-streaming clients wait for the collected terminal response;
RoboDojo's HTTP timeout and overall planner deadline are both 300 seconds. The
outer deadline may fire first and mask the eventual HTTP exception. Increasing
timeouts alone does not diagnose this condition. Historical requests made before
these diagnostics cannot be reconstructed from the old startup-only logs.

Offline fault injection (loopback HTTP plus a mocked upstream, no model calls):

```bash
python -m unittest tools.codex_api.test_server tools.codex_api.test_diagnostics -v
```

Existing experiment source snapshots remain immutable. Future runs must freeze
the updated client/planner/runner to receive these diagnostics; old custom HTTP
error observers that consume the error body should not wrap the new client.

Relevant inspected source files:

- `internal/runtime/executor/codex_executor_execute.go`: endpoint and SSE flow.
- `internal/runtime/executor/codex_executor_request.go`: bearer/account headers;
  also contains the default image-generation tool injection omitted here.
- `internal/translator/codex/openai/responses/codex_openai-responses_request.go`:
  required stream/store settings and system-to-developer conversion.

No CLIProxyAPI implementation files are vendored into this repository.
