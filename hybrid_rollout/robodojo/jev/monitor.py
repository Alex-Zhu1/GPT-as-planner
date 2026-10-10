"""Valen supplies evidence; a single asynchronous worker confirms constraints."""
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from queue import Empty, Queue
from threading import Thread
import time
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


QUESTIONS = {
    'target_mismatch': 'Is the robot manipulating an object inconsistent with the original task?',
    'collision': 'Do the images show unintended contact or collision during the executed motion? '
                 'Do not predict future collision safety.',
    'grasp_attempted': 'Has grasping or carrying the target actually been attempted? '
                       'Merely approaching an ungrasped object is not a grasp attempt.',
    'grasp_failure': 'Do attempted grasps visibly fail, slip or drop the object? '
                     'Not yet holding an object before any attempt is not failure.',
    'progress_attempted': 'Does the sequence show an actual attempt to make progress rather than '
                          'an intentional wait? Insufficient temporal evidence is uncertain.',
    'stuck': 'Does the sequence show repeated unsuccessful behavior despite attempted progress? '
             'A single frame or an intentional hold does not establish this.',
    'subgoal_complete': 'Is the explicit success_condition of the active subgoal visibly satisfied? '
                        'If success_condition is absent, answer low. Local completion is not whole-task success.',
}
CONSTRAINTS = {'target_mismatch': (None, 2), 'collision': (None, 2),
               'grasp_failure': ('grasp_attempted', 2), 'stuck': ('progress_attempted', 3)}
IDENTITY = ('run_id', 'episode_id', 'plan_revision', 'subgoal_id', 'program_id')


@dataclass(frozen=True)
class MonitorConfig:
    low: float = .2
    high: float = .8
    max_age_s: float = 120.
    max_gap_s: float = 120.
    max_step_gap: int = 1
    history: int = 2

    def __post_init__(self):
        for value in (self.low, self.high, self.max_age_s, self.max_gap_s):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Monitor limits must be finite numbers')
        if not 0 <= self.low < self.high <= 1 or min(self.max_age_s, self.max_gap_s) <= 0:
            raise ValueError('Invalid monitor thresholds or time limits')
        for value in (self.max_step_gap, self.history):
            if type(value) is not int or not 1 <= value <= 100:
                raise ValueError('History and step gap must be integers in [1, 100]')

    @property
    def program_id(self):
        spec = dict(config=asdict(self), questions=QUESTIONS, constraints=CONSTRAINTS)
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class JSONEndpoint:
    def __init__(self, url, timeout_s):
        parsed = urlsplit(url)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('Use an explicit HTTP(S) endpoint without credentials/query/fragment')
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Endpoint timeout must be positive')
        self.url, self.timeout_s = url, timeout_s

    def post(self, payload):
        request = Request(self.url, data=json.dumps(payload, allow_nan=False).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
        with build_opener(NoRedirect()).open(request, timeout=self.timeout_s) as response:
            body = response.read(65537)
        if len(body) > 65536:
            raise ValueError('Model response exceeds 64 KiB')
        return json.loads(body)


class ValenMonitor(JSONEndpoint):
    def evaluate(self, frames):
        # Frames are allowlisted public observations, not simulator outcome packets.
        content = [{'type': 'text', 'text': json.dumps({'scope': 'RoboDojo post-action monitoring',
                    'frames': frames}, allow_nan=False)}]
        for frame in frames:
            for image in frame['images']:
                content += [{'type': 'text', 'text': f"Step {frame['step_id']}: {image['camera']}"},
                            {'type': 'image_url', 'image_url': {'url': image['path']}}]
        raw = self.post({'state': {'messages': [{'role': 'user', 'content': content}]},
                         'questions': {key: {'type': 'noul', 'instructions': text}
                                       for key, text in QUESTIONS.items()}})
        if not isinstance(raw, dict) or not isinstance(raw.get('answers'), dict) or set(raw['answers']) != set(QUESTIONS):
            raise ValueError('Valen must answer every compiled question')
        values = {}
        for key, answer in raw['answers'].items():
            if not isinstance(answer, dict) or answer.get('type', 'noul') != 'noul':
                raise ValueError('Expected Noul evidence')
            values[key] = answer.get('noul')
        validate_scores(values)
        return values


def validate_scores(values):
    if not isinstance(values, dict) or set(values) != set(QUESTIONS):
        raise ValueError('Evidence questions do not match the compiled program')
    for value in values.values():
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError('Evidence must contain finite probabilities in [0, 1]')


class ConstraintRuntime:
    def __init__(self, config):
        self.config = config
        self.identity = None
        self.last = None
        self.seen = set()
        self.counts = {key: 0 for key in CONSTRAINTS}
        self.completion_count = 0

    def invalidate(self):
        self.counts = {key: 0 for key in CONSTRAINTS}
        self.completion_count = 0

    def update(self, frame, scores, now=None):
        try:
            return self._update(frame, scores, time.time() if now is None else now)
        except Exception:
            self.invalidate()
            raise

    def _update(self, frame, scores, now):
        validate_scores(scores)
        config = self.config
        identity = tuple(frame[key] for key in IDENTITY)
        if frame['program_id'] != config.program_id or (self.identity is not None and self.identity != identity):
            raise ValueError('Wrong monitor generation; activate a fresh runtime')
        step, capture = frame['step_id'], frame['captured_at']
        if type(step) is not int or step < 1 or not math.isfinite(capture):
            raise ValueError('Expected a timestamped executed endpoint')
        if frame['observation_id'] in self.seen or (self.last and step <= self.last['step_id']):
            raise ValueError('Repeated evidence cannot confirm failure')
        age = now - capture
        if not 0 <= age <= config.max_age_s or (self.last and capture <= self.last['captured_at']):
            raise ValueError('Stale or non-increasing capture time')
        gap = bool(self.last and (step - self.last['step_id'] > config.max_step_gap
                                 or capture - self.last['captured_at'] > config.max_gap_s))
        if gap:
            self.invalidate()
        self.identity, self.last = identity, deepcopy(frame)
        self.seen.add(frame['observation_id'])
        checks = []
        for key, (guard, required) in CONSTRAINTS.items():
            applicable = scores[guard] if guard else 1.
            if applicable <= config.low:
                status = 'not_applicable'
            elif applicable < config.high or config.low < scores[key] < config.high:
                status = 'unknown'
            elif scores[key] <= config.low:
                status = 'clear'
            else:
                status = 'suspected'
            self.counts[key] = self.counts[key] + 1 if status == 'suspected' else 0
            if self.counts[key] >= required:
                status = 'violated'
            checks.append(dict(constraint_id=key, probability=scores[key], applicability=applicable,
                               status=status, confirmations=self.counts[key], required=required,
                               threshold=config.high))
        violations = [c for c in checks if c['status'] == 'violated']
        uncertain = any(c['status'] == 'unknown' for c in checks)
        clear = all(c['status'] in ('clear', 'not_applicable') for c in checks)
        self.completion_count = self.completion_count + 1 if (
            clear and frame.get('success_condition') and scores['subgoal_complete'] >= config.high) else 0
        return dict(**{k: frame[k] for k in IDENTITY}, step_id=step, captured_at=capture,
                    observation_id=frame['observation_id'], gap_reset=gap, scores=scores, checks=checks,
                    violations=violations, state='FAILED' if violations else
                    'UNCERTAIN' if uncertain else
                    'SUSPECTED' if any(c['status'] == 'suspected' for c in checks) else 'RUNNING',
                    subgoal_complete=self.completion_count >= 2)


class PendingCall:
    """Timeout detaches a result, never kills/retries a model or executes an action."""
    def __init__(self, call, timeout_s, name):
        self.deadline = time.monotonic() + timeout_s
        self.result = Queue(maxsize=1)
        self.thread = Thread(target=self._work, args=(call,), name=name, daemon=True)
        self.thread.start()

    def _work(self, call):
        try:
            value = call()
            self.result.put((True, value, time.monotonic()))
        except Exception as error:
            self.result.put((False, error, time.monotonic()))

    def get(self, discard=False):
        try:
            ok, value, completed = self.result.get(timeout=max(0, self.deadline - time.monotonic()))
        except Empty as error:
            raise TimeoutError(f'{self.thread.name} deadline exceeded') from error
        if discard:
            return None
        if completed > self.deadline:
            raise TimeoutError(f'{self.thread.name} returned after its deadline')
        if not ok:
            raise value
        return value


class AsyncMonitor:
    """Single in-flight evaluation with lossless backpressure at chunk boundaries."""
    def __init__(self, monitor, config, timeout_s):
        self.monitor, self.config, self.timeout_s = monitor, config, timeout_s
        self.runtime = ConstraintRuntime(config)
        self.frames = deque(maxlen=config.history)
        self.pending = None

    def submit(self, frame, *, inspection=False):
        if self.pending is not None:
            raise RuntimeError('Review the previous endpoint before submitting another')
        if inspection:
            if (self.runtime.last is None or frame['step_id'] != self.runtime.last['step_id']
                    or tuple(frame[k] for k in IDENTITY) != self.runtime.identity):
                raise ValueError('Supplemental observation must belong to the paused endpoint')
            # A new camera capture of unchanged physics never contributes another
            # confirmation vote. Clear old counts and inspect it in isolation.
            self.runtime.invalidate()
            frames = [*deepcopy(list(self.frames)), {**deepcopy(frame), 'inspection_only': True}]
            runtime = ConstraintRuntime(self.config)
        else:
            self.frames.append(deepcopy(frame))
            frames, runtime = deepcopy(list(self.frames)), self.runtime
        def evaluate():
            started = time.monotonic()
            try:
                report = runtime.update(frames[-1], self.monitor.evaluate(frames))
                report.update(completed_at=time.time(), inference_seconds=time.monotonic()-started,
                              render_to_report_seconds=time.time()-frames[-1]['captured_at'],
                              inspection_only=inspection)
                return report
            except Exception:
                runtime.invalidate()
                raise
        self.pending = PendingCall(evaluate, self.timeout_s, 'jev-monitor')

    def review(self):
        if self.pending is None:
            return None
        result = self.pending.get()
        self.pending = None
        return result

    def activate(self):
        if self.pending is not None:
            raise RuntimeError('Cannot activate while old evidence is in flight')
        self.runtime = ConstraintRuntime(self.config)
        self.frames.clear()
