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
VISUAL_INPUT_FORMAT = 'jev_scene_evidence_v1'
VISUAL_FRAME_FIELDS = (*IDENTITY, 'step_id', 'observation_id', 'captured_at',
    'task', 'subgoal', 'success_condition', 'executed_start_step',
    'context_only', 'inspection_only')


def visual_frame(frame):
    """Project host audit records to scene evidence at the model boundary."""
    result = {key: deepcopy(frame[key]) for key in VISUAL_FRAME_FIELDS if key in frame}
    result['images'] = [{key: image[key] for key in ('camera', 'path')}
                        for image in frame['images']]
    return result

# Opt-in profile: separate task advancement from motion and safety constraints.
TASK_QUESTIONS = {
    'task_advancing': 'Between the two latest distinct executed endpoints, is there NEW visible progress '
        'toward the original task or a necessary prerequisite? Robot motion, repeated failed attempts, '
        'or progress already visible at both endpoints do not count. If ambiguous answer uncertain.',
    'task_satisfied': 'Are ALL requirements of the original task visibly satisfied now? '
        'Do not substitute completion of a recovery instruction for the original task.',
    'target_visible': 'Is the object being manipulated for the current subgoal identifiable in the latest views? '
        'If multiple possible targets cannot be distinguished, answer uncertain.',
    'target_held': 'Is that target visibly held by the gripper, supported by relative motion across views? '
        'A closed gripper command alone is not evidence.',
    'target_on_support': 'Is that target visibly resting on an environmental support surface now, '
        'rather than suspended by the gripper?',
    'target_at_destination': 'Is that target visibly at the destination required by the original task? '
        'If no destination is specified or identifiable, answer uncertain.',
    'scene_comparable': 'Can the target and its relations be reliably compared between the two latest '
        'distinct executed endpoints? Camera/view changes or occlusion are not object movement. '
        'If continuity is lost answer low or uncertain.',
    'arm_approaching': 'Does the latest executed segment visibly approach or align the gripper with '
        'the current target? Judge executed behavior, not the instruction alone.',
    'arm_transporting': 'Does the latest executed segment visibly transport the target together with the gripper?',
    'arm_releasing': 'Does the latest executed segment visibly release the target from the gripper?',
}


# Candidate descriptions, not IDs, are the text scored by Valen.
CHOICES = {
    'scene_comparable': ('Compare the latest two endpoints, matching cameras and using both views.', {
        'comparable': 'The target and its relations can be reliably compared.',
        'view_changed_comparable': 'The viewpoint changed but target continuity remains reliable.',
        'insufficient_evidence': 'Occlusion, ambiguity or viewpoint change prevents reliable comparison.'}),
    'task_advancing': ('Compare the latest executed segment against the CURRENT instruction, including necessary retreat or alignment. Motion alone is not progress.', {
        'advancing': 'There is new visible progress toward the current instruction.',
        'attempt_no_effect': 'An attempt is visible without visible progress toward the instruction.',
        'no_attempt': 'No actual attempt is visible.',
        'insufficient_evidence': 'The images do not reliably distinguish progress from no progress.'}),
    'stuck': ('Compare the entire chronological window. Judge executed behavior, not proposed actions.', {
        'repeated_no_effect': 'Repeated attempts are visible without visible effect toward the current instruction.',
        'advancing': 'The sequence shows visible progress.',
        'no_repeated_attempt': 'Repeated attempts are not visible.',
        'insufficient_evidence': 'Temporal or visual evidence is insufficient.'}),
    'grasp_failure': ('Compare actual grasp attempts and their aftermath across the supplied views.', {
        'stable_hold': 'The target visibly moves stably with the gripper.',
        'failed_attempt': 'An attempted grasp visibly misses, slips or drops the target.',
        'no_attempt': 'No grasp attempt is observed.',
        'insufficient_evidence': 'Grasp outcome cannot be reliably distinguished.'}),
    'subgoal_complete': ('Evaluate only the explicit recovery success_condition against current and recent images.', {
        'satisfied': 'The stated condition is visibly satisfied.',
        'not_satisfied': 'The stated condition is visibly not satisfied.',
        'not_applicable': 'No recovery success condition is provided.',
        'insufficient_evidence': 'The condition cannot be reliably evaluated from the images.'}),
}
PROJECTIONS = {'scene_comparable': ('comparable','view_changed_comparable'),
    'task_advancing': ('advancing',), 'stuck': ('repeated_no_effect',),
    'grasp_failure': ('failed_attempt',), 'subgoal_complete': ('satisfied',)}


def compiled_questions(profile):
    result = {k: dict(type='noul', instructions=v) for k,v in question_spec(profile).items()}
    if profile == 'task_progress':
        result.update({k: dict(type='choice', instructions=v[0], criteria=v[1]) for k,v in CHOICES.items()})
    return result


class EvidenceScores(dict):
    """Numeric compatibility view plus lossless model answers for the board."""
    def __init__(self, values, answers):
        super().__init__(values)
        self.answers = deepcopy(answers)


def question_spec(profile):
    if profile not in ('constraints', 'task_progress'):
        raise ValueError('Unknown observation profile')
    return dict(QUESTIONS, **TASK_QUESTIONS) if profile == 'task_progress' else dict(QUESTIONS)


@dataclass(frozen=True)
class MonitorConfig:
    low: float = .2
    high: float = .8
    max_age_s: float = 120.
    max_gap_s: float = 120.
    max_step_gap: int = 1
    history: int = 2
    observation_profile: str = 'constraints'

    def __post_init__(self):
        question_spec(self.observation_profile)
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
        spec = dict(config=asdict(self), questions=compiled_questions(self.observation_profile),
                    constraints=CONSTRAINTS, input_format=VISUAL_INPUT_FORMAT,
                    frame_fields=VISUAL_FRAME_FIELDS)
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
    def __init__(self, url, timeout_s, *, scope='RoboDojo post-action monitoring', observation_profile='constraints'):
        super().__init__(url, timeout_s)
        self.scope = scope
        self.questions = question_spec(observation_profile)
        self.compiled = compiled_questions(observation_profile)

    def evaluate(self, frames):
        # All evaluation paths share this boundary; host audit records retain controls.
        frames = [visual_frame(frame) for frame in frames]
        content = [{'type': 'text', 'text': json.dumps({'input_format': VISUAL_INPUT_FORMAT, 'scope': self.scope,
                    'frames': frames}, allow_nan=False)}]
        for frame in frames:
            for image in frame['images']:
                content += [{'type': 'text', 'text': f"Step {frame['step_id']}: {image['camera']}"},
                            {'type': 'image_url', 'image_url': {'url': image['path']}}]
        raw = self.post({'state': {'messages': [{'role': 'user', 'content': content}]},
                         'questions': self.compiled})
        if not isinstance(raw, dict) or not isinstance(raw.get('answers'), dict) or set(raw['answers']) != set(self.questions):
            raise ValueError('Valen must answer every compiled question')
        values = {}
        for key, answer in raw['answers'].items():
            expected = self.compiled[key]
            if not isinstance(answer, dict) or answer.get('type') != expected['type']:
                raise ValueError('Unexpected Valen answer type')
            if expected['type'] == 'choice':
                probabilities = answer.get('probabilities')
                if not isinstance(probabilities, dict) or set(probabilities) != set(expected['criteria']):
                    raise ValueError('Choice probability keys must match candidates')
                validate_scores(probabilities, expected['criteria'])
                if not math.isclose(sum(probabilities.values()), 1., abs_tol=1e-5):
                    raise ValueError('Choice probabilities must sum to one')
                values[key] = min(1., sum(probabilities[k] for k in PROJECTIONS[key]))
            else:
                values[key] = answer.get('noul')
        validate_scores(values, self.questions)
        return EvidenceScores(values, raw['answers'])


def validate_scores(values, questions=QUESTIONS):
    if not isinstance(values, dict) or set(values) != set(questions):
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
        validate_scores(scores, question_spec(self.config.observation_profile))
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
        if config.observation_profile=='task_progress':
            statuses={'violated':'high_score_candidate','suspected':'high_score_candidate',
                      'clear':'low_score','unknown':'unresolved_scores','not_applicable':'low_applicability_score'}
            checks=[{**c,'status':statuses[c['status']]} for c in checks]
        return dict(**{k: frame[k] for k in IDENTITY}, step_id=step, captured_at=capture,
                    observation_id=frame['observation_id'], gap_reset=gap, scores=dict(scores),
                    raw_answers=deepcopy(getattr(scores, 'answers', {})), checks=checks,
                    violations=[] if config.observation_profile=='task_progress' else violations,
                    state='EVIDENCE' if config.observation_profile=='task_progress' else 'FAILED' if violations else
                    'UNCERTAIN' if uncertain else
                    'SUSPECTED' if any(c['status'] == 'suspected' for c in checks) else 'RUNNING',
                    subgoal_complete=config.observation_profile=='constraints' and self.completion_count >= 2)


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
    def __init__(self, monitor, config, timeout_s, *, runtime_factory=ConstraintRuntime):
        self.monitor, self.config, self.timeout_s = monitor, config, timeout_s
        self.runtime_factory = runtime_factory
        self.runtime = runtime_factory(config)
        self.frames = deque(maxlen=config.history)
        self.pending = None

    def seed_context(self, frame):
        """Provide a real pre-action view without contributing confirmation votes."""
        if self.pending is not None or self.runtime.last is not None or self.frames:
            raise ValueError('Initial context requires a fresh monitor generation')
        if frame['program_id'] != self.config.program_id:
            raise ValueError('Wrong context program')
        context = deepcopy(frame)
        context['context_only'] = True
        for key in ('executed_actions', 'executed_start_step', 'action_source'):
            context.pop(key, None)
        self.frames.append(context)

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
            runtime = self.runtime_factory(self.config)
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
        self.runtime = self.runtime_factory(self.config)
        self.frames.clear()
