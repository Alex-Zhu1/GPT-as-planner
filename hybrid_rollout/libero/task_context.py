"""Append-only evidence board; semantic interpretations belong to Astra only."""
from collections import deque
from copy import deepcopy

from ..robodojo.jev.monitor import IDENTITY, TASK_QUESTIONS


def board_evidence_ids(request):
    frames = [request['current_scene'], *request.get('evidence_frames', []),
              *request.get('task_context', {}).get('evidence_frames', [])]
    frames += list(request.get('task_context', {}).get('keyframes', {}).values())
    frames += request.get('failure', {}).get('evidence', [])
    frames += [row['evidence'] for row in request.get('task_context', {}).get('recent_history', [])]
    return sorted({f['observation_id'] for f in frames if 'observation_id' in f})


def board_assessment_schema(request):
    from ..robodojo.skill.schema import _obj
    text = {'type':'string', 'minLength':1, 'maxLength':4000}
    return _obj(dict(robot_status=text, task_status={'type':'string',
        'enum':['pending','in_progress','completed','uncertain']}, task_reason=text, scene_context=text,
        uncertainties=text, evidence_observation_ids={'type':'array', 'minItems':1, 'maxItems':12,
            'items':{'type':'string','enum':board_evidence_ids(request)}}))


def validate_board_assessment(value, request):
    fields = board_assessment_schema(request)['properties']
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError('Astra board assessment required')
    for key in ('robot_status','task_reason','scene_context','uncertainties'):
        if not isinstance(value[key],str) or not value[key].strip() or len(value[key])>4000:
            raise ValueError('Astra board assessment requires bounded text')
    if value['task_status'] not in fields['task_status']['enum']:
        raise ValueError('Invalid Astra task status')
    refs=value['evidence_observation_ids']
    if (not isinstance(refs,list) or not 1<=len(refs)<=12
            or any(not isinstance(ref,str) or ref not in board_evidence_ids(request) for ref in refs)):
        raise ValueError('Astra board assessment must cite supplied evidence')


class TaskContext:
    def __init__(self, config, *, stall_steps=30, min_reports=4, max_sample_gap=None,
                 ambiguity_steps=30, ambiguity_review_interval=80):
        if type(stall_steps) is not int or stall_steps<1 or type(min_reports) is not int or not 2<=min_reports<=6:
            raise ValueError('Progress review requires positive span and 2–6 minimum candidate reports')
        self.config, self.stall_steps, self.min_reports = config, stall_steps, min_reports
        if any(type(v) is not int or v < 1 for v in (ambiguity_steps, ambiguity_review_interval)):
            raise ValueError('Observability review requires positive executed-step intervals')
        self.ambiguity_steps = ambiguity_steps
        self.ambiguity_review_interval = ambiguity_review_interval
        self.max_sample_gap = config.max_step_gap if max_sample_gap is None else max_sample_gap
        self.last_capture = None
        self.keyframes = {}
        self.largest_pixel_change = 0.
        self.episode = None
        self.last_step = self.last_advancement_signal = self.last_review = None
        self.recent, self.milestones, self.plans = deque(maxlen=12), deque(maxlen=32), deque(maxlen=32)
        self.pending_evidence = []  # All observations since the last review, even beyond recent_history.
        self.astra_reviews = deque(maxlen=16)
        self.votes = deque(maxlen=6)
        self.valid_votes = deque()  # Bounded by executed-control span, not sample count.
        self.progress_streak = 0
        self.robot = {}
        self.first_frame = self.progress_frame = self.latest_frame = None

    def evidence(self, frame):
        return {k:deepcopy(frame[k]) for k in (*IDENTITY,'step_id','observation_id','captured_at','images')}

    def activate(self, frame, goals):
        episode=(frame['run_id'],frame['episode_id'])
        if self.episode is not None and episode!=self.episode:
            raise ValueError('Task memory cannot cross episodes')
        self.episode=episode
        self.original_task=frame['task']
        self.active_identity=tuple(frame[k] for k in IDENTITY)
        self.robot=dict(author='host_measurement',current_eef=deepcopy(frame['current_eef']),
                        current_state=deepcopy(frame['current_state']),evidence=self.evidence(frame))
        self.plans.append(dict(author='host_execution_log',plan_revision=frame['plan_revision'],
            step_id=frame['step_id'],goals=deepcopy(list(goals)),active_instruction=frame['subgoal']))
        self.votes.clear()
        self.valid_votes.clear()
        self.progress_streak = 0
        if self.first_frame is None:
            self.first_frame=self.evidence(frame)

    def update(self, report, frame):
        if (tuple(frame[k] for k in IDENTITY)!=self.active_identity
                or any(report[k]!=frame[k] for k in (*IDENTITY,'step_id','observation_id'))):
            raise ValueError('Task context received unrelated evidence')
        step=report['step_id']
        if report.get('inspection_only') or (self.last_step is not None and step<=self.last_step):
            return False
        # Constraint confirmation gaps stay strict. Advisory sampling tolerates
        # bounded coalescing, but never bridges long step or capture-time gaps.
        if ((self.last_step is not None and step-self.last_step>self.max_sample_gap)
                or (self.last_capture is not None and
                    frame['captured_at']-self.last_capture>self.config.max_gap_s)):
            self.votes.clear()
            self.valid_votes.clear()
            self.progress_streak = 0
        self.last_step=step
        self.last_capture=frame['captured_at']
        ref=self.evidence(frame)
        self.latest_frame=ref
        self.robot=dict(author='host_measurement',current_eef=deepcopy(frame['current_eef']),
                        current_state=deepcopy(frame['current_state']),evidence=ref)
        s=report['scores']
        # Store evidence verbatim, including contradictory/uncertain scores.
        # It is not a task status, scene narrative or proof of object movement.
        self.recent.append(dict(author='valen',evidence=ref,scores=deepcopy(s),
            raw_answers=deepcopy(report.get('raw_answers', {})),
            scope=dict(subgoal_id=frame['subgoal_id'],instruction=frame['subgoal']),
            executed_actions=deepcopy(frame.get('executed_actions',[]))))
        if 'executed_start_step' in frame and 'executed_actions' in frame:
            self.recent[-1]['execution_receipt']=dict(author='host_execution',
                start_tick=frame['executed_start_step'],end_tick=step,
                executed_steps=len(frame['executed_actions']),source_detail=frame.get('action_source'))
        self.pending_evidence.append(deepcopy(self.recent[-1]))
        if not set(TASK_QUESTIONS)<=set(s):
            return True
        if self.last_review is not None and step<=self.last_review:
            return True
        high,low=self.config.high,self.config.low
        raw=report.get('raw_answers', {})
        def probability(key, candidate, fallback):
            answer=raw.get(key)
            return answer['probabilities'][candidate] if answer and answer.get('type')=='choice' else fallback
        comparable=s['target_visible']>=high and s['scene_comparable']>=high
        usable=comparable  # Holding an object on a support surface is not a contradiction.
        progress_unknown=(probability('task_advancing','insufficient_evidence',0.)>=low or
            ('task_advancing' in raw and max(probability('task_advancing',k,0.) for k in
                ('advancing','attempt_no_effect','no_attempt'))<high))
        usable=usable and not progress_unknown
        attempt=max(s[k] for k in ('progress_attempted','arm_approaching','arm_transporting','arm_releasing'))>=high
        no_effect=probability('task_advancing','attempt_no_effect',float(attempt and s['task_advancing']<=low))>=high
        progressing=usable and s['task_advancing']>=high
        self.progress_streak=self.progress_streak+1 if progressing else 0
        if self.progress_streak>=2:
            self.last_advancement_signal=self.progress_frame=ref
            self.milestones.append(dict(author='jev_sampling_policy',kind='sustained_high_advancement_scores',
                                       score=s['task_advancing'],evidence=ref))
            self.valid_votes.clear()
        # These booleans describe threshold crossings, never semantic scene facts.
        row=dict(step_id=step,start_step=frame.get('executed_start_step',step),evidence=ref,
            completion_candidates=[k for k in ('task_satisfied','subgoal_complete')
                                   if comparable and s[k]>=high],
            low_progress_score=usable and s['task_advancing']<=low and (no_effect or
                (attempt and s['stuck']>=high)),
            # Intermediate scores and diffuse progress candidates are unknown,
            # not affirmative evidence that the cameras cannot observe the task.
            # They neither vote for no progress nor request rapid visual review.
            ambiguous_scores=s['target_visible']<=low or s['scene_comparable']<=low,
            anomaly_candidates=[k for k in ('collision','target_mismatch','grasp_failure')
                if s[k]>=high and (k!='grasp_failure' or
                    probability('grasp_failure','failed_attempt',s['grasp_attempted'])>=high)])
        self.votes.append(row)
        if usable:
            self.valid_votes.append(row)
        # Bound the temporal window even when usable samples are sparse.
        while self.valid_votes and step-self.valid_votes[0]['step_id']>3*self.stall_steps:
            self.valid_votes.popleft()
        return True

    def review(self):
        rows=list(self.votes)
        if not rows or (self.last_review is not None and rows[-1]['step_id']<=self.last_review):
            return None
        last_two=rows[-2:]
        anomaly=(set(last_two[0]['anomaly_candidates']) & set(last_two[1]['anomaly_candidates'])
                 if len(last_two)==2 and last_two[1]['step_id']-last_two[0]['step_id']<=self.max_sample_gap else set())
        completion=(set(last_two[0]['completion_candidates']) & set(last_two[1]['completion_candidates'])
                    if len(last_two)==2 and last_two[1]['step_id']-last_two[0]['step_id']<=self.max_sample_gap else set())
        if anomaly:
            rows=last_two
            trigger='anomaly_scores_review'
        elif completion:
            rows=last_two
            trigger='completion_scores_review'
        elif (len(rows)>=3 and all(r['ambiguous_scores'] for r in rows[-3:])
              and rows[-1]['step_id']-(self.last_review if self.last_review is not None
                                       else self.first_frame['step_id']) >=
                  (self.ambiguity_review_interval if self.last_review is not None else self.ambiguity_steps)):
            rows=rows[-3:]
            trigger='ambiguous_observations_review'
        else:
            rows=list(self.valid_votes)
            if len(rows)<6 or rows[-1]['step_id']-rows[0]['start_step']<self.stall_steps:
                return None
            # Shortest suffix covering the required executed span and six samples.
            start=max(i for i in range(len(rows)-5)
                      if rows[-1]['step_id']-rows[i]['start_step']>=self.stall_steps)
            rows=rows[start:]
            rows=[rows[i*(len(rows)-1)//5] for i in range(6)]
            if sum(r['low_progress_score'] for r in rows)<max(4,self.min_reports):
                return None
            trigger='low_progress_scores_review'
        return dict(trigger=trigger,author='jev_sampling_policy',confirmed_failure=False,violations=[],
            first_step=rows[0]['start_step'],last_step=rows[-1]['step_id'],report_count=len(rows),
            low_progress_votes=sum(r['low_progress_score'] for r in rows),
            anomaly_candidates=sorted(anomaly),completion_candidates=sorted(completion), evidence=[r['evidence'] for r in rows],
            last_high_advancement_score=deepcopy(self.last_advancement_signal),
            reason='Numeric evidence-window threshold reached; Astra must interpret the board. No task-status conclusion.')

    def reviewed(self, step):
        self.pending_evidence.clear()
        self.last_review=step
        self.votes.clear()
        self.valid_votes.clear()
        self.progress_streak=0

    def completed_subgoal(self, goal, frame):
        self.milestones.append(dict(author='host_execution_log',kind='scheduler_advanced_subgoal',
            goal=deepcopy(goal),evidence=self.evidence(frame),semantic_completion_not_asserted=True))

    def record_astra(self, review):
        if review is not None:
            self.astra_reviews.append(deepcopy(review))

    def remember(self, role, frame):
        if role not in ('visual_change_before','visual_change_after','last_decision','last_decision_after','eef_before','eef_after'):
            raise ValueError('Unknown keyframe role')
        if (frame['run_id'],frame['episode_id']) != self.episode:
            raise ValueError('Keyframe cannot cross episodes')
        if role=='last_decision':
            self.keyframes.pop('last_decision_after',None)
        self.keyframes[role] = self.evidence(frame)

    def record_execution_images(self, before, after, mode):
        from .visual_evidence import pixel_change
        if mode == 'eef':
            self.remember('eef_before',before)
            self.remember('eef_after',after)
        decision=self.keyframes.get('last_decision')
        if decision and before['step_id']==decision['step_id']:
            self.remember('last_decision_after',after)
        score = pixel_change(before,after)
        if score > self.largest_pixel_change:
            self.largest_pixel_change = score
            self.remember('visual_change_before',before)
            self.remember('visual_change_after',after)

    def snapshot(self):
        frames=[]
        for frame in (self.first_frame,self.progress_frame,self.latest_frame):
            if frame and not any(f['observation_id']==frame['observation_id'] for f in frames):
                frames.append(deepcopy(frame))
        return deepcopy(dict(version=2,robot=self.robot,
            keyframes=self.keyframes, keyframe_selection=dict(author='jev_sampling_policy',
                metric='largest_mean_thumbnail_pixel_difference',score=self.largest_pixel_change,
                semantic_change_not_asserted=True),
            task=dict(original_instruction=getattr(self,'original_task',None)),
            scheduling=dict(author='jev_sampling_policy',last_high_advancement_score=self.last_advancement_signal,
                            last_review_step=self.last_review,
                            policy=dict(valid_window=6,minimum_low_progress_votes=max(4,self.min_reports),
                                        minimum_control_span=self.stall_steps,ambiguity_reports=3,
                                        ambiguity_first_minimum_steps=self.ambiguity_steps,
                                        ambiguity_repeat_minimum_steps=self.ambiguity_review_interval,
                                        ambiguity_evidence='target_visible_or_scene_comparable_at_or_below_low',
                                        progress_unknown_policy='record_only_no_stall_vote_execution_budget_fallback',
                                        anomaly_reports=2,completion_reports=2,progress_reset_reports=2)),
            new_evidence=list(self.pending_evidence),
            astra_interpretations=list(self.astra_reviews),
            plan_history=list(self.plans),milestones=list(self.milestones),recent_history=list(self.recent),
            evidence_frames=frames,latest_source_step=self.last_step,
            interpretation_policy='Only Astra supplies robot/task/scene interpretations. Valen rows are raw evidence. '
                'Host fields are measurements, execution records and numeric scheduling counters. Astra interpretations '
                'retain their original source step and may be stale; they never set native task success.'))
