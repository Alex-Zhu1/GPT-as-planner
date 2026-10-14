"""Language planning, real controller boundaries and choice evidence; no paid calls."""
from copy import deepcopy
from threading import Event

import pytest

from .jev.monitor import AsyncMonitor, IDENTITY
from .jev.subgoal_monitor import (ANOMALIES, SubgoalMonitorConfig, SubgoalRuntime,
                                  subgoal_questions, decode_answers)
from .jev.subgoal_plan import (PLAN_VERSION, PLANNER_ROLE, goal_for, task_requirements,
                              validate_planning_response, planning_instructions, planning_context,
                              validate_pi_instruction)
from .jev.subgoals import SubgoalScheduler
from .jev.task_instructions import ORGANIZE_TABLE
from .prompt_context import task_context
from .robodojo_server.client import RoboDojoTools
from .test_contract import Student
from .test_jev import LongSim


def evidence(frame, **overrides):
    labels = dict(observability='usable', **{k:'clear' for k in ANOMALIES})
    labels.update(overrides)
    answers = {}
    for key, question in subgoal_questions(frame).items():
        label = labels[key]
        criteria = question['criteria']
        probabilities = {c: .95 if c == label else .05/(len(criteria)-1) for c in criteria}
        answers[key] = dict(type='choice', probabilities=probabilities)
    return decode_answers(answers, subgoal_questions(frame))


def plan_response(request, mode='plan', completed=None):
    initial = request['review_kind']=='initial_plan'
    if completed is None:
        completed = {r['requirement_id'] for r in request['completed_requirements']}
        if mode=='advance': completed.update(request['remaining_plan'][0]['requirements'])
    assessments = [] if initial else [dict(requirement_id=r['requirement_id'],
        status='satisfied' if r['requirement_id'] in completed else 'pending',
        execution='complete' if r['requirement_id'] in completed else 'progressing',
        evidence='The current observed result supports this assessment.') for r in request['all_requirements']]
    rows = [r for r in request['all_requirements'] if r['requirement_id'] not in completed]
    goals = [goal_for([r]) for r in rows]
    return dict(**{k: request[k] for k in ('request_id', *IDENTITY)},
                decision=dict(mode=mode, reason='Use the observed task stages.', assessments=assessments,
                              subgoals=goals if mode=='plan' else []))


def request():
    rows = task_requirements(ORGANIZE_TABLE, task_context('organize_table'))
    return dict(version=PLAN_VERSION, planner_role=PLANNER_ROLE, review_kind='initial_plan',
        request_id='p', run_id='r', episode_id='e', plan_revision=0, subgoal_id='original-task', program_id='program',
        original_task=ORGANIZE_TABLE, remaining_requirements=rows, all_requirements=rows,
        completed_requirements=[], remaining_plan=[], task_context={}, execution_budget=dict(stage_executed_controls=25))


@pytest.mark.parametrize('fault', ['grouped', 'empty', 'omit', 'duplicate', 'reorder', 'blank_instruction', 'check', 'stale', 'action', 'stop', 'keep_initial'])
def test_invalid_plans_rejected(fault):
    req = request()
    raw = plan_response(req)
    goals = raw['decision']['subgoals']
    if fault == 'grouped': goals[0]['requirements'].append(goals[1]['requirements'][0])
    if fault == 'empty': goals.clear()
    if fault == 'omit': goals.pop()
    if fault == 'duplicate': goals[1] = deepcopy(goals[0])
    if fault == 'reorder': goals[0], goals[1] = goals[1], goals[0]
    if fault == 'blank_instruction': goals[0]['instruction'] = ' '
    if fault == 'check': goals[0]['completion_checks'] = ['Done.']
    if fault == 'stale': raw['plan_revision'] = 1
    if fault == 'action': raw['decision']['steps'] = 25
    if fault == 'stop': raw['decision']['mode'] = 'stop'
    if fault == 'keep_initial': raw = plan_response(req, mode='keep_plan')
    with pytest.raises(ValueError): validate_planning_response(raw, req)


def test_source_coverage_uses_atomic_cards_and_unknown_task_is_retained():
    req = request()
    decision = validate_planning_response(plan_response(req), req)
    assert len(decision['subgoals']) == len(req['all_requirements'])
    assert decision['subgoals'][-1]['instruction'].endswith('Reset both robot arms.')
    assert decision['subgoals'][-1]['requirements'][-1] == 'reset-arm'
    assert task_requirements('An unfamiliar instruction.', {})[0]['instruction'] == 'An unfamiliar instruction.'
    prompt, schema = planning_instructions(req)
    assert 'Stable goal board' in prompt and 'student, edit, EEF' in prompt
    assert schema['properties']['decision']['properties']['subgoals']['maxItems'] == len(req['all_requirements'])


def runtime_frame(step, config, **extra):
    rows = task_requirements(ORGANIZE_TABLE, {})
    return dict(run_id='r', episode_id='e', plan_revision=1, subgoal_id='mouse-on-pad', program_id=config.program_id,
        step_id=step, executed_start_step=step-25, captured_at=100.+step, observation_id=f'o{step}',
        plan_goals=[goal_for([r]) for r in rows], images=[],
        completion_checks=['Both checks'], completed_requirements=[], active_goal_index=0, **extra)


def test_valen_is_only_anomaly_trigger_and_rejects_replayed_evidence():
    config = SubgoalMonitorConfig()
    runtime = SubgoalRuntime(config)
    first = runtime_frame(25, config)
    report = runtime.update(first, evidence(first), now=125.1)
    assert not report['review_requested'] and 'subgoal_complete' not in report
    assert 'completion_votes' not in report['evidence']
    with pytest.raises(ValueError, match='Repeated'):
        runtime.update(first, evidence(first), now=125.2)


def test_explicit_anomaly_questions_remove_overlapping_negative_candidates():
    f = runtime_frame(25, SubgoalMonitorConfig())
    q = subgoal_questions(f)
    assert not {'completion','goal_match','phase','progress'} & set(q)
    assert set(q['placement_error']['criteria']) == {'present','clear','insufficient_evidence'}
    assert f['plan_goals'][0]['instruction'] in q['wrong_object']['instructions']
    raw=evidence(f).answers
    raw['collision']['probabilities']['not_applicable']=0.
    with pytest.raises(ValueError,match='candidates'): decode_answers(raw,q)


@pytest.mark.parametrize('kind,seconds', [('wrong_object',3), ('placement_error',3), ('stalled',3), ('occlusion',6)])
def test_physical_time_windows(kind, seconds):
    config=SubgoalMonitorConfig(); runtime=SubgoalRuntime(config)
    for step in range(25,seconds*25+1,25):
        f=runtime_frame(step,config)
        overrides=dict(observability='occluded') if kind=='occlusion' else {kind:'present'}
        report=runtime.update(f,evidence(f,**overrides),now=100.1+step)
        assert report['review_requested']==(step==seconds*25)
        assert 'subgoal_complete' not in report


def test_diffuse_probabilities_do_not_repeatedly_call_astra():
    config=SubgoalMonitorConfig();runtime=SubgoalRuntime(config)
    for step in range(25,1001,25):
        f=runtime_frame(step,config)
        raw=evidence(f).answers
        for key,answer in raw.items():
            if key!='observability':answer['probabilities']=dict(present=.15,clear=.55,insufficient_evidence=.30)
        report=runtime.update(f,decode_answers(raw,subgoal_questions(f)),now=100.1+step)
        assert not report['review_requested']


def test_stage_generation_and_static_inspection_never_trigger_reviews():
    config=SubgoalMonitorConfig();runtime=SubgoalRuntime(config)
    f=runtime_frame(25,config,inspection_only=True)
    assert not runtime.update(f,evidence(f,collision='present'),now=125.1)['review_requested']
    f=runtime_frame(50,config);f['plan_revision']=2
    with pytest.raises(ValueError,match='generation'):runtime.update(f,evidence(f),now=150.1)


def make_runner(tmp_path, *, horizon=150, planner_fn=None, monitor_fn=None, student=None,
                task='organize_table', instruction=ORGANIZE_TABLE, config=None, **options):
    instructions, calls, windows, events = [], [], [], []
    class Sim(LongSim):
        def request(self, op, **args):
            result = super().request(op, **args)
            if op in ('metadata','teacher_observation'):
                result['instruction'] = instruction
            if op == 'metadata': result['task'] = task
            if op == 'reset': result['metadata'] = dict(evaluation_case=dict(case_id='fixed'))
            return result
    Sim.horizon = horizon
    class TrackingStudent(Student):
        def infer(self, obs, output):
            instructions.append(obs['instruction'])
            return super().infer(obs, output)
    class Monitor:
        def evaluate(self, frames):
            windows.append(deepcopy(frames))
            return monitor_fn(frames) if monitor_fn else evidence(frames[-1])
    class Planner:
        def replan(self, req):
            calls.append(deepcopy(req))
            step = rollout.tick
            raw = planner_fn(req, rollout) if planner_fn else plan_response(req) if req['review_kind']=='initial_plan' else plan_response(req,mode='advance')
            assert rollout.tick == step
            return raw
    rollout = RoboDojoTools(tmp_path, task, student or TrackingStudent(),
                           rpc_factory=Sim, max_decisions=0, student_step_limit=25)
    config = config or SubgoalMonitorConfig()
    monitor = AsyncMonitor(Monitor(), config, 2., runtime_factory=SubgoalRuntime)
    runner = SubgoalScheduler(rollout, monitor, Planner(), chunk_steps=25, max_replans=options.pop('max_replans',12),
                              policy_timeout=2., planner_timeout=options.pop('planner_timeout',2.), emit=events.append, **options)
    return runner, rollout, instructions, calls, windows, events


def fast_review_config():
    return SubgoalMonitorConfig(min_evidence_seconds=1,uncertainty_seconds=2,
        review_seconds=2,review_cooldown_seconds=1)


def test_astra_advances_atomic_goals_with_clear_valen_without_original_task_append(tmp_path):
    runner,rollout,instructions,calls,windows,events=make_runner(tmp_path,horizon=300,config=fast_review_config())
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert rollout.sim.calls.count('reset')==1 and len(calls)==6
    goals=result['plan_goals']
    executed=[e['instruction'] for e in events if e['kind']=='executed_chunk']
    assert executed==[g['instruction'] for g in goals[:-1] for _ in range(2)]+[goals[-1]['instruction']]*4
    completed=[e for e in events if e['kind']=='subgoal_completed']
    assert len(completed)==5 and all(e['decision_author']=='astra' for e in completed)
    assert result['final_stage_confirmed'] and len(result['remaining_plan'])==1
    assert all(len({(f['plan_revision'],f['subgoal_id']) for f in window})==1 for window in windows)
    assert all(h['source_detail']=='student' for h in rollout.history)


def test_initial_pi_and_planner_overlap_without_physics(tmp_path):
    pi_started, plan_started = Event(), Event()
    class SlowStudent(Student):
        def infer(self, obs, output):
            pi_started.set()
            assert plan_started.wait(1)
            return super().infer(obs, output)
    def planner(req, rollout):
        assert pi_started.wait(1)
        plan_started.set()
        assert rollout.tick == 0 and not rollout.history
        return plan_response(req)
    runner, _, _, calls, _, _ = make_runner(tmp_path, student=SlowStudent(), planner_fn=planner)
    result=runner.run()
    assert result['state']=='COMPLETED', result['reason']
    assert calls[0]['review_kind']=='initial_plan' and calls[0]['current_scene']['step_id']==0


def test_off_stage_behavior_replans_fresh_scene_instead_of_skipping(tmp_path):
    def monitor(frames):
        return evidence(frames[-1], wrong_object='present' if frames[-1]['plan_revision']==1 else 'clear')
    def planner(req, rollout):
        if req['review_kind']=='initial_plan': return plan_response(req)
        assert rollout.tick==75 and req['current_scene']['step_id']==75
        assert req['current_scene']['observation_id'] != req['monitor_evidence']['observation_id']
        assert req['review_reason']=='subgoal_anomaly'
        return plan_response(req)
    runner, rollout, _, calls, _, events = make_runner(tmp_path, planner_fn=planner, monitor_fn=monitor)
    result=runner.run()
    assert result['state']=='COMPLETED', result['reason']
    assert len(calls)==2 and not any(e['kind']=='subgoal_completed' for e in events)
    assert rollout.sim.calls.count('reset')==1


def test_astra_reopens_damaged_previously_completed_work(tmp_path):
    def monitor(frames):
        f=frames[-1]
        return evidence(f,regression='present') if f['subgoal_id']=='keyboard-in-frame' else evidence(f)
    def planner(req,rollout):
        if req['review_kind']=='initial_plan':return plan_response(req)
        if rollout.tick==50:return plan_response(req,mode='advance')
        assert rollout.tick==75 and len(req['completed_requirements'])==1
        return plan_response(req,completed=set())
    runner,_,_,calls,_,events=make_runner(tmp_path,horizon=100,config=fast_review_config(),planner_fn=planner,monitor_fn=monitor)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert len(calls)==3 and not result['completed_requirements']
    assert any(e['kind']=='requirements_reopened' for e in events)


def test_periodic_reviews_are_sparse_even_with_low_confidence_valen(tmp_path):
    def planner(req,rollout):
        return plan_response(req) if req['review_kind']=='initial_plan' else plan_response(req,mode='keep_plan')
    runner,_,_,calls,_,_=make_runner(tmp_path,horizon=1000,planner_fn=planner)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert [r['current_scene']['step_id'] for r in calls]==[0,300,600,900]


def test_repeated_anomalies_obey_cooldown(tmp_path):
    runner,_,_,calls,_,events=make_runner(tmp_path,horizon=400,
        planner_fn=lambda req,r:plan_response(req) if req['review_kind']=='initial_plan' else plan_response(req,mode='keep_plan'),
        monitor_fn=lambda frames:evidence(frames[-1],wrong_object='present'))
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert [r['current_scene']['step_id'] for r in calls]==[0,75,225,375]
    assert any(e['kind']=='review_deferred' for e in events)


@pytest.mark.parametrize('fault', ['malformed','stale','stop','timeout'])
def test_initial_plan_failure_never_executes_original_proposal(tmp_path,fault):
    release=Event()
    def planner(req, rollout):
        if fault=='timeout': release.wait(1)
        raw=plan_response(req)
        if fault=='malformed': raw['decision']['subgoals'].clear()
        if fault=='stale': raw['request_id']='old'
        if fault=='stop': raw['decision']['mode']='stop'
        return raw
    runner,rollout,_,_,_,_=make_runner(tmp_path,planner_fn=planner,planner_timeout=.1 if fault=='timeout' else 2.)
    try:
        result=runner.run()
        assert result['state']=='STOPPED'
        assert rollout.tick==0 and not rollout.history
        assert not result['native_result']['complete']
    finally: release.set()


def test_exhausted_budget_preserves_incomplete_result(tmp_path):
    runner, rollout, _, calls, _, _=make_runner(tmp_path,max_replans=1,
        monitor_fn=lambda frames:evidence(frames[-1], collision='present'))
    result=runner.run()
    assert result['state']=='STOPPED' and 'budget exhausted' in result['reason']
    assert len(calls)==1 and rollout.tick==25 and not result['native_result']['complete']


def test_exact_original_instruction_reuses_initial_inference(tmp_path):
    runner, rollout, instructions, _, _, events=make_runner(tmp_path,task='make_kong',
        instruction='An unfamiliar complete task.',horizon=50,
        planner_fn=lambda req, rollout:plan_response(req),monitor_fn=lambda frames:evidence(frames[-1]))
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert len(instructions)==len(rollout.history)==2
    assert next(e for e in events if e['kind']=='subgoal_plan_response')['reused_initial_inference']


def test_monitor_error_stops_before_next_chunk(tmp_path):
    def monitor(frames):raise ValueError('Malformed choice probabilities')
    runner,rollout,_,_,_,_=make_runner(tmp_path,monitor_fn=monitor)
    result=runner.run()
    assert result['state']=='PAUSED_MONITOR_UNAVAILABLE'
    assert rollout.tick==25 and not result['native_result']['complete']


def test_native_failure_is_not_visual_success(tmp_path):
    runner,rollout,_,_,_,_=make_runner(tmp_path,horizon=300,config=fast_review_config())
    class NativeFailure(rollout.rpc_factory):
        def request(self,op,**args):
            result=super().request(op,**args)
            if op=='chunk_step':
                for row in result['steps']:row['success']=False
            if op=='finish_pilot':result['success']=False
            return result
    rollout.rpc_factory=NativeFailure
    result=runner.run()
    assert result['state']=='COMPLETED' and result['native_result']['complete']
    assert result['native_result']['success'] is False
    assert result['final_stage_confirmed']


def test_persistent_collision_does_not_bypass_review_cooldown(tmp_path):
    runner,_,_,calls,_,_=make_runner(tmp_path,horizon=225,
        planner_fn=lambda req,r:plan_response(req) if req['review_kind']=='initial_plan' else plan_response(req,mode='keep_plan'),
        monitor_fn=lambda frames:evidence(frames[-1],collision='present'))
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert [r['current_scene']['step_id'] for r in calls]==[0,25,175]


def test_new_collision_after_clear_evidence_bypasses_cooldown(tmp_path):
    runner,_,_,calls,_,_=make_runner(tmp_path,horizon=100,
        planner_fn=lambda req,r:plan_response(req) if req['review_kind']=='initial_plan' else plan_response(req,mode='keep_plan'),
        monitor_fn=lambda frames:evidence(frames[-1],collision='clear' if frames[-1]['step_id']==50 else 'present'))
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert [r['current_scene']['step_id'] for r in calls]==[0,25,75]


def test_fresh_astra_assessment_revokes_final_stage_confirmation(tmp_path):
    def planner(req,rollout):
        if req['review_kind']=='initial_plan':return plan_response(req)
        return plan_response(req,mode='advance' if rollout.tick==50 else 'plan',
            completed={r['requirement_id'] for r in req['all_requirements']} if rollout.tick==50 else set())
    runner,_,_,_,_,_=make_runner(tmp_path,horizon=125,config=fast_review_config(),planner_fn=planner)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert not result['final_stage_confirmed'] and not result['completed_requirements']


def test_slow_planner_revalidates_scene_without_reusing_old_initial_inference(tmp_path,monkeypatch):
    runner,rollout,instructions,_,_,events=make_runner(tmp_path,task='make_kong',
        instruction='An unfamiliar complete task.',horizon=25,
        planner_fn=lambda req,rollout:plan_response(req))
    validate=runner.validate_recovery_scene
    def expired_scene(scene):
        # Model elapsed wall time without a brittle millisecond I/O deadline.
        expired=deepcopy(scene)
        expired['captured_at']-=runner.monitor.config.max_age_s+1
        validate(expired)
    monkeypatch.setattr(runner,'validate_recovery_scene',expired_scene)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert len(instructions)==2
    assert any(e['kind']=='planner_scene_revalidated' for e in events)
    assert not next(e for e in events if e['kind']=='subgoal_plan_response')['reused_initial_inference']


def test_actual_organize_table_contract_and_visual_repair_language():
    context=planning_context('organize_table',ORGANIZE_TABLE,task_context('organize_table'))
    rows=task_requirements(ORGANIZE_TABLE,context)
    assert [r['requirement_id'] for r in rows]==['mouse-on-pad','keyboard-in-frame','figurine-on-stand','clock-on-drawer','reset-arm']
    assert 'upright' in rows[2]['completion_check'] and 'static drawer' in context['instruction_discrepancy']
    req=request();req.update(all_requirements=rows,remaining_requirements=rows,task_context=context)
    raw=plan_response(req);raw['decision']['subgoals'][2]['instruction']='Pick up the fallen figurine and put it upright on the stand.'
    validate_planning_response(raw,req)


@pytest.mark.parametrize('fault',['pending_current','missing_assessment','empty_evidence','no_execution','early_arm_return','stale_version'])
def test_astra_cannot_advance_without_complete_current_evidence(fault):
    req=request();req.update(review_kind='progress_review',remaining_plan=plan_response(req)['decision']['subgoals'])
    raw=plan_response(req,mode='advance')
    if fault=='pending_current':raw['decision']['assessments'][0]['status']='pending'
    if fault=='missing_assessment':raw['decision']['assessments'].pop()
    if fault=='empty_evidence':raw['decision']['assessments'][0]['evidence']=' '
    if fault=='no_execution':req['execution_budget']['stage_executed_controls']=0
    if fault=='early_arm_return':raw['decision']['assessments'][-1]['status']='satisfied'
    if fault=='stale_version':req['version']=3
    with pytest.raises(ValueError):validate_planning_response(raw,req)


def test_reopened_requirement_cannot_be_left_out_of_queue():
    req=request();goals=plan_response(req)['decision']['subgoals']
    req.update(review_kind='subgoal_review',remaining_plan=goals[1:],completed_requirements=req['all_requirements'][:2])
    with pytest.raises(ValueError,match='reopened'):
        validate_planning_response(plan_response(req,mode='keep_plan',completed=set()),req)


@pytest.mark.parametrize('instruction', [
    'Pick up the fallen figurine and put it upright on the stand. '
    'Release it so it remains stable and supported, without disturbing the clock on the cabinet.',
    'Grasp the green body of the fallen figurine. Lift and rotate it so the yellow base faces down, '
    'then place the base flat on the white stand.',
    'Place ' + 'the ' * 24 + 'figurine on the stand.',
    'Place the ' + 'x' * 160 + ' on the stand.',
])
def test_pi_instruction_rejects_verbose_or_verification_suffixes(instruction):
    with pytest.raises(ValueError, match='Pi instruction'):
        validate_pi_instruction(instruction, 1)


@pytest.mark.parametrize('instruction,count', [
    ('Pick up the fallen figurine and put it upright on the stand.', 1),
    ('Lift the fallen figurine onto the stand with its base down.', 1),
    ('Release the figurine.', 1),
    ('Reset both robot arms.', 1),
])
def test_short_atomic_actions_remain_valid(instruction,count):
    validate_pi_instruction(instruction,count)


def test_periodic_review_can_reword_failed_grasp_and_retains_execution_history(tmp_path):
    replacement = 'Lift the fallen figurine onto the stand with its base down.'
    reason = 'Two observed grasps missed the figurine; try a short lift-and-place description.'
    def planner(req, rollout):
        history = req['instruction_history']
        if req['review_kind'] == 'initial_plan':
            assert history == []
            return plan_response(req)
        if rollout.tick == 50:
            assert history[-1]['executed_controls'] == 50
            return plan_response(req, mode='advance', completed={'mouse-on-pad','keyboard-in-frame','clock-on-drawer'})
        if rollout.tick == 100:
            assert req['review_kind'] == 'progress_review'
            assert history[-1]['instruction'] == 'Put the figurine upright on the stand.'
            assert history[-1]['executed_controls'] == 50
            raw = plan_response(req)
            raw['decision']['subgoals'][0]['instruction'] = replacement
            raw['decision']['reason'] = reason
            raw['decision']['assessments'][2]['execution'] = 'ineffective'
            raw['decision']['assessments'][2]['evidence'] = 'Both attempted grasps missed; the figurine remains fallen.'
            return raw
        assert rollout.tick == 150
        assert [h['executed_controls'] for h in history] == [50,50,50]
        assert history[-1]['instruction'] == replacement and history[-1]['reason'] == reason
        return plan_response(req,mode='keep_plan')
    runner,rollout,_,calls,_,events=make_runner(tmp_path,task='organize_table',horizon=200,
        planner_fn=planner,config=fast_review_config())
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert rollout.sim.calls.count('reset')==1
    executed=[e['instruction'] for e in events if e['kind']=='executed_chunk']
    assert executed[2:4]==['Put the figurine upright on the stand.']*2
    assert executed[4:]==[replacement]*4
    rewrites=[e for e in events if e['kind']=='subgoal_instruction_rewritten']
    assert len(rewrites)==1 and rewrites[0]['previous_executed_controls']==50
    assert rewrites[0]['reason']==reason
    assert [h['executed_controls'] for h in result['instruction_history']]==[50,50,100]
    assert len(result['completed_requirements'])==3
    assert result['remaining_plan'][0]['completion_checks']==[runner.requirements[2]['completion_check']]
    assert all(c['pi_instruction_contract']['max_words']==24 for c in calls)


def test_invalid_pi_suffix_never_reaches_execution(tmp_path):
    def planner(req, rollout):
        raw=plan_response(req)
        raw['decision']['subgoals'][0]['instruction']='Place the figurine upright without disturbing the clock.'
        return raw
    runner,rollout,_,_,_,_=make_runner(tmp_path,planner_fn=planner)
    result=runner.run()
    assert result['state']=='STOPPED' and 'instruction_options' in result['reason']
    assert rollout.tick==0 and not rollout.history


@pytest.mark.parametrize('suffix', [' while leaving the clock untouched.', ' and make sure it stays supported.'])
def test_short_paraphrases_cannot_bypass_exact_goal_language(suffix):
    req=request();raw=plan_response(req)
    raw['decision']['subgoals'][2]['instruction']='Stand the figurine upright'+suffix
    with pytest.raises(ValueError,match='instruction_options'):
        validate_planning_response(raw,req)


def test_grouping_cannot_multiply_absolute_instruction_limits():
    with pytest.raises(ValueError,match='atomic'):
        validate_pi_instruction('Place the mouse. Place the keyboard.',2)


@pytest.mark.parametrize('mode', ['keep_plan','plan'])
def test_confirmed_ineffective_execution_requires_changed_description(mode):
    req=request();req.update(review_kind='progress_review',remaining_plan=plan_response(req)['decision']['subgoals'])
    raw=plan_response(req,mode=mode)
    raw['decision']['assessments'][0]['execution']='ineffective'
    with pytest.raises(ValueError,match='[Cc]hange|Ineffective'):
        validate_planning_response(raw,req)
    raw=plan_response(req)
    raw['decision']['assessments'][0]['execution']='ineffective'
    raw['decision']['subgoals'][0]['instruction']='Move the mouse onto the mouse pad.'
    validate_planning_response(raw,req)


def independent_review():
    req = request()
    req['task_context'] = planning_context('organize_table', ORGANIZE_TABLE, task_context('organize_table'))
    req['all_requirements'] = task_requirements(ORGANIZE_TABLE, req['task_context'])
    req['remaining_plan'] = plan_response(req)['decision']['subgoals']
    req['review_kind'] = 'progress_review'
    req['instruction_history'] = [dict(requirements=['mouse-on-pad'], instruction=req['remaining_plan'][0]['instruction'],
        executed_controls=100, start_step=0, end_step=100)]
    return req


@pytest.mark.parametrize('mode', ['keep_plan', 'plan'])
def test_ineffective_independent_goal_yields_even_with_new_wording(mode):
    req = independent_review()
    raw = plan_response(req, mode=mode)
    raw['decision']['assessments'][0]['execution'] = 'ineffective'
    if mode == 'plan':
        raw['decision']['subgoals'][0]['instruction'] = 'Move the mouse onto the mouse pad.'
    with pytest.raises(ValueError, match='must yield'):
        validate_planning_response(raw, req)


def test_deferring_failure_keeps_pending_card_and_untried_wording():
    req = independent_review()
    raw = plan_response(req)
    raw['decision']['assessments'][0]['execution'] = 'ineffective'
    goals = raw['decision']['subgoals']
    goals.insert(0, goals.pop(2))  # Figurine selected, mouse remains unfinished.
    validate_planning_response(raw, req)
    assert goals[1] == req['remaining_plan'][0]
    # Incidental keyboard failure does not force rewriting its never-executed text.
    raw['decision']['assessments'][1]['execution'] = 'ineffective'
    validate_planning_response(raw, req)


def test_unserved_figurine_cannot_be_starved_by_mouse_keyboard_alternation():
    req = independent_review()
    req['remaining_plan'].insert(0, req['remaining_plan'].pop(1))
    req['instruction_history'].append(dict(requirements=['keyboard-in-frame'],
        instruction=req['remaining_plan'][0]['instruction'], executed_controls=100, start_step=100, end_step=200))
    raw = plan_response(req, completed={'clock-on-drawer'})
    raw['decision']['assessments'][1]['execution'] = 'ineffective'
    with pytest.raises(ValueError, match='figurine-on-stand'):
        validate_planning_response(raw, req)  # Mouse is not the eligible alternative.
    goals = raw['decision']['subgoals']
    goals.insert(0, goals.pop(2))
    validate_planning_response(raw, req)


def test_all_served_yields_to_oldest_alternative():
    req = independent_review()
    for i, goal in enumerate(req['remaining_plan'][1:4], 1):
        req['instruction_history'].append(dict(requirements=goal['requirements'], instruction=goal['instruction'],
            executed_controls=100, start_step=i*100, end_step=(i+1)*100))
    raw = plan_response(req)
    raw['decision']['assessments'][0]['execution'] = 'ineffective'
    goals = raw['decision']['subgoals']
    goals.insert(0, goals.pop(2))
    with pytest.raises(ValueError, match='keyboard-in-frame'):
        validate_planning_response(raw, req)
    goals.insert(0, goals.pop(2))
    validate_planning_response(raw, req)


def test_resume_rejects_last_failed_wording_even_if_latest_assessment_is_uncertain():
    req = independent_review()
    req['instruction_history'][0]['reviews'] = [dict(execution='ineffective')]
    req['remaining_plan'].insert(0, req['remaining_plan'].pop(2))
    raw = plan_response(req)
    raw['decision']['assessments'][0].update(status='uncertain', execution='uncertain')
    with pytest.raises(ValueError, match='before retrying'):
        validate_planning_response(raw, req)
    raw['decision']['subgoals'][0]['instruction'] = 'Move the mouse onto the mouse pad.'
    validate_planning_response(raw, req)


def test_scheduler_defers_then_resumes_without_completion_or_reset(tmp_path):
    def planner(req, rollout):
        raw = plan_response(req)
        if rollout.tick == 0:
            return raw
        if rollout.tick == 50:
            raw['decision']['assessments'][0]['execution'] = 'ineffective'
            raw['decision']['subgoals'].insert(0, raw['decision']['subgoals'].pop(2))
        elif rollout.tick == 100:
            mouse = req['goal_board'][0]
            assert mouse['deferred'] and mouse['status'] == 'pending'
            assert mouse['executed_controls'] == mouse['waiting_controls'] == 50
            assert mouse['last_executed_step'] == 50
            raw['decision']['subgoals'][0]['instruction'] = 'Move the mouse onto the mouse pad.'
        else:
            return plan_response(req, mode='keep_plan')
        return raw
    runner, rollout, _, calls, _, events = make_runner(tmp_path, horizon=200,
        planner_fn=planner, config=fast_review_config())
    result = runner.run()
    assert result['state'] == 'COMPLETED', result['reason']
    assert rollout.sim.calls.count('reset') == 1
    assert result['completed_requirements'] == []
    assert [h['requirements'] for h in result['instruction_history']] == [
        ['mouse-on-pad'], ['figurine-on-stand'], ['mouse-on-pad']]
    assert [h['executed_controls'] for h in result['instruction_history']] == [50, 50, 100]
    assert sum(e['kind'] == 'subgoal_deferred' for e in events) == 2
    assert sum(e['kind'] == 'subgoal_resumed' for e in events) == 1
    rewrites = [e for e in events if e['kind'] == 'subgoal_instruction_rewritten']
    assert len(rewrites) == 1 and rewrites[0]['previous_executed_controls'] == 50
    assert all(c['dispatch_policy']['version'] == 1 for c in calls)


def test_completed_future_card_pruned_and_preserved_in_board(tmp_path):
    def planner(req,rollout):
        if rollout.tick==0:return plan_response(req)
        return plan_response(req,mode='keep_plan',completed={'keyboard-in-frame'})
    runner,_,_,calls,_,_=make_runner(tmp_path,horizon=125,config=fast_review_config(),planner_fn=planner)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert 'keyboard-in-frame' not in [g['subgoal_id'] for g in result['remaining_plan']]
    card=next(c for c in result['goal_board'] if c['requirement_id']=='keyboard-in-frame')
    assert card['status']=='satisfied' and not card['active'] and card['executed_controls']==0
    assert len(result['goal_board'])==5
    assert calls[-1]['goal_board'][1]['status']=='satisfied'


def test_bounded_format_correction_preserves_paused_scene_and_commits_once(tmp_path):
    def planner(req,rollout):
        raw=plan_response(req)
        if 'format_repair' not in req:
            raw['decision']['subgoals'][0]['instruction']='Place the mouse while leaving the clock untouched.'
        else:
            assert req['format_repair']['correction_number']==1
            assert 'instruction_options' in req['format_repair']['error']
            assert not rollout.history and rollout.tick==0
            assert not req['instruction_history']
            assert all(c['execution']=='not_attempted' for c in req['goal_board'])
        return raw
    runner,rollout,_,calls,_,events=make_runner(tmp_path,horizon=25,planner_fn=planner)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert result['replan_count']==2 and len(result['format_repairs'])==1
    assert len(result['attempts'])==1 and result['plan_revision']==1
    assert len({r['request_id'] for r in calls})==2
    assert calls[0]['current_scene']==calls[1]['current_scene']
    assert calls[1]['format_repair']['root_request_id']==calls[0]['request_id']
    assert len(rollout.history)==1
    assert all(e['controls_during_planning']==0 for e in events if e['kind']=='subgoal_plan_rejected')


@pytest.mark.parametrize('budget,expected', [(12,3),(2,2)])
def test_format_correction_exhaustion_never_executes(budget,expected,tmp_path):
    def planner(req,rollout):
        raw=plan_response(req);raw['decision']['subgoals'].clear();return raw
    runner,rollout,_,calls,_,_=make_runner(tmp_path,planner_fn=planner,max_replans=budget)
    result=runner.run()
    assert result['state']=='STOPPED' and not result['native_result']['complete']
    assert len(calls)==expected and not rollout.history and rollout.tick==0
    assert not result['instruction_history'] and not result['attempts']


@pytest.mark.parametrize('error', [TimeoutError('transport timeout'),RuntimeError('transport unavailable')])
def test_transport_failure_does_not_consume_format_retries(error,tmp_path):
    def planner(req,rollout):raise error
    runner,rollout,_,calls,_,_=make_runner(tmp_path,planner_fn=planner)
    result=runner.run()
    assert result['state']=='STOPPED' and len(calls)==1 and not result['format_repairs']
    assert not rollout.history


def test_review_evidence_spans_interval_instead_of_last_two_seconds(tmp_path):
    from .jev.subgoal_plan import planning_packet
    def planner(req,rollout):
        if rollout.tick==0:return plan_response(req)
        frames=req['evidence_frames']
        assert len(frames)==8
        assert frames[0]['step_id']==25 and frames[-1]['step_id']==300
        assert req['evidence_coverage']['endpoint_count']==12
        assert req['evidence_coverage']['selected_steps']==[25,75,100,150,175,225,250,300]
        polluted=deepcopy(req);polluted['evidence_frames'][0]['native_success']='SECRET'
        packet,_=planning_packet(polluted)
        import json
        assert 'SECRET' not in json.dumps(packet)
        assert len({s['observation_id'] for s in packet['scenes']})==len(packet['scenes'])
        return plan_response(req,mode='keep_plan')
    runner,_,_,calls,_,_=make_runner(tmp_path,horizon=325,planner_fn=planner)
    result=runner.run()
    assert result['state']=='COMPLETED',result['reason']
    assert len(calls)==2
