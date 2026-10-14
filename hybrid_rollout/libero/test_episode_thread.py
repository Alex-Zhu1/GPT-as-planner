"""Offline RPC contract tests: no Codex inference, GPU, or simulator required."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
from .test_planner import request
from .planner_server import AstraPlanner
from ..robodojo.jev.runner import automatic_response


class FakeTransport:
    def __init__(self, argv, folder, payload=None):
        self.calls=[];self.events=[];self.closed=False;self.payload=payload;self.turns=0;self.starts=0
    def request(self, method, params, timeout):
        self.calls.append((method,deepcopy(params)))
        if method=='initialize':return {}
        if method in ('thread/start','thread/resume'):
            self.starts+=1
            return dict(thread={'id':params.get('threadId',f'thread-{self.starts}')},
                        model='gpt-6-astra',reasoningEffort='xhigh')
        if method=='turn/start':
            self.turns+=1;turn=f'turn-{self.turns}';tid=params['threadId']
            packet=json.loads(params['input'][0]['text'])
            decision=automatic_response(packet['native_request'],1,{})
            decision.pop('evaluation_method');decision.pop('decision_author')
            payload=self.payload or {'decision':decision}
            self.events=[dict(method='item/completed',params=dict(threadId=tid,turnId=turn,
                         item=dict(type='agentMessage',phase='final_answer',text=json.dumps(payload)))),
                         dict(method='turn/completed',params=dict(threadId=tid,turn=dict(id=turn,status='completed')))]
            return {'turn':{'id':turn}}
        raise AssertionError(method)
    def notify(self,*args):pass
    def reply(self,*args):self.calls.append(('reply',args))
    def next_message(self,timeout):return self.events.pop(0)
    def close(self):self.closed=True


def setup(tmp_path):
    made=[]
    def factory(*args):
        t=FakeTransport(*args);made.append(t);return t
    return AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory),made


def next_request(r, step):
    r=deepcopy(r);r['request_id']=f'outer-{step}'
    r['native_request']['request_id']=f'proposal-{step}';r['native_request']['step_id']=step
    r['current_scene'].update(step_id=step,observation_id=f'obs-{step}')
    return r


def test_same_episode_new_turn_dedup_history_and_separate_episode(tmp_path):
    planner,made=setup(tmp_path);r=next_request(request(tmp_path),1)
    r['native_request']['gate_instruction']='FIXED-GATE-MUST-NOT-REPEAT'
    row=dict(evidence=dict(observation_id='old',step_id=0,images=[]),scores={'collision':.1},raw_answers={})
    r['task_context']=dict(recent_history=[row],astra_interpretations=[{'assessment':'PAST-SEMANTICS'}])
    first=planner.replan(r)
    r2=next_request(r,6);r2['previous_attempts']=[{'decision':{'reason':'OLDER-ATTEMPT'}},{'decision':first['decision']}]
    r2['task_context']['recent_history'].append(dict(row,evidence=dict(observation_id='new',step_id=5,images=[])))
    planner.replan(r2)
    turns=[p for m,p in made[0].calls if m=='turn/start']
    assert turns[0]['threadId']==turns[1]['threadId']
    assert len([m for m,p in made[0].calls if m=='thread/start'])==1
    packet=json.loads(turns[1]['input'][0]['text'])
    assert [e['observation_id'] for e in packet['new_evidence']]==['new']
    assert 'OLDER-ATTEMPT' not in turns[1]['input'][0]['text']
    assert 'FIXED-GATE-MUST-NOT-REPEAT' not in turns[1]['input'][0]['text']
    assert 'PAST-SEMANTICS' not in turns[1]['input'][0]['text']
    assert packet['previous_result'] is None  # No actual execution in this fixture.
    assert json.loads(Path(packet['board_path']).read_text())==r2['task_context']
    assert len(turns[1]['input'])==3  # Fresh current dual cameras always attached.
    other=deepcopy(r2);other['episode_id']='different';other['request_id']='other'
    planner.replan(other)
    assert [p for m,p in made[0].calls if m=='turn/start'][-1]['threadId']!=turns[0]['threadId']
    planner.close();assert made[0].closed


def test_retry_cached_and_restart_resumes_exact_thread(tmp_path):
    planner,made=setup(tmp_path);r=next_request(request(tmp_path),1)
    response=planner.replan(r)
    assert planner.replan(r)==response and made[0].turns==1
    changed=deepcopy(r);changed['max_chunk_steps']=2
    with pytest.raises(ValueError,match='reused'):planner.replan(changed)
    planner.close()
    restarted,new=setup(tmp_path);restarted.replan(next_request(r,6))
    assert any(m=='thread/resume' and p['threadId']=='thread-1' for m,p in new[0].calls)
    assert not any(m=='thread/start' for m,p in new[0].calls)


def test_invalid_result_cannot_advance_cursor_or_silently_create_new_thread(tmp_path):
    def factory(*args):return FakeTransport(*args,payload={'decision':{'request_id':'wrong'}})
    p=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory)
    r=next_request(request(tmp_path),1)
    with pytest.raises(ValueError):p.replan(r)
    state=json.loads(next((tmp_path/'audit/episodes').glob('*/thread.json')).read_text())
    assert state['pending'] and not state.get('evidence_keys')
    with pytest.raises(RuntimeError,match='unresolved'):p.replan(r)


def test_historical_images_are_not_reattached(tmp_path):
    p,made=setup(tmp_path);r=next_request(request(tmp_path),1)
    history=[]
    for cam in ('agentview','wrist'):
        f=tmp_path/f'old-{cam}.png';f.write_bytes(b'old')
        history.append(dict(camera=cam,path=str(f)))
    r['evidence_frames']=[dict(observation_id='history',step_id=0,images=history)]
    p.replan(r);p.replan(next_request(r,6))
    turns=[v for m,v in made[0].calls if m=='turn/start']
    assert len(turns[0]['input'])==5 and len(turns[1]['input'])==3
    packet=json.loads(turns[1]['input'][0]['text'])
    assert {i['observation_id'] for i in packet['retained_image_references']}=={'history'}


def test_reader_cannot_access_another_episode(tmp_path):
    p,made=setup(tmp_path);p.replan(next_request(request(tmp_path),1))
    folder=next((tmp_path/'audit/episodes').iterdir())
    for path,success in [(folder/'board.json',True),(tmp_path/'secret.json',False)]:
        event=dict(id=99,params=dict(threadId='thread-1',turnId='t',tool='read_episode_record',
                    arguments=dict(path=str(path),offset=0,limit=10)))
        p.threads.read_record(event,folder,'thread-1','t')
        assert made[0].calls[-1][1][1]['success']==success


def test_timeout_keeps_pending_journal_and_closes_owned_transport(tmp_path):
    made=[]
    class TimeoutTransport(FakeTransport):
        def next_message(self, timeout):raise TimeoutError('fixture timeout')
    def factory(*args):
        t=TimeoutTransport(*args);made.append(t);return t
    p=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory)
    r=next_request(request(tmp_path),1)
    with pytest.raises(TimeoutError):p.replan(r)
    assert made[0].closed
    with pytest.raises(RuntimeError,match='unresolved'):p.replan(next_request(r,6))
    assert len(made)==1


def test_rejects_model_drift_before_turn_submission(tmp_path):
    class WrongModel(FakeTransport):
        def request(self, method, params, timeout):
            result=super().request(method,params,timeout)
            if method=='thread/start':result['model']='other'
            return result
    made=[]
    def factory(*args):
        t=WrongModel(*args);made.append(t);return t
    p=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory)
    with pytest.raises(RuntimeError,match='model'):p.replan(next_request(request(tmp_path),1))
    assert made[0].turns==0 and made[0].closed


def test_foreign_thread_and_old_turn_messages_cannot_supply_decision(tmp_path):
    class Noisy(FakeTransport):
        def request(self,method,params,timeout):
            result=super().request(method,params,timeout)
            if method=='turn/start':
                old=deepcopy(self.events[0]);old['params']['turnId']='old'
                foreign=deepcopy(self.events[0]);foreign['params']['threadId']='other'
                old['params']['item']['text']=foreign['params']['item']['text']='NOT JSON'
                self.events[:0]=[old,foreign]
            return result
    p=AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=Noisy)
    assert p.replan(next_request(request(tmp_path),1))['decision']['request_id']=='proposal-1'


def test_full_audit_growth_does_not_grow_next_turn_prompt(tmp_path):
    from .episode_thread import delta_packet
    from .visual_evidence import evidence_images
    r=next_request(request(tmp_path),1)
    r['previous_attempts']=[{'result':'latest'}]
    r['task_context']={'recent_history':[],'plan_history':[],'astra_interpretations':[]}
    images=evidence_images(r,tmp_path)
    before=delta_packet(r,images,{},tmp_path)[0]
    r['previous_attempts']=[{'result':'OLD'*10000}]*100+r['previous_attempts']
    r['task_context']['plan_history']=[{'instruction':'OLD'*10000}]*100
    r['task_context']['astra_interpretations']=[{'assessment':'OLD'*10000}]*100
    assert delta_packet(r,images,{},tmp_path)[0]==before


def test_fixed_native_protocol_only_sent_once(tmp_path):
    p,made=setup(tmp_path);r=next_request(request(tmp_path),1)
    r['native_request']['action_contract']={'units':'FIXED_UNITS'}
    p.replan(r);p.replan(next_request(r,6))
    turns=[v for m,v in made[0].calls if m=='turn/start']
    assert 'FIXED_UNITS' in turns[0]['input'][0]['text']
    assert 'FIXED_UNITS' not in turns[1]['input'][0]['text']


def test_new_observations_beyond_recent_board_window_are_not_lost(tmp_path):
    from .test_task_context import ledger, update
    from .episode_thread import delta_packet
    from .visual_evidence import evidence_images
    board=ledger()
    for step in range(5,81,5):update(board,step)
    snapshot=board.snapshot()
    assert len(snapshot['recent_history'])==12 and len(snapshot['new_evidence'])==16
    r=next_request(request(tmp_path),80);r['task_context']=snapshot
    packet=delta_packet(r,evidence_images(r,tmp_path),{},tmp_path)[0]
    assert len(packet['new_evidence'])==16
    board.reviewed(80)
    assert not board.snapshot()['new_evidence'] and len(board.snapshot()['recent_history'])==12
