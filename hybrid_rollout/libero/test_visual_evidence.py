"""Public image selection and actual CLI attachment contract; CPU only."""
from copy import deepcopy
import importlib
import json

import pytest
from PIL import Image

from .task_context import TaskContext, board_evidence_ids
from .visual_evidence import evidence_images, pixel_change
from .test_task_context import frame, goals
from .test_recovery_protocol import operation_request, student, assessment
from ..robodojo.jev.monitor import MonitorConfig


def pictured(root, step, value, revision=0):
    f=frame(step,revision)
    for camera in ('agentview','wrist'):
        p=root/f'{revision}-{step}-{camera}.png'
        Image.new('RGB',(8,8),(value,)*3).save(p)
        f['images'].append(dict(camera=camera,path=str(p)))
    return f


def test_public_change_pair_survives_replan_and_is_not_semantic_fact(tmp_path):
    board=TaskContext(MonitorConfig(observation_profile='task_progress'))
    a=pictured(tmp_path,0,0);b=pictured(tmp_path,5,200);c=pictured(tmp_path,10,201)
    board.activate(a,goals())
    board.record_execution_images(a,b,'student')
    board.record_execution_images(b,c,'student')
    d=pictured(tmp_path,10,201,1);board.activate(d,goals())
    snap=board.snapshot()
    assert snap['keyframes']['visual_change_before']['observation_id']==a['observation_id']
    assert snap['keyframes']['visual_change_after']['observation_id']==b['observation_id']
    assert snap['keyframe_selection']['semantic_change_not_asserted']
    assert 'status' not in snap['task']
    assert pixel_change(b,c)<pixel_change(a,b)
    assert pixel_change(b,b)==0
    wrong=deepcopy(c);wrong['episode_id']='other'
    with pytest.raises(ValueError,match='cross episodes'):board.remember('last_decision',wrong)


def test_event_pairs_have_priority_and_dedup_keeps_all_observation_references(tmp_path):
    a=pictured(tmp_path,0,0);b=pictured(tmp_path,5,200);c=pictured(tmp_path,10,210)
    board=TaskContext(MonitorConfig(observation_profile='task_progress'));board.activate(a,goals())
    board.record_execution_images(a,b,'student');board.remember('last_decision',b)
    board.record_execution_images(b,c,'eef')
    r=dict(current_scene=c,review_kind='progress',failure={'first_step':10},
           evidence_frames=[a,b,c],task_context=board.snapshot())
    images=evidence_images(r,tmp_path)
    assert len(images)==6  # Both cameras from 3 endpoints; repeated roles do not add pixels.
    assert [i['label'] for i in images[:2]]==['current_scene']*2
    assert all(i['observation_id'] for i in images)
    refs=[ref for i in images for ref in i['references']]
    assert {ref['role'] for ref in refs}>={'current_scene','visual_change_before','visual_change_after','last_decision','eef_before','eef_after'}
    assert all(ref['observation_id'] in board_evidence_ids(r) for ref in refs)
    assert any(ref['step_id']==0 for ref in refs)  # Pinned evidence is not removed by trigger first_step.
    bad=deepcopy(r);bad['task_context']['keyframes']['eef_before']['images'][0]['path']=str(tmp_path.parent/'outside.png')
    with pytest.raises(ValueError,match='evidence-root'):evidence_images(bad,tmp_path)


@pytest.mark.parametrize('module',['planner_server','stage1_planner'])
def test_prompt_manifest_and_rpc_actually_attach_current_and_keyframe_pixels(tmp_path,monkeypatch,module):
    mod=importlib.import_module('hybrid_rollout.libero.'+module)
    r=operation_request(tmp_path)
    current=pictured(tmp_path,20,150,3);old=pictured(tmp_path,15,100,2)
    r['current_scene']=current
    r['task_context']={'keyframes':{'last_decision':old,'eef_before':old,'eef_after':current}}
    from .test_episode_thread import FakeTransport
    made=[]
    def factory(*args):
        t=FakeTransport(*args,payload=dict(decision=student(r),board_assessment=assessment(r)))
        made.append(t);return t
    mod.AstraPlanner('/unused',tmp_path/'audit',tmp_path,transport_factory=factory).replan(r)
    manifest=json.loads(next((tmp_path/'audit').glob('*/image_manifest.json')).read_text())
    turn=next(p for m,p in made[0].calls if m=='turn/start')
    attached=[i['path'] for i in turn['input'] if i['type']=='localImage']
    seen={'prompt':next(p['developerInstructions'] for m,p in made[0].calls if m=='thread/start')+turn['input'][0]['text']}
    assert attached==[i['path'] for i in manifest] and len(attached)==4
    assert {i['observation_id'] for i in manifest}=={current['observation_id'],old['observation_id']}
    for phrase in ('Combine the current two camera images','status board','NOT a detected perturbation',
                   'never carry forward an old judgment','Only cite attached observations as visual evidence'):
        assert phrase in seen['prompt']
    assert old['observation_id'] in seen['prompt']
