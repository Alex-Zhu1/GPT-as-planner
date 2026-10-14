import json
from pathlib import Path
import pytest
from .stage1 import fixed_step, eligible, freeze

@pytest.mark.parametrize('horizon,step', [(220,45),(280,60),(300,60),(520,105)])
def test_fixed_time_is_preregistered_horizon_fraction(horizon,step):
    assert fixed_step(horizon)==step
    assert not eligible('fixed',step-5,horizon,0)
    assert eligible('fixed',step,horizon,0)
    assert not eligible('fixed',step,horizon,1)
    assert not eligible('none',step,horizon,0)
    assert not eligible('jev',step,horizon,0)


def test_manifest_balanced_and_frozen_before_rollouts(tmp_path):
    repo=tmp_path/'repo'; (repo/'hybrid_rollout/libero').mkdir(parents=True)
    (repo/'hybrid_rollout/robodojo/jev').mkdir(parents=True)
    for name in ('stage1.py','stage1_planner.py','max_run.py','planner_server.py'):
        (repo/'hybrid_rollout/libero'/name).write_text('preserved')
    for name in ('runner.py','monitor.py'):
        (repo/'hybrid_rollout/robodojo/jev'/name).write_text('preserved')
    cases=[]
    for suite in ('libero_spatial','libero_object','libero_goal','libero_10'):
        for index in (9,3):
            cases.append(dict(task_suite_name=suite,task_index=index,case_id=suite+str(index),
                policy_seed=195,scenario={'change_type':'camera_shift'}))
    path=tmp_path/'model_/datasets/LIBERO-MAX/benchmark/max8000/libero_max_8000.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps({'cases':cases}))
    source=tmp_path/'source';source.mkdir()
    (source/'astra_on-config.json').write_text(json.dumps(dict(services=[],controller={'argv':['python']})))
    output=tmp_path/'frozen';manifest=freeze(repo,source,output)
    assert len(manifest['episodes'])==36
    assert all(c['task_index']==3 for c in manifest['original_cases'])
    for pair in {e['pair_id'] for e in manifest['episodes']}:
        rows=[e for e in manifest['episodes'] if e['pair_id']==pair]
        assert {e['mode'] for e in rows}=={'none','fixed','jev'}
        assert rows[0]['case']==rows[1]['case']==rows[2]['case']
    with pytest.raises(FileExistsError):freeze(repo,source,output)
