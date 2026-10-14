"""Pure pi05 prefixes, native outcomes and explicit absence of model review."""
import json

import numpy as np
import pytest

from .baseline import Pi05OnlyTools, run_baseline
from .test_contract import Student
from .test_jev import LongSim


@pytest.mark.parametrize('chunk', [15, 25])
def test_pure_pi_preserves_instruction_and_unmodified_prefix(tmp_path, chunk):
    inputs, commands, headers = [], [], []
    class DistinctStudent(Student):
        def infer(self, obs, output):
            inputs.append(obs['instruction'])
            actions, info = super().infer(obs, output)
            actions[:,0] = np.arange(50)/100
            np.savez_compressed(output, actions=actions, raw_actions=actions)
            return actions, info
    class Sim(LongSim):
        horizon = 63
        def request(self, op, **kwargs):
            assert op not in ('refresh_observation','switch_control_source','eef_joint_target')
            if op=='chunk_step':commands.append(kwargs['actions'].copy())
            if op=='begin_combination':headers.append(kwargs)
            return super().request(op, **kwargs)
    rollout = Pi05OnlyTools(tmp_path,'arrange_largest_number',DistinctStudent(),
                           rpc_factory=Sim,max_decisions=0,student_step_limit=chunk)
    result = run_baseline(rollout,chunk_steps=chunk)
    assert result['state']=='COMPLETED' and result['native_result']['success']
    assert rollout.tick==63 and rollout.sim.calls.count('reset')==1
    assert all(x=='Arrange the digits.' for x in inputs)
    assert len(commands)==(63+chunk-1)//chunk
    for command in commands:
        np.testing.assert_allclose(command[:,0], np.arange(chunk)/100)
    for record in rollout.history:
        assert record['source_detail']=='student' and record['takeover_trigger']=='none'
        assert record['response']['decision_author']=='pi05_baseline'
    assert result['monitor_calls']==result['astra_calls']==0
    assert headers[0]['teacher_model'] is None and headers[0]['teacher_off']
    run=json.loads((tmp_path/'run.json').read_text())
    assert run['teacher_off'] and not run['monitor_enabled'] and run['teacher_model'] is None


def test_native_failure_is_completed_evaluation_not_infrastructure_error(tmp_path):
    class FailedSim(LongSim):
        horizon=30
        def request(self,op,**kwargs):
            result=super().request(op,**kwargs)
            rows=result['steps'] if op=='chunk_step' else [result] if op=='finish_pilot' else []
            for row in rows:
                if row['terminated']:
                    row.update(terminated=False,truncated=True,success=False)
            return result
    rollout=Pi05OnlyTools(tmp_path,'arrange_largest_number',Student(),
                         rpc_factory=FailedSim,max_decisions=0,student_step_limit=25)
    result=run_baseline(rollout)
    assert result['state']=='COMPLETED'
    assert result['native_result']['complete'] and not result['native_result']['success']
    assert result['step_id']==30


def test_infrastructure_failure_remains_incomplete(tmp_path):
    class BrokenStudent(Student):
        def infer(self,*args):raise RuntimeError('policy unavailable')
    rollout=Pi05OnlyTools(tmp_path,'arrange_largest_number',BrokenStudent(),
                         rpc_factory=LongSim,max_decisions=0,student_step_limit=25)
    result=run_baseline(rollout)
    assert result['state']=='STOPPED' and 'policy unavailable' in result['reason']
    assert result['step_id']==0 and not result['native_result']['complete']


def test_decision_cutoff_is_rejected_before_simulator_reset(tmp_path):
    rollout=Pi05OnlyTools(tmp_path,'arrange_largest_number',Student(),rpc_factory=LongSim,
                         max_decisions=1,student_step_limit=25)
    with pytest.raises(ValueError,match='native termination'):run_baseline(rollout)
    assert rollout.sim is None
