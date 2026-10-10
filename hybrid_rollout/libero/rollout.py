"""File-backed adapter implementing the existing Jev scheduler protocol."""
from copy import deepcopy
import os
from pathlib import Path
import uuid

import numpy as np
from PIL import Image

from ..robodojo.io import write_json
from ..robodojo.jev.runner import automatic_response
from .contract import PROFILE, RECOVERY_CONTRACT, execution_actions, validate_response
from .session import CAMERAS


class LiberoRollout:
    validate = staticmethod(validate_response)
    recovery_contract = RECOVERY_CONTRACT

    def __init__(self, output, student, session_factory, *, max_decisions=10, chunk_steps=1):
        if type(max_decisions) is not int or max_decisions < 0:
            raise ValueError('max_decisions must be nonnegative; zero uses suite horizon')
        if type(chunk_steps) is not int or not 1 <= chunk_steps <= 5:
            raise ValueError('chunk_steps must be in [1,5]')
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.student, self.session_factory = student, session_factory
        self.max_decisions, self.chunk_steps = max_decisions, chunk_steps
        self.session, self.episode, self.tick = None, None, 0
        self.phase, self.history, self.run = 'start', [], {}
        self.observation_packet = {}

    def next_call(self):
        if self.phase == 'start':
            return {'tool': 'libero_start'}
        if self.phase == 'execute':
            return {'tool': 'libero_execute'}
        if self.phase == 'infer':
            return {'tool': 'pi05_infer'}
        return None

    def start(self):
        if self.phase != 'start':
            raise ValueError('Cannot reset a running LIBERO episode')
        self.session = self.session_factory()
        # Assign identity before initialization so failures still produce an audit result.
        self.episode = uuid.uuid4().hex
        self.session.start()
        self.meta = dict(instruction=self.session.instruction, robot_profile=PROFILE,
                         max_episode_steps=self.session.horizon)
        self.run = dict(evaluation_method='pi05_valen_replan', environment='LIBERO/MuJoCo',
                        episode_id=self.episode, initialization_settle_steps=self.session.settle_steps,
                        provenance=self.session.provenance,
                        policy=self.student.metadata, action_space='normalized OSC_POSE xyz/rotation/gripper')
        write_json(self.output / 'run.json', self.run)
        self._observation(self.output / 'observations/000')
        self.phase = 'infer'

    def _observation(self, directory, *, refresh=False, result=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        self.obs = self.session.observe(refresh=refresh)
        self.tick = self.session.tick
        images = []
        for name in CAMERAS:
            path = directory / f'{name}.png'
            Image.fromarray(self.obs[name]).save(path)
            images.append({'camera': name, 'path': str(path)})
        np.savez_compressed(directory / 'observation.npz', **{
            k: self.obs[k] for k in (*CAMERAS, 'states', 'eef_position', 'eef_quaternion_xyzw')})
        self.observation_packet = dict(episode_id=self.episode, step_id=self.tick,
            observation_id=uuid.uuid4().hex, captured_at=self.obs['captured_at'],
            instruction=self.session.instruction, images=images, current_state=self.obs['states'].tolist(),
            current_eef={'position': self.obs['eef_position'].tolist(),
                         'quaternion_xyzw': self.obs['eef_quaternion_xyzw'].tolist(),
                         'frame': 'MuJoCo world', 'gripper_qpos': self.obs['states'][6:].tolist()},
            robot_profile=PROFILE)
        if result is not None:
            self.observation_packet['result'] = result
        write_json(directory / 'observation.json', self.observation_packet)
        return self.observation_packet

    def _inference_packet(self, directory, *, student_instruction):
        if self.phase != 'infer':
            raise ValueError('Cannot install a proposal at this phase')
        index = len(self.history)
        os.link(self.proposal_path, self.output / f'proposal_{index:03d}.npz')
        self.request = deepcopy(self.observation_packet)
        self.request.pop('result', None)
        self.request.update(request_id=uuid.uuid4().hex, student_instruction=student_instruction,
            max_chunk_steps=self.chunk_steps, proposal_actions=self.actions.tolist(),
            action_contract=RECOVERY_CONTRACT, **self.prediction)
        self.request['assessment_example'] = automatic_response(self.request, 1, None if self.tick == 0 else {})['assessment']
        write_json(self.output / f'request_{index:03d}.json', self.request)
        self.phase = 'execute'
        return dict(self.request, proposal_path=str(self.proposal_path))

    def execute(self, *, response):
        if self.phase != 'execute':
            raise ValueError('No fresh proposal is ready')
        self.validate(response, self.request)
        if response['mode'] == 'stop':
            return self.finish('planner_stop')
        index, before = len(self.history), self.tick
        actions = execution_actions(self.actions, response)
        write_json(self.output / f'response_{index:03d}.json', response)
        executed = self.session.execute(actions)
        self.tick = self.session.tick
        np.savez_compressed(self.output / f'execution_{index:03d}.npz', actions=executed)
        self.history.append(dict(decision=index, source_detail=response['mode'], start_tick=before,
                                 end_tick=self.tick, executed_steps=len(executed)))
        write_json(self.output / 'history.json', self.history)
        capped = self.max_decisions > 0 and len(self.history) >= self.max_decisions
        self.phase = 'done' if self.session.terminal or capped else 'infer'
        result = self.finish('native_terminal' if self.session.terminal else 'decision_budget') if self.phase == 'done' else None
        return self._observation(self.output / 'observations' / f'{index + 1:03d}', result=result)

    def finish(self, reason):
        # Never derive success from Valen, planner text, or an exhausted budget.
        result = dict(episode_id=self.episode, step_id=self.session.tick if self.session else self.tick,
                      success=bool(self.session and self.session.success and not self.session.poisoned),
                      complete=bool(self.session and self.session.terminal and not self.session.poisoned),
                      reason=reason, execution_uncertain=bool(self.session and self.session.poisoned))
        self.phase = 'done'
        self.observation_packet['result'] = result
        write_json(self.output / 'result.json', result)
        return result

    def close(self):
        if self.session is not None:
            self.session.close()
