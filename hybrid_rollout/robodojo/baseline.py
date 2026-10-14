"""Pure pi05 with the same native execution, prefix and recording as the harness."""
import argparse
import json
from pathlib import Path

from .io import write_json
from .robodojo_server.client import RoboDojoTools


class Pi05OnlyTools(RoboDojoTools):
    def _rpc(self, op, **kwargs):
        if op == 'begin_combination':
            kwargs.update(teacher_model=None, teacher_model_provider=None,
                          teacher_off=True, evaluation_method='pi05_only')
        return super()._rpc(op, **kwargs)

    def start(self, **kwargs):
        packet = super().start(**kwargs)
        self.run.update(evaluation_method='pi05_only', teacher=None, teacher_off=True,
            teacher_model=None, teacher_model_provider=None, teacher_reasoning_effort=None,
            teacher_invocation='none', monitor_enabled=False, gate_policy='none',
            chunk_steps=self.student_step_limit, original_instruction_only=True)
        write_json(self.output/'run.json', self.run)
        return packet

    def _inference_packet(self, directory, *, student_instruction=None):
        if student_instruction is not None and student_instruction != self.obs['instruction']:
            raise ValueError('Pure pi05 must retain the native task instruction')
        packet = super()._inference_packet(directory, student_instruction=student_instruction)
        fields = dict(gate_policy='student-only', evaluation_method='pi05_only')
        self.request.update(fields, gate_instruction='Execute the student prefix without model review.')
        packet.update(fields)
        write_json(self.output/f"request_{len(self.history):03d}.json", self.request)
        write_json(directory/'proposal.json', packet)
        return packet


def student_response(request, steps):
    task = request['instruction']
    return dict(request_id=request['request_id'], mode='student', steps=steps,
        evaluation_method='pi05_only', decision_author='pi05_baseline',
        reason='Execute the unmodified student prefix; no monitor or teacher is active.',
        assessment=dict(task_progress=dict(verified_completed=[], currently_attempting=task, remaining=[task]),
            current_subgoal=task, execution_status='not_started' if request['step_id']==0 else 'uncertain',
            execution_evidence='Not assessed: pure pi05 baseline.', expected_next_intent=task,
            predicted_next_intent='Not assessed: pure pi05 baseline.', intent_status='uncertain',
            intent_evidence='No intent gate or model review was performed.'))


def run_baseline(rollout, *, chunk_steps=25, emit=None):
    if type(chunk_steps) is not int or not 1 <= chunk_steps <= rollout.student_step_limit:
        raise ValueError('Chunk must fit the host student limit')
    if rollout.max_decisions != 0:
        raise ValueError('Baseline requires native termination, without a decision-count cutoff')
    emit = emit or (lambda event: None)
    state, reason, native = 'STOPPED', '', None
    try:
        args = rollout.next_call().copy(); args.pop('tool')
        rollout.start(**args)
        emit(dict(kind='started', step_id=rollout.tick, instruction=rollout.meta['instruction']))
        while rollout.phase != 'done':
            args = rollout.next_call().copy(); args.pop('tool')
            rollout.infer(**args)
            if rollout.request['student_instruction'] != rollout.meta['instruction']:
                raise ValueError('Baseline instruction changed during the episode')
            args = rollout.next_call().copy(); args.pop('tool')
            args['response'] = student_response(rollout.request, chunk_steps)
            before = rollout.tick
            packet = rollout.execute(**args)
            if not 0 < rollout.tick-before <= chunk_steps:
                raise RuntimeError('Simulator violated the student prefix bound')
            emit(dict(kind='executed_chunk', step_id=rollout.tick, start_step=before,
                      executed_steps=rollout.tick-before, prediction_id=rollout.prediction['prediction_id']))
            native = packet.get('result')
        state = 'COMPLETED' if native and native['complete'] else 'STOPPED'
        reason = 'Native rollout ended' if state=='COMPLETED' else 'Incomplete native result'
    except Exception as error:
        reason = f'{type(error).__name__}: {error}'
        if rollout.episode is not None and rollout.phase != 'done':
            try:
                native = rollout.finish('pi05_baseline_error')
            except Exception as finish_error:
                native = dict(complete=False, success=None, finish_error=str(finish_error))
    summary = dict(evaluation_method='pi05_only', state=state, reason=reason,
        step_id=rollout.tick, chunk_steps=chunk_steps, native_result=native,
        predictions=rollout.counters['predictions'], monitor_calls=0, astra_calls=0,
        original_instruction_only=True)
    write_json(rollout.output/'baseline_summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--sim-port', type=int, default=19113)
    parser.add_argument('--student-port', type=int, default=18830)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk-steps', type=int, choices=range(1,26), default=25)
    args = parser.parse_args(argv)
    from .pi05_server.client import Pi05Client
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    student = rollout = None
    try:
        student = Pi05Client(args.student_port, args.checkpoint)
        rollout = Pi05OnlyTools(output, args.task, student, sim_port=args.sim_port,
                               seed=args.seed, max_decisions=0, student_step_limit=args.chunk_steps)
        with (output/'baseline_events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event, allow_nan=False)+'\n'); stream.flush()
            result = run_baseline(rollout, chunk_steps=args.chunk_steps, emit=emit)
        print(json.dumps(result, indent=2))
        return 0 if result['state']=='COMPLETED' else 2
    finally:
        if rollout is not None:rollout.close()
        if student is not None:student.close()


if __name__ == '__main__':
    raise SystemExit(main())
