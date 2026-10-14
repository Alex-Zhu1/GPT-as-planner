"""Native-task smoke with a separately reported, non-executing Astra probe."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import uuid

from ..robodojo.io import write_json
from ..robodojo.jev.monitor import IDENTITY
from ..robodojo.jev.planner import HTTPPlanner
from ..robodojo.jev.plans import validate_decision
from .contract import RECOVERY_CONTRACT, validate_response
from . import run


def planner_probe(output, summary, url, timeout):
    # Use the complete recorded observation and its own real Pi proposal together.
    # This is an offline protocol probe, never a request to resume the simulator.
    paths = sorted(output.glob('request_*.json'))
    if not paths:
        raise ValueError('No real policy proposal is available for the Astra probe')
    native = json.loads(paths[-1].read_text())
    scene = {key: deepcopy(native[key]) for key in (
        'episode_id', 'step_id', 'observation_id', 'captured_at', 'instruction', 'images',
        'current_state', 'current_eef', 'robot_profile')}
    request = dict(version=1, request_id=uuid.uuid4().hex,
        **{key: summary[key] for key in IDENTITY},
        original_task=native['instruction'], remaining_task=native['instruction'],
        current_scene=scene, native_request=native, remaining_plan=summary['remaining_plan'],
        failure={'state': 'SMOKE_PROBE', 'note': 'Recorded-scene protocol probe only; '
                 'no confirmed failure is asserted and the decision will not be executed.'},
        confirmation_reports=[], evidence_frames=[], previous_attempts=[],
        max_chunk_steps=native['max_chunk_steps'], contract=RECOVERY_CONTRACT)
    write_json(output / 'smoke_planner_request.json', request)
    response = HTTPPlanner(url, timeout).replan(request)
    if set(response) != {'request_id', *IDENTITY, 'decision'} or any(
            type(response[k]) is not type(request[k]) or response[k] != request[k]
            for k in ('request_id', *IDENTITY)):
        raise ValueError('Astra probe returned a stale envelope')
    validate_decision(response['decision'], native, native['max_chunk_steps'], validate_response)
    write_json(output / 'smoke_planner_response.json', response)
    return dict(status='passed', mode=response['decision']['mode'], simulation_executed=False)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--planner-url', required=True)
    p.add_argument('--planner-timeout', type=float, default=300.)
    a, _ = p.parse_known_args(argv)
    output = a.output.resolve()
    controller_code = run.main(argv)
    summary = json.loads((output / 'jev_summary.json').read_text())
    report = dict(controller_exit_code=controller_code, controller_state=summary['state'],
                  controller_reason=summary['reason'], executed_steps=summary['step_id'],
                  native_result=summary['native_result'], recovery_calls=summary['replan_count'],
                  uncertainty_reviews=summary.get('uncertainty_review_count', 0),
                  uncertainty_policy=summary.get('uncertainty_policy', 'pause'),
                  uncertain_continuations=summary.get('uncertain_continuation_count', 0),
                  monitor_responded=summary['latest_feedback'] is not None)
    try:
        report['astra_probe'] = planner_probe(output, summary, a.planner_url, a.planner_timeout)
    except Exception as error:
        report['astra_probe'] = dict(status='failed', error=f'{type(error).__name__}: {error}')
    expected_end = summary['state'] == 'COMPLETED' or (
        summary['state'] == 'STOPPED' and summary['reason'] == 'Decision budget exhausted')
    report['component_check'] = 'passed' if (expected_end and summary['step_id'] > 0
        and report['monitor_responded'] and report['astra_probe']['status'] == 'passed') else 'incomplete'
    native = summary['native_result'] or {}
    report['task_success'] = (native.get('success') is True and native.get('complete') is True
                              and not native.get('execution_uncertain', False))
    report['status'] = 'passed' if report['task_success'] and controller_code == 0 else 'incomplete'
    report['note'] = ('Task smoke requires native instruction success. Budget/horizon exhaustion '
                     'and a successful separate Astra probe never establish task success.')
    write_json(output / 'smoke_report.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
