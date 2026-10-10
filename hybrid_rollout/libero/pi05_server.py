"""Load a local official pi05_libero checkpoint and serve its 10x7 actions."""
import argparse
import hashlib
import json
from pathlib import Path

from ..robodojo.io import sha256, write_json
from .contract import CONFIG, HORIZON, PROFILE, action_chunk


def checkpoint_identity(path):
    root = Path(path).resolve()
    files = {str(p.relative_to(root)): sha256(p) for folder in ('params', 'assets')
             for p in sorted((root / folder).rglob('*')) if p.is_file()}
    if not any(name.startswith('params/') for name in files):
        raise ValueError('Expected a local OpenPI JAX checkpoint with params/')
    if not any(name.endswith('/norm_stats.json') for name in files if name.startswith('assets/')):
        raise ValueError('Checkpoint normalization assets are missing')
    return dict(checkpoint=str(root), checkpoint_sha256=hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest(), files_sha256=files,
        config=CONFIG, robot_profile=PROFILE, backend='OpenPI/JAX', action_dim=7, action_horizon=HORIZON)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--identity-output', type=Path, required=True)
    p.add_argument('--port', type=int, default=18830)
    a = p.parse_args(argv)
    if a.identity_output.exists():
        p.error('Use a fresh identity output for each server')
    from openpi.training import config
    from openpi.policies.policy_config import create_trained_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    cfg = config.get_config(CONFIG)
    if cfg.model.action_horizon != HORIZON:
        raise ValueError('Incompatible pi05_libero action horizon')
    identity = checkpoint_identity(a.checkpoint)
    # create_trained_policy loads this config's normalization asset from the checkpoint.
    policy = create_trained_policy(cfg, a.checkpoint.resolve())
    if identity != checkpoint_identity(a.checkpoint):
        raise ValueError('Checkpoint changed while loading')
    metadata = dict(identity, inferences=0)

    class CountedPolicy:
        def infer(self, obs):
            result = policy.infer(obs)
            action_chunk(result['actions'], horizon=HORIZON)
            result['policy_identity'] = dict(checkpoint_sha256=identity['checkpoint_sha256'],
                                             inference_index=metadata['inferences'])
            metadata['inferences'] += 1
            return result

    a.identity_output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.identity_output, metadata)
    print(json.dumps({'event': 'loaded', 'config': CONFIG, 'port': a.port}), flush=True)
    WebsocketPolicyServer(CountedPolicy(), host='127.0.0.1', port=a.port, metadata=metadata).serve_forever()


if __name__ == '__main__':
    main()
