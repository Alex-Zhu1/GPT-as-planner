"""Identified Pi 0.5 LIBERO websocket client; no simulator access in workers."""
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

from .contract import CONFIG, HORIZON, PROFILE, action_chunk


def policy_image(value):
    image = Image.fromarray(value)
    w, h = image.size
    scale = 224 / max(w, h)
    resized = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
    canvas = Image.new('RGB', (224, 224))
    canvas.paste(resized, ((224 - resized.width) // 2, (224 - resized.height) // 2))
    return np.asarray(canvas)


class Pi05Client:
    def __init__(self, checkpoint, *, host='127.0.0.1', port=18830, client_factory=None):
        if client_factory is None:
            from openpi_client.websocket_client_policy import WebsocketClientPolicy
            client_factory = WebsocketClientPolicy
        self.client = client_factory(host, port)
        try:
            self.client._ws.ping_timeout = None
            self.metadata = self.client.get_server_metadata()
            expected = dict(config=CONFIG, robot_profile=PROFILE, action_dim=7,
                            action_horizon=HORIZON, inferences=0,
                            checkpoint=str(Path(checkpoint).resolve()))
            if (any(self.metadata.get(k) != v for k, v in expected.items())
                    or type(self.metadata.get('inferences')) is not int
                    or not self.metadata.get('checkpoint_sha256')):
                raise ValueError('A fresh identified pi05_libero server is required; RoboDojo weights are incompatible')
        except BaseException:
            self.close()
            raise
        self.index = 0

    def infer(self, observation, output):
        before = time.monotonic()
        state = np.asarray(observation['states'], np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError('LIBERO state must be EEF position3 + axis-angle3 + gripper qpos2')
        result = self.client.infer({'observation/image': policy_image(observation['agentview']),
            'observation/wrist_image': policy_image(observation['wrist']),
            'observation/state': state, 'prompt': observation['instruction']})
        identity = result.get('policy_identity', {})
        if (identity.get('checkpoint_sha256') != self.metadata['checkpoint_sha256']
                or type(identity.get('inference_index')) is not int or identity['inference_index'] != self.index):
            raise ValueError('Stale or unrelated LIBERO policy response')
        actions = action_chunk(result['actions'], horizon=HORIZON)
        np.savez_compressed(output, raw_actions=actions, actions=actions, states=state,
                            agentview=observation['agentview'], wrist=observation['wrist'])
        self.index += 1
        return actions, dict(prediction_id=uuid.uuid4().hex, inference_index=self.index - 1,
                             inference_seconds=time.monotonic() - before)

    def close(self):
        self.client._ws.close()
