"""Reject stable-but-empty EGL buffers before they become model evidence."""
import numpy as np


def validate_camera_images(raw):
    for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
        value = np.asarray(raw[camera])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise RuntimeError(f'Invalid RGB buffer: {camera}')
        if float(value.std()) < .5 or int(value.max())-int(value.min()) < 5:
            raise RuntimeError(f'Empty/flat EGL image: {camera}')


def install_render_validation():
    from libero_max import env_factory
    if getattr(env_factory, '_jev_nonempty_render_validation', False):
        return
    original = env_factory.prime_offscreen_renderer

    def prime(*args, **kwargs):
        result = original(*args, **kwargs)
        if result.get('status')=='passed' and any(
                value <= .01 for value in result.get('neighbor_deltas', {}).values()):
            raise env_factory.RenderStabilityError('Stable but empty/flat EGL camera buffer')
        return result

    env_factory.prime_offscreen_renderer = prime
    env_factory._jev_nonempty_render_validation = True
