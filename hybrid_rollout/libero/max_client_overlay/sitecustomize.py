"""MAX 客户端专用：允许首次 JAX 编译期间延迟响应 WebSocket 心跳。

由 start_libero_max.sh 通过官方 OPENPI_CLIENT_OVERLAY 接口加载；
不修改 benchmark、动作、种子或成功判定，也不影响原始 LIBERO 环境。
"""

import functools
import os

# Official paired shell forces EGL. Override it before any MuJoCo/OpenGL import
# when the caller explicitly selects the validated software rendering backend.
if os.environ.get('JEV_MAX_GL'):
    backend = os.environ['JEV_MAX_GL']
    if backend not in ('egl', 'osmesa'):
        raise ValueError('JEV_MAX_GL must be egl or osmesa')
    os.environ['MUJOCO_GL'] = os.environ['PYOPENGL_PLATFORM'] = backend
    os.environ.setdefault('LP_NUM_THREADS', '2')

if os.environ.get("JEV_MAX_WS_PING_TIMEOUT"):
    import websockets.sync.client

    websockets.sync.client.connect = functools.partial(
        websockets.sync.client.connect,
        ping_timeout=float(os.environ["JEV_MAX_WS_PING_TIMEOUT"]),
    )
