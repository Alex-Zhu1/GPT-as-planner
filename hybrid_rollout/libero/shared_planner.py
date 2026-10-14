"""Record this client's calls to an existing serial Astra HTTP service.

The service owns its login and process lifetime. This client never starts or
stops it. Episode images must already be inside the service's evidence root.
"""
from pathlib import Path
import time

from ..robodojo.io import write_json
from ..robodojo.jev.planner import HTTPPlanner


class RecordedHTTPPlanner(HTTPPlanner):
    def __init__(self, url, timeout, audit_root, before_request=None):
        super().__init__(url, timeout)
        self.audit_root = Path(audit_root)
        self.before_request = before_request

    def replan(self, request):
        # Use a numbered local directory, never an untrusted request path.
        self.audit_root.mkdir(parents=True, exist_ok=True)
        index = len(list(self.audit_root.glob('call-*')))
        directory = self.audit_root / f'call-{index:03d}'
        directory.mkdir(exist_ok=False)
        write_json(directory / 'request.json', request)
        if self.before_request is not None:
            self.before_request()
        started = time.monotonic()
        try:
            response = super().replan(request)
            if response.get('request_id') != request['request_id']:
                raise ValueError('Shared planner returned another request identity')
            write_json(directory / 'response.json', response)
            return response
        except Exception as error:
            write_json(directory / 'error.json', dict(type=type(error).__name__, detail=str(error)))
            raise
        finally:
            write_json(directory / 'timing.json', dict(
                request_id=request['request_id'], wall_seconds=time.monotonic()-started,
                includes_serial_queue=True, shared_service_owned_by_client=False))
