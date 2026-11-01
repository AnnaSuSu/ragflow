#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

"""Exercise parsing callbacks through loopback HTTP without backend fixtures."""

import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest
from ragflow_sdk import RAGFlow
from ragflow_sdk.modules.dataset import DataSet


@pytest.mark.p2
@pytest.mark.parametrize(
    "status_field, terminal_status, expected_status",
    [
        ("ingestion_status", "COMPLETED", "DONE"),
        ("ingestion_status", "FAILED", "FAIL"),
        ("ingestion_status", "STOPPED", "CANCEL"),
        ("run", "DONE", "DONE"),
        ("run", "FAIL", "FAIL"),
        ("run", "CANCEL", "CANCEL"),
    ],
)
def test_parse_documents_reports_changed_snapshots_over_http(monkeypatch, status_field, terminal_status, expected_status):
    """Report status-only terminal changes while suppressing duplicate snapshots."""
    first = {"id": "doc-1", status_field: "RUNNING", "progress": 0.25, "progress_msg": "Parsing", "chunk_count": 2, "token_count": 4}
    progress = {**first, "progress": 0.5}
    message = {**progress, "progress_msg": "Still parsing"}
    terminal = {**message, status_field: terminal_status}
    snapshots = [first, first, progress, message, message, terminal]
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            """Suppress local HTTP request logging."""

        def respond(self, payload):
            """Return an API envelope using a real HTTP response."""
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            """Accept the SDK's parsing request."""
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((self.command, self.path, body))
            self.respond({"code": 0})

        def do_GET(self):
            """Serve successive document snapshots to the SDK polling loop."""
            url = urlsplit(self.path)
            requests.append((self.command, url.path, parse_qs(url.query)))
            self.respond({"code": 0, "data": {"docs": [snapshots.pop(0)]}})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(time, "sleep", lambda _interval: None)
    observed = []
    try:
        dataset = DataSet(RAGFlow("test-key", f"http://127.0.0.1:{server.server_port}"), {"id": "dataset-1"})

        def on_progress(doc):
            """Record the real Document values delivered to user code."""
            observed.append((getattr(doc, status_field), doc.progress, doc.progress_msg))

        assert dataset.parse_documents(["doc-1"], on_progress=on_progress) == [("doc-1", expected_status, 2, 4)]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert observed == [(row[status_field], row["progress"], row["progress_msg"]) for row in (first, progress, message, terminal)]
    assert not snapshots
    assert requests[0] == ("POST", "/api/v1/datasets/dataset-1/chunks", {"document_ids": ["doc-1"]})
    assert len(requests) == 7
    assert all(method == "GET" and path == "/api/v1/datasets/dataset-1/documents" and query["id"] == ["doc-1"] for method, path, query in requests[1:])
