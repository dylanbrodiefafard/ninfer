from __future__ import annotations

import pytest

from tools.bench.run_serve_corpus import CampaignError, require_server_log_identity


def test_request_log_v3_identity_is_accepted() -> None:
    current = {
        "artifact_type": "ninfer_serve_request_log",
        "schema_version": 3,
        "event": "server_start",
    }
    require_server_log_identity(current, "server_start")

    stale = dict(current, schema_version=2)
    with pytest.raises(CampaignError):
        require_server_log_identity(stale, "server_start")
