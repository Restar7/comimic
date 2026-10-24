"""PolicyServer roundtrip for the RTC endpoints (CPU, no weights).

Guards the P5 wiring: ``get_candidates`` must be registerable on top of the
official server and survive the msgpack-numpy wire with float32 arrays.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest


class _StubPolicy:
    """Minimal policy surface used by ``PolicyServer`` and the RTC client."""

    def get_action(self, observation, options=None):
        return (
            {"motion_token": np.zeros((1, 40, 64), np.float32)},
            {"selected_index": [0], "anchor_tick": None if options is None else options.get("anchor_tick")},
        )

    def reset(self, options=None):
        return {}

    def get_candidates(self, observation, options=None):
        return {
            "candidates": np.zeros((1, 2, 40, 78), np.float32),
            "normalized": np.zeros((1, 2, 40, 132), np.float32),
            "candidate_ids": ["c0", "c1"],
            "anchor_tick": None if options is None else options.get("anchor_tick"),
        }


def test_policy_server_roundtrips_get_action_and_candidates() -> None:
    zmq = pytest.importorskip("zmq")
    from gr00t.policy.server_client import PolicyClient, PolicyServer

    server = PolicyServer(_StubPolicy(), host="127.0.0.1", port=0)
    server.register_endpoint("get_candidates", server.policy.get_candidates)
    endpoint = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    port = int(endpoint.rsplit(":", 1)[1])
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    client = PolicyClient(host="127.0.0.1", port=port, timeout_ms=5000)
    try:
        payload = client.call_endpoint(
            "get_candidates", {"observation": {}, "options": {"anchor_tick": 5}}
        )
        assert payload["candidates"].shape == (1, 2, 40, 78)
        assert payload["normalized"].shape == (1, 2, 40, 132)
        assert payload["candidates"].dtype == np.float32
        assert payload["anchor_tick"] == 5

        action, info = client.get_action({})
        assert action["motion_token"].shape == (1, 40, 64)
        assert info["selected_index"] == [0]

        # Stop the REP loop through its own kill endpoint so the socket is not
        # closed underneath a blocked recv().
        client.call_endpoint("kill", requires_input=False)
        thread.join(timeout=5)
        assert not thread.is_alive()
    finally:
        client.close()
        server.close()
