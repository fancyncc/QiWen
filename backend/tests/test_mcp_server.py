from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient


def test_mcp_exposes_only_public_read_only_knowledge(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from app import db
    from app.mcp_server import check_herb_safety, lookup_herb, mcp, search_formulas

    monkeypatch.setattr(db, "DB_PATH", tmp_path / "mcp.sqlite3")
    db.init_db()
    assert lookup_herb("甘草")["name"] == "甘草"
    assert search_formulas("桂枝汤")[0]["name"] == "桂枝汤"
    assert check_herb_safety(["甘草", "海藻"])["safe"] is False
    assert check_herb_safety(["未知药材"])["safe"] is False
    names = {tool.name for tool in asyncio.run(mcp.list_tools())}
    assert names == {"search_tcm_classics", "lookup_herb", "search_formulas", "check_herb_safety", "get_my_patient_profile"}
    assert "get_patient_profile" not in names


@pytest.fixture
def identities(tmp_path, monkeypatch):
    pytest.importorskip("mcp")
    from app import db
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "profiles.sqlite3")
    db.init_db()
    with db.connection() as conn:
        for user_id in ("alice", "bob"):
            conn.execute("INSERT INTO users VALUES(?,?,?,?)", (user_id, user_id, "not-used", db.now()))
            conn.execute("INSERT INTO profiles VALUES(?,?,?,?,?)", (user_id, user_id + "体质", "", user_id + "病史", db.now()))
            conn.execute("INSERT INTO user_permissions VALUES(?,'profile:read')", (user_id,))
    return db


def test_stdio_bound_identity_revocation_and_readonly_audit(identities):
    from app.mcp_server import create_server
    from app.profile_access import read_profile
    from mcp.server.mcpserver.exceptions import ToolError
    db = identities
    before = db.all_rows("SELECT * FROM profiles ORDER BY user_id")
    server = create_server("stdio", "alice")
    result = asyncio.run(server.call_tool("get_my_patient_profile", {}))
    assert not result.is_error
    assert result.structured_content == {"constitution": "alice体质", "allergies": "", "history": "alice病史"}
    with pytest.raises(PermissionError):
        read_profile("alice", "bob", transport="stdio")
    with pytest.raises(ToolError):
        asyncio.run(create_server("stdio").call_tool("get_my_patient_profile", {}))
    with db.connection() as conn:
        conn.execute("DELETE FROM user_permissions WHERE user_id='alice'")
    db.init_db()  # Restart must not silently restore a revoked scope.
    with pytest.raises(ToolError):
        asyncio.run(server.call_tool("get_my_patient_profile", {}))
    assert db.all_rows("SELECT * FROM profiles ORDER BY user_id") == before
    audit = db.all_rows("SELECT caller_id,patient_id,transport,result,created_at FROM profile_access_audit ORDER BY created_at")
    assert [row["result"] for row in audit] == ["success", "forbidden", "unauthenticated", "forbidden"]
    assert all(row["created_at"] and row["transport"] == "stdio" for row in audit)


def test_remote_mcp_protocol_authentication_and_identity(identities):
    from app.auth import make_token
    from app.mcp_server import create_server
    db = identities
    server = create_server("streamable-http", "bob")  # HTTP must ignore the local stdio binding.
    remote = server.streamable_http_app(stateless_http=True, json_response=True)
    headers = {"Accept": "application/json, text/event-stream", "MCP-Protocol-Version": "2025-06-18"}
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "get_my_patient_profile", "arguments": {}}}
    before = db.all_rows("SELECT * FROM profiles ORDER BY user_id")
    with TestClient(remote, base_url="http://127.0.0.1:28001") as client:
        assert client.post("/mcp", headers=headers, json=body).status_code == 401
        assert client.post("/mcp", headers={**headers, "Authorization": "Bearer invalid"}, json=body).status_code == 401
        for user_id in ("alice", "bob"):
            response = client.post("/mcp", headers={**headers, "Authorization": "Bearer " + make_token(user_id)}, json=body)
            assert response.status_code == 200, response.text
            result = response.json()["result"]
            assert not result.get("isError", False)
            assert result["structuredContent"]["history"] == user_id + "病史"
            assert set(result["structuredContent"]) == {"constitution", "allergies", "history"}
        with db.connection() as conn:
            conn.execute("DELETE FROM user_permissions WHERE user_id='alice'")
        response = client.post("/mcp", headers={**headers, "Authorization": "Bearer " + make_token("alice")}, json=body)
        assert response.json()["result"]["isError"]
        assert "bob病史" not in response.text
    assert db.all_rows("SELECT * FROM profiles ORDER BY user_id") == before
    audit = db.all_rows("SELECT caller_id,result FROM profile_access_audit WHERE transport='streamable-http' ORDER BY created_at")
    assert [row["result"] for row in audit] == ["unauthenticated", "success", "success", "forbidden"]
