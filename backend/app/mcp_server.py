"""Read-only knowledge and caller-bound patient profile tools over stdio or authenticated HTTP."""

from __future__ import annotations

import json
import os
import argparse

from fastapi import HTTPException
from pydantic import BaseModel
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp_types import ToolAnnotations

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .db import all_rows, init_db, one
from .retrieval import retrieve_with_status
from .safety import check_herbs
from .auth import token_identity
from .profile_access import read_profile, audit


class PatientProfile(BaseModel):
    constitution: str
    allergies: str
    history: str

class QiWenTokenVerifier:
    async def verify_token(self, token):
        try:
            user, claims = token_identity(token)
        except HTTPException:
            audit(None, None, "streamable-http", "unauthenticated")
            return None
        scopes = [row["scope"] for row in all_rows("SELECT scope FROM user_permissions WHERE user_id=?", (user["id"],))]
        return AccessToken(token=token, client_id=user["id"], subject=user["id"], scopes=scopes,
                           expires_at=claims["exp"], resource=os.getenv("QIWEN_MCP_RESOURCE_URL", "http://127.0.0.1:28001/mcp"))


def create_server(transport="stdio", bound_user_id=None):
    options = {}
    if transport == "streamable-http":
        options = {"token_verifier": QiWenTokenVerifier(), "auth": AuthSettings(
            issuer_url=os.getenv("QIWEN_MCP_ISSUER_URL", "http://127.0.0.1:23000"),
            resource_server_url=os.getenv("QIWEN_MCP_RESOURCE_URL", "http://127.0.0.1:28001/mcp"),
            required_scopes=[], validate_token_resource=True)}
    server = MCPServer("岐问知识库", **options)
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False)
    for function in (search_tcm_classics, lookup_herb, search_formulas, check_herb_safety):
        server.tool(annotations=readonly)(function)

    @server.tool(annotations=readonly)
    def get_my_patient_profile() -> PatientProfile:
        """Read only the authenticated caller's permitted profile fields; no patient ID argument."""
        if transport == "stdio":
            caller = bound_user_id
        else:
            token = get_access_token()
            caller = token.subject if token else None
        try:
            return PatientProfile(**read_profile(caller, caller, transport=transport))
        except PermissionError as exc:
            raise ToolError(str(exc)) from None
    return server


def search_tcm_classics(query: str) -> list[dict]:
    """Search classical text and return traceable excerpts."""
    if not query.strip():
        return []
    docs, _warning = retrieve_with_status(query[:200], limit=5)
    return [{"title": row["title"], "chapter": row["chapter"], "text": row["body"][:800]}
            for row in docs]


def lookup_herb(name: str) -> dict:
    """Look up a herb in the bundled public knowledge seed."""
    row = one("SELECT name,herb_id,properties,effects,source,structured_dosage,processing,source_ids,verification_status FROM herbs WHERE name=?", (name.strip()[:100],))
    if row:
        row["dosage"] = json.loads(row.pop("structured_dosage") or '{"min":null,"max":null,"unit":"g"}')
        row["source_ids"] = json.loads(row["source_ids"] or "[]")
    return row or {"error": "未找到"}


def search_formulas(query: str) -> list[dict]:
    """Search named formulas and their recorded source."""
    term = query.strip()[:100]
    if not term:
        return []
    escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    rows = all_rows(
        "SELECT name,structured_herbs,indication,source FROM formulas "
        "WHERE name LIKE ? ESCAPE '\\' OR indication LIKE ? ESCAPE '\\' LIMIT 10",
        (f"%{escaped}%", f"%{escaped}%"),
    )
    return [{key: value for key, value in row.items() if key != "structured_herbs"} | {"herbs": json.loads(row["structured_herbs"] or "[]")} for row in rows]


def check_herb_safety(herbs: list[str]) -> dict:
    """Check named herbs against the bundled compatibility rules."""
    errors = check_herbs(herbs)
    return {"safe": not errors, "violations": errors,
            "compatibility_status": "failed" if errors else "passed", "dosage_status": "unverified",
            "warning": "未命中内置规则不代表临床安全，仍需医师或中药师审核。"}


mcp = create_server("stdio", os.getenv("QIWEN_MCP_USER_ID", "").strip() or None)


def main() -> None:
    init_db()
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=28001)
    args = parser.parse_args()
    server = create_server(args.transport, os.getenv("QIWEN_MCP_USER_ID", "").strip() or None)
    if args.transport == "stdio":
        server.run(transport="stdio")
    else:
        import uvicorn
        uvicorn.run(server.streamable_http_app(stateless_http=True, json_response=True, host=args.host),
                    host=args.host, port=args.port)


if __name__ == "__main__":
    main()
