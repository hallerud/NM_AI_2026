import asyncio
import base64
import json
import logging
import os
from typing import Any

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from pydantic import BaseModel, Field


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tripletex-agent")

TRIPLETEX_TIMEOUT_SECONDS = 30
EXPECTED_ENDPOINT_API_KEY = os.getenv("TRIPLETEX_ENDPOINT_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

app = FastAPI(
    title="Tripletex AI Accounting Agent",
    version="0.1.0",
    description="Competition endpoint skeleton for Tripletex accounting tasks.",
)


class FileAttachment(BaseModel):
    filename: str
    content_base64: str
    mime_type: str


class TripletexCredentials(BaseModel):
    base_url: str
    session_token: str


class SolveRequest(BaseModel):
    prompt: str
    files: list[FileAttachment] = Field(default_factory=list)
    tripletex_credentials: TripletexCredentials


class SolveResponse(BaseModel):
    status: str = "completed"


class PreparedAttachment(BaseModel):
    filename: str
    mime_type: str
    size_bytes: int
    content_text_preview: str | None = None


class TripletexClient:
    def __init__(self, base_url: str, session_token: str, timeout: int = TRIPLETEX_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.auth = ("0", session_token)
        self.session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | list[Any] | None = None,
    ) -> dict[str, Any]:
        response = self.session.request(
            method=method,
            url=f"{self.base_url}/{path.lstrip('/')}",
            params=params,
            json=json_body,
            timeout=self.timeout,
        )

        if response.status_code >= 400:
            detail = self._extract_error(response)
            raise HTTPException(status_code=502, detail=f"Tripletex API error ({response.status_code}): {detail}")

        if not response.content:
            return {}

        return response.json()

    def verify_connection(self) -> dict[str, Any]:
        return self.request(
            "GET",
            "/employee",
            params={
                "count": 1,
                "fields": "id,firstName,lastName,email",
            },
        )

    @staticmethod
    def _extract_error(response: requests.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return response.text[:500]

        if isinstance(payload, dict):
            for key in ("message", "errorMessage", "description", "developerMessage"):
                value = payload.get(key)
                if value:
                    return str(value)
        return json.dumps(payload)[:500]


def validate_endpoint_api_key(authorization: str | None) -> None:
    if not EXPECTED_ENDPOINT_API_KEY:
        return

    expected = f"Bearer {EXPECTED_ENDPOINT_API_KEY}"
    if authorization != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def prepare_attachment(attachment: FileAttachment) -> PreparedAttachment:
    try:
        decoded = base64.b64decode(attachment.content_base64, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail=f"Invalid base64 content for file '{attachment.filename}'",
        ) from exc

    preview: str | None = None
    if attachment.mime_type.startswith("text/"):
        preview = decoded[:2_000].decode("utf-8", errors="replace")

    return PreparedAttachment(
        filename=attachment.filename,
        mime_type=attachment.mime_type,
        size_bytes=len(decoded),
        content_text_preview=preview,
    )


def build_gemini_client() -> genai.Client | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY is not set; continuing without Gemini planning.")
        return None
    return genai.Client(api_key=api_key)


def build_planning_prompt(request: SolveRequest, attachments: list[PreparedAttachment]) -> str:
    attachment_lines = [
        f"- {attachment.filename} ({attachment.mime_type}, {attachment.size_bytes} bytes)"
        for attachment in attachments
    ]
    attachment_summary = "\n".join(attachment_lines) if attachment_lines else "- No attachments provided"

    return f"""
You are building a plan for a Tripletex accounting agent.

User task:
{request.prompt}

Available attachments:
{attachment_summary}

Return a concise action plan describing:
1. The likely task category.
2. The Tripletex endpoints that will likely be needed.
3. Any missing information that must be derived from attachments or looked up in Tripletex.
4. The order of operations.

Do not invent data that is not present.
""".strip()


async def create_execution_plan(
    request: SolveRequest,
    attachments: list[PreparedAttachment],
) -> str | None:
    client = build_gemini_client()
    if client is None:
        return None

    prompt = build_planning_prompt(request, attachments)

    def _generate() -> str:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
        return getattr(response, "text", "") or ""

    try:
        plan = await asyncio.to_thread(_generate)
    except Exception:  # noqa: BLE001
        logger.exception("Gemini planning failed")
        return None

    return plan.strip() or None


async def execute_accounting_task(
    request: SolveRequest,
    client: TripletexClient,
    attachments: list[PreparedAttachment],
) -> None:
    connection_snapshot = await asyncio.to_thread(client.verify_connection)
    planning_note = await create_execution_plan(request, attachments)

    logger.info(
        "Received task. prompt=%s attachments=%s employee_probe_count=%s",
        request.prompt,
        len(attachments),
        len(connection_snapshot.get("values", [])),
    )

    if planning_note:
        logger.info("Gemini plan:\n%s", planning_note)

    # Skeleton only:
    # 1. Parse multilingual instructions and attachment contents.
    # 2. Select the relevant Tripletex operations.
    # 3. Execute the writes through the provided proxy base_url.
    # 4. Verify the resulting state before returning success.


@app.post("/solve", response_model=SolveResponse)
async def solve(
    request_body: SolveRequest,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    validate_endpoint_api_key(authorization)

    attachments = [prepare_attachment(file) for file in request_body.files]
    tripletex_client = TripletexClient(
        base_url=request_body.tripletex_credentials.base_url,
        session_token=request_body.tripletex_credentials.session_token,
    )

    await execute_accounting_task(request_body, tripletex_client, attachments)
    return JSONResponse(content=SolveResponse().model_dump())


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
