import asyncio
import base64
import json
import logging
import os
import re
import time
import traceback
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import requests
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tripletex-agent")

GEMINI_API_KEY = "AIzaSyBI2N7eEJcn9-TUaizlJJMOuJtQIInOJHs"
TRIPLETEX_TIMEOUT_SECONDS = 30
EXPECTED_ENDPOINT_API_KEY = os.getenv("TRIPLETEX_ENDPOINT_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")
SPEC_PATH = Path(__file__).with_name("tripletex.json")
MAX_TOOL_CALLS = 24
MAX_ATTACHMENT_BYTES_FOR_MODEL = 8 * 1024 * 1024
MAX_TOOL_RESULT_CHARS = 8_000
REQUEST_TIME_BUDGET_SECONDS = int(os.getenv("REQUEST_TIME_BUDGET_SECONDS", "240"))
PRIMARY_TOOL_CALL_BUDGET = 8
ACTION_TOOL_CALL_BUDGET = 6

app = FastAPI(
    title="Tripletex AI Accounting Agent",
    version="1.0.0",
    description="Competition-ready endpoint for Tripletex accounting tasks.",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
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


@dataclass
class TaskRunReport:
    run_id: str
    task_prompt: str
    write_calls: int
    read_calls: int
    write_errors: int
    success: bool
    failure_reason: str | None = None
    summary: str | None = None


@dataclass
class PreparedAttachment:
    filename: str
    mime_type: str
    size_bytes: int
    content: bytes
    content_text_preview: str | None = None


@dataclass
class EndpointDoc:
    method: str
    path: str
    operation_id: str
    summary: str
    description: str
    tags: list[str]
    parameters: list[dict[str, Any]]
    request_body_schema: dict[str, Any] | None
    request_body_required: bool


TASK_FAMILY_HINTS: list[dict[str, Any]] = [
    {
        "keywords": ["employee", "ansatt", "administrator", "admin", "kontoadministrator"],
        "endpoints": [("POST", "/employee"), ("GET", "/employee")],
        "hint": "Employee creation or update task. Find existing employee first if this sounds like an update.",
    },
    {
        "keywords": ["customer", "kunde"],
        "endpoints": [("POST", "/customer"), ("GET", "/customer")],
        "hint": "Customer task. Create only if the customer does not already exist.",
    },
    {
        "keywords": ["product", "produkt", "item", "vare"],
        "endpoints": [("POST", "/product"), ("GET", "/product")],
        "hint": "Product task. Prefer one create call after checking if the product already exists.",
    },
    {
        "keywords": ["invoice", "faktura", "credit", "kreditnota"],
        "endpoints": [("POST", "/invoice"), ("GET", "/invoice"), ("POST", "/order"), ("GET", "/order")],
        "hint": "Invoice-like task. You may need an order/customer prerequisite before the invoice write.",
    },
    {
        "keywords": ["project", "prosjekt"],
        "endpoints": [("POST", "/project"), ("GET", "/project")],
        "hint": "Project task. Check whether a customer link is required.",
    },
    {
        "keywords": ["department", "avdeling", "hr", "sales", "finance"],
        "endpoints": [("POST", "/department"), ("GET", "/department")],
        "hint": "Department task. Search departments first if the prompt sounds like bookkeeping to an existing department.",
    },
    {
        "keywords": ["travel", "reise", "utlegg", "expense", "receipt", "kvittering", "bokfort", "bokfør"],
        "endpoints": [("POST", "/travelExpense"), ("GET", "/travelExpense"), ("GET", "/ledger/account")],
        "hint": "Expense or receipt task. Use receipt details and department/account lookups, then post the travel expense or voucher with correct VAT handling.",
    },
    {
        "keywords": ["voucher", "bilag", "ledger", "mva", "vat", "expense account", "utgiftskonto"],
        "endpoints": [("POST", "/ledger/voucher"), ("GET", "/ledger/account"), ("GET", "/department")],
        "hint": "Ledger/voucher bookkeeping task. Read ledger accounts first, then create the smallest correct voucher/write.",
    },
]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-zA-Z0-9_:/>.-]+", text.lower()) if len(token) > 1}


def compact_json(value: Any, *, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        rendered = str(value)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}\n... [truncated]"


def build_attachment_summary(attachments: list["PreparedAttachment"]) -> str:
    if not attachments:
        return "- No attachments provided"

    lines: list[str] = []
    for attachment in attachments:
        line = f"- {attachment.filename} ({attachment.mime_type}, {attachment.size_bytes} bytes)"
        if attachment.content_text_preview:
            preview = normalize_text(attachment.content_text_preview)[:300]
            line += f" preview={preview}"
        lines.append(line)
    return "\n".join(lines)


def guess_task_family_hints(prompt: str) -> list[dict[str, Any]]:
    prompt_lower = prompt.lower()
    matches: list[dict[str, Any]] = []
    for hint in TASK_FAMILY_HINTS:
        if any(keyword in prompt_lower for keyword in hint["keywords"]):
            matches.append(hint)
    return matches[:3]


def build_direct_endpoint_hints(catalog: "OpenAPICatalog", prompt: str) -> str:
    matched_hints = guess_task_family_hints(prompt)
    if not matched_hints:
        return "- No direct heuristic hint available"

    lines: list[str] = []
    for hint in matched_hints:
        lines.append(f"- {hint['hint']}")
        for method, path in hint["endpoints"][:3]:
            try:
                details = catalog.details(method, path)
            except ValueError:
                continue
            lines.append(
                f"  {details['method']} {details['path']} - {details['summary']}"
            )
    return "\n".join(lines)


def prune_for_model(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "[truncated-depth]"

    if isinstance(value, dict):
        pruned: dict[str, Any] = {}
        items = list(value.items())
        for key, child in items[:40]:
            pruned[key] = prune_for_model(child, depth=depth + 1)
        if len(items) > 40:
            pruned["_truncated_keys"] = len(items) - 40
        return pruned

    if isinstance(value, list):
        pruned_items = [prune_for_model(item, depth=depth + 1) for item in value[:20]]
        if len(value) > 20:
            pruned_items.append(f"... [{len(value) - 20} more items truncated]")
        return pruned_items

    if isinstance(value, str) and len(value) > 800:
        return f"{value[:800]}... [truncated]"

    return value


class OpenAPICatalog:
    def __init__(self, spec_path: Path):
        self.spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.schemas = self.spec.get("components", {}).get("schemas", {})
        self.endpoint_docs = self._build_endpoint_docs()
        self.endpoint_lookup = {(doc.method, doc.path): doc for doc in self.endpoint_docs}

    def _resolve_ref(self, ref: str) -> dict[str, Any]:
        if not ref.startswith("#/components/schemas/"):
            return {}
        schema_name = ref.rsplit("/", 1)[-1]
        return self.schemas.get(schema_name, {})

    def _resolve_schema(self, schema: dict[str, Any] | None) -> dict[str, Any]:
        if not schema:
            return {}
        if "$ref" in schema:
            return self._resolve_schema(self._resolve_ref(schema["$ref"]))
        if schema.get("type") == "array" and isinstance(schema.get("items"), dict):
            return {
                "type": "array",
                "items": self._resolve_schema(schema["items"]),
            }
        if "allOf" in schema:
            merged: dict[str, Any] = {"type": "object", "properties": {}, "required": []}
            for item in schema["allOf"]:
                resolved = self._resolve_schema(item)
                merged["properties"].update(resolved.get("properties", {}))
                merged["required"].extend(resolved.get("required", []))
            merged["required"] = sorted(set(merged["required"]))
            return merged
        return schema

    def _schema_excerpt(self, schema: dict[str, Any] | None, *, depth: int = 0) -> dict[str, Any]:
        resolved = self._resolve_schema(schema)
        if not resolved:
            return {}

        schema_type = resolved.get("type")
        if schema_type == "array":
            return {"type": "array", "items": self._schema_excerpt(resolved.get("items"), depth=depth + 1)}

        excerpt: dict[str, Any] = {}
        if schema_type:
            excerpt["type"] = schema_type
        if "enum" in resolved:
            excerpt["enum"] = resolved["enum"]
        if "format" in resolved:
            excerpt["format"] = resolved["format"]
        if "description" in resolved:
            excerpt["description"] = normalize_text(str(resolved["description"]))[:300]

        if schema_type == "object" and depth < 2:
            properties: dict[str, Any] = {}
            for name, child in list(resolved.get("properties", {}).items())[:20]:
                properties[name] = self._schema_excerpt(child, depth=depth + 1)
            excerpt["properties"] = properties
            required_fields = resolved.get("required", [])
            if required_fields:
                excerpt["required"] = required_fields

        return excerpt

    def _request_body_schema(self, operation: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
        request_body = operation.get("requestBody") or {}
        required = bool(request_body.get("required"))
        content = request_body.get("content") or {}
        for media_type in (
            "application/json",
            "application/json; charset=utf-8",
            "application/*+json",
        ):
            if media_type in content:
                return self._schema_excerpt(content[media_type].get("schema")), required

        if content:
            _, first = next(iter(content.items()))
            return self._schema_excerpt(first.get("schema")), required
        return None, required

    def _build_endpoint_docs(self) -> list[EndpointDoc]:
        docs: list[EndpointDoc] = []
        for path, path_item in self.spec.get("paths", {}).items():
            for method in ("get", "post", "put", "delete", "patch"):
                operation = path_item.get(method)
                if not operation:
                    continue

                parameters: list[dict[str, Any]] = []
                for parameter in operation.get("parameters", []):
                    schema = parameter.get("schema", {})
                    parameters.append(
                        {
                            "name": parameter.get("name"),
                            "in": parameter.get("in"),
                            "required": bool(parameter.get("required")),
                            "description": normalize_text(str(parameter.get("description", "")))[:240],
                            "type": schema.get("type"),
                            "format": schema.get("format"),
                        }
                    )

                request_body_schema, request_body_required = self._request_body_schema(operation)
                docs.append(
                    EndpointDoc(
                        method=method.upper(),
                        path=path,
                        operation_id=operation.get("operationId", ""),
                        summary=normalize_text(str(operation.get("summary", ""))),
                        description=normalize_text(str(operation.get("description", ""))),
                        tags=operation.get("tags", []),
                        parameters=parameters,
                        request_body_schema=request_body_schema,
                        request_body_required=request_body_required,
                    )
                )
        return docs

    def search(self, query: str, method: str | None = None, *, limit: int = 8) -> list[dict[str, Any]]:
        query_tokens = tokenize(query)
        filtered_method = method.upper() if method else None
        scored: list[tuple[float, EndpointDoc]] = []
        domain_prefix_boosts = {
            "employee": "/employee",
            "ansatt": "/employee",
            "customer": "/customer",
            "kunde": "/customer",
            "product": "/product",
            "produkt": "/product",
            "invoice": "/invoice",
            "faktura": "/invoice",
            "order": "/order",
            "ordre": "/order",
            "travel": "/travelExpense",
            "expense": "/travelExpense",
            "reise": "/travelExpense",
            "utlegg": "/travelExpense",
            "project": "/project",
            "prosjekt": "/project",
            "department": "/department",
            "avdeling": "/department",
            "ledger": "/ledger",
            "bilag": "/ledger",
            "voucher": "/ledger/voucher",
        }

        for doc in self.endpoint_docs:
            if filtered_method and doc.method != filtered_method:
                continue

            haystack = " ".join(
                [
                    doc.method,
                    doc.path,
                    doc.operation_id,
                    doc.summary,
                    doc.description,
                    " ".join(doc.tags),
                    " ".join(param["name"] for param in doc.parameters if param.get("name")),
                ]
            ).lower()
            haystack_tokens = tokenize(haystack)

            overlap = len(query_tokens & haystack_tokens)
            path_bonus = sum(1 for token in query_tokens if token in doc.path.lower())
            summary_bonus = sum(1 for token in query_tokens if token in doc.summary.lower())
            score = overlap * 3 + path_bonus * 2 + summary_bonus

            if query.lower() in haystack:
                score += 5

            for keyword, prefix in domain_prefix_boosts.items():
                if keyword in query_tokens and doc.path.startswith(prefix):
                    score += 8

            if score > 0:
                scored.append((score, doc))

        scored.sort(key=lambda item: (-item[0], item[1].method, item[1].path))

        results: list[dict[str, Any]] = []
        for _, doc in scored[:limit]:
            results.append(
                {
                    "method": doc.method,
                    "path": doc.path,
                    "summary": doc.summary,
                    "tags": doc.tags,
                    "operation_id": doc.operation_id,
                }
            )
        return results

    def details(self, method: str, path: str) -> dict[str, Any]:
        doc = self.endpoint_lookup.get((method.upper(), path))
        if not doc:
            raise ValueError(f"Unknown endpoint: {method.upper()} {path}")

        return {
            "method": doc.method,
            "path": doc.path,
            "operation_id": doc.operation_id,
            "summary": doc.summary,
            "description": doc.description,
            "tags": doc.tags,
            "parameters": doc.parameters,
            "request_body_required": doc.request_body_required,
            "request_body_schema": doc.request_body_schema,
        }


@lru_cache(maxsize=1)
def load_catalog() -> OpenAPICatalog:
    return OpenAPICatalog(SPEC_PATH)


class TripletexClient:
    def __init__(self, base_url: str, session_token: str, timeout: int = TRIPLETEX_TIMEOUT_SECONDS):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.auth = ("0", session_token)
        self.session.headers.update({"Accept": "application/json"})
        self.get_cache: dict[str, Any] = {}
        self.successful_write_fingerprints: set[str] = set()
        self.failed_write_fingerprints: set[str] = set()
        self.read_calls = 0
        self.write_calls = 0
        self.write_errors = 0

    def _fingerprint(self, method: str, path: str, params: dict[str, Any] | None, json_body: Any) -> str:
        return json.dumps(
            {
                "method": method.upper(),
                "path": path,
                "params": params or {},
                "body": json_body,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        method = method.upper()
        clean_path = f"/{path.lstrip('/')}"
        fingerprint = self._fingerprint(method, clean_path, params, json_body)

        if method == "GET" and fingerprint in self.get_cache:
            return self.get_cache[fingerprint]

        if method in {"POST", "PUT", "PATCH", "DELETE"}:
            if fingerprint in self.successful_write_fingerprints:
                return {
                    "skipped": True,
                    "reason": "Duplicate successful write avoided to minimize penalties.",
                }
            if fingerprint in self.failed_write_fingerprints:
                return {
                    "skipped": True,
                    "reason": "Duplicate failed write avoided to prevent another penalty.",
                }

        response: requests.Response | None = None
        last_error: Exception | None = None
        max_attempts = 3 if method == "GET" else 1

        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.request(
                    method=method,
                    url=f"{self.base_url}{clean_path}",
                    params=params,
                    json=json_body,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                last_error = exc
                if method == "GET" and attempt < max_attempts:
                    time.sleep(0.4 * attempt)
                    continue
                raise HTTPException(status_code=502, detail=f"Tripletex request failed: {exc}") from exc

            if method == "GET" and response.status_code in {429, 500, 502, 503, 504} and attempt < max_attempts:
                time.sleep(0.4 * attempt)
                continue
            break

        if response is None:
            raise HTTPException(status_code=502, detail=f"Tripletex request failed: {last_error}")

        if method == "GET":
            self.read_calls += 1
        else:
            self.write_calls += 1

        if response.status_code >= 400:
            if method != "GET":
                self.write_errors += 1
                self.failed_write_fingerprints.add(fingerprint)
            detail = self._extract_error(response)
            raise HTTPException(status_code=502, detail=f"Tripletex API error ({response.status_code}): {detail}")

        if response.status_code == 204 or not response.content:
            payload: Any = {}
        else:
            try:
                payload = response.json()
            except ValueError:
                payload = {"text": response.text}

        if method == "GET":
            self.get_cache[fingerprint] = payload
        elif method in {"POST", "PUT", "PATCH", "DELETE"}:
            self.successful_write_fingerprints.add(fingerprint)

        return payload

    def verify_connection(self) -> dict[str, Any]:
        payload = self.request(
            "GET",
            "/employee",
            params={"count": 1, "fields": "id,firstName,lastName,email"},
        )
        if not isinstance(payload, dict):
            raise HTTPException(status_code=502, detail="Unexpected Tripletex response shape during connection probe.")
        return payload

    def stats(self) -> dict[str, int]:
        return {
            "read_calls": self.read_calls,
            "write_calls": self.write_calls,
            "write_errors": self.write_errors,
        }

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
            validation_messages = payload.get("validationMessages")
            if validation_messages:
                return json.dumps(validation_messages, ensure_ascii=False)[:500]

        return json.dumps(payload, ensure_ascii=False)[:500]


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
        preview = decoded[:4_000].decode("utf-8", errors="replace")

    return PreparedAttachment(
        filename=attachment.filename,
        mime_type=attachment.mime_type,
        size_bytes=len(decoded),
        content=decoded,
        content_text_preview=preview,
    )


def build_gemini_client() -> genai.Client:
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing Gemini API key.")
    return genai.Client(api_key=api_key)


def build_system_instruction() -> str:
    return """
You are an elite accounting execution agent for the Tripletex competition.

Your goal is to maximize competition score while obeying all rules:
- Only use the provided Tripletex proxy base URL through the available tools.
- Minimize write calls aggressively. GET calls are free and should be used to verify state before risky writes.
- Avoid duplicate writes, trial-and-error, and speculative POST/PUT/DELETE calls.
- Use the API search and endpoint details tools before uncertain writes.
- You are latency-sensitive. Finish quickly and avoid unnecessary tool calls.
- Prompts may be in Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French.
- Attachments may contain critical facts. Use them when relevant.
- Prefer exact field values from the prompt or files. Do not invent missing data.
- When a task can be solved by updating or deleting an existing object, find the exact target with GET calls first.
- After completing the task, verify the resulting state with GET calls when needed, then stop.

Fast execution strategy:
1. Infer the task type immediately from the prompt and attachment summary.
2. Use the candidate endpoints already provided in the prompt before broader searching.
3. Only call search_api or get_api_details if the route is unclear.
4. Use GET calls to find prerequisite IDs and confirm targets.
5. Perform the fewest writes possible, usually one per entity actually required.
6. Verify only the critical final state, then stop.

You must use tools, not free-form reasoning, for any Tripletex action.
When done, answer with a very short plain-language summary.
""".strip()


def build_task_prompt(request: SolveRequest, attachments: list[PreparedAttachment], catalog: OpenAPICatalog) -> str:
    attachment_summary = build_attachment_summary(attachments)
    initial_candidates = catalog.search(request.prompt, limit=6)
    direct_hints = build_direct_endpoint_hints(catalog, request.prompt)

    return f"""
Task prompt:
{request.prompt}

Attachments:
{attachment_summary}

Candidate endpoints from the local OpenAPI search:
{compact_json(initial_candidates, max_chars=3_500)}

Direct heuristic endpoint hints:
{direct_hints}

Remember:
- You are working against a fresh Tripletex competition account unless the prompt implies otherwise.
- GET calls do not incur the efficiency penalty. Write calls do.
- The local API catalog comes from the provided `tripletex.json`.
- Prefer the listed candidate endpoints first.
- Use the tool docstrings carefully and only write when you are confident.
""".strip()


def build_contents(task_prompt: str, attachments: list[PreparedAttachment]) -> list[Any]:
    contents: list[Any] = [task_prompt]
    for attachment in attachments:
        if attachment.mime_type == "application/pdf":
            contents.append(
                f"PDF attachment available: {attachment.filename} ({attachment.size_bytes} bytes). "
                "Use the attachment only if the task clearly depends on it."
            )
            continue

        if attachment.size_bytes > MAX_ATTACHMENT_BYTES_FOR_MODEL:
            contents.append(
                f"Attachment {attachment.filename} was omitted from the multimodal payload because it exceeds "
                f"{MAX_ATTACHMENT_BYTES_FOR_MODEL} bytes. Rely on metadata and any text preview if available."
            )
            if attachment.content_text_preview:
                contents.append(
                    f"Text preview for {attachment.filename}:\n{attachment.content_text_preview[:4_000]}"
                )
            continue

        contents.append(
            types.Part.from_bytes(
                data=attachment.content,
                mime_type=attachment.mime_type,
            )
        )
    return contents


def build_action_prompt(request: SolveRequest, attachments: list[PreparedAttachment], catalog: OpenAPICatalog) -> str:
    return f"""
You have already spent read/tool budget and still have not written anything.

Your task now is to finish decisively.
- This competition task is expected to change state in Tripletex.
- Prefer at most 1-2 additional GET calls before the first write.
- If the exact endpoint family is clear, stop searching and act.
- If account/VAT handling is involved, use `GET /ledger/account` and the prompt/attachment cues, then write the result.
- If a department is named, search the department and include it in the write.
- Do not end without attempting the required write unless the prompt is truly impossible.

Task prompt:
{request.prompt}

Attachments:
{build_attachment_summary(attachments)}

Direct heuristic endpoint hints:
{build_direct_endpoint_hints(catalog, request.prompt)}
""".strip()


def summarize_task_prompt(prompt: str, *, max_length: int = 160) -> str:
    normalized = normalize_text(prompt)
    if len(normalized) <= max_length:
        return normalized
    return f"{normalized[:max_length]}..."


def log_task_report(report: TaskRunReport) -> None:
    logger.info("=== Task Result ===")
    logger.info("Run ID: %s", report.run_id)
    logger.info("Task: %s", summarize_task_prompt(report.task_prompt))
    logger.info("Success: %s", "yes" if report.success else "no")
    logger.info(
        "Tripletex calls: reads=%s writes=%s write_errors=%s",
        report.read_calls,
        report.write_calls,
        report.write_errors,
    )
    if report.failure_reason:
        logger.info("Failure reason: %s", report.failure_reason)
    if report.summary:
        logger.info("Model summary: %s", report.summary)
    logger.info("===================")


def parse_json_string(value: str | None, *, default: Any) -> Any:
    if value is None or not value.strip():
        return default
    parsed = json.loads(value)
    return parsed


async def execute_accounting_task(
    run_id: str,
    request: SolveRequest,
    client: TripletexClient,
    attachments: list[PreparedAttachment],
) -> str:
    catalog = load_catalog()
    gemini_client = build_gemini_client()

    logger.info(
        "Received task. run_id=%s prompt=%s attachments=%s",
        run_id,
        request.prompt,
        len(attachments),
    )

    task_prompt = build_task_prompt(request, attachments, catalog)

    def search_api(query: str, method: str = "", limit: int = 8) -> dict[str, Any]:
        """Search the local Tripletex OpenAPI catalog for relevant endpoints before making uncertain API calls."""
        safe_limit = max(1, min(limit, 15))
        matches = catalog.search(query=query, method=method or None, limit=safe_limit)
        return {"matches": matches}

    def get_api_details(method: str, path: str) -> dict[str, Any]:
        """Get detailed local documentation for a specific endpoint, including parameters and request body schema."""
        return catalog.details(method=method, path=path)

    def tripletex_get(path: str, params_json: str = "{}") -> dict[str, Any]:
        """Execute a read-only GET request against the provided Tripletex proxy. params_json must be a JSON object string."""
        params = parse_json_string(params_json, default={})
        payload = client.request("GET", path, params=params)
        return {"stats": client.stats(), "result": prune_for_model(payload)}

    def tripletex_write(method: str, path: str, params_json: str = "{}", body_json: str = "{}") -> dict[str, Any]:
        """Execute a write request against Tripletex. Use only for POST, PUT, PATCH, or DELETE once you are confident."""
        method_upper = method.upper()
        if method_upper not in {"POST", "PUT", "PATCH", "DELETE"}:
            raise ValueError("tripletex_write only supports POST, PUT, PATCH, or DELETE.")

        params = parse_json_string(params_json, default={})
        body = parse_json_string(body_json, default={})
        if method_upper == "DELETE" and body == {}:
            body = None

        payload = client.request(method_upper, path, params=params, json_body=body)
        return {"stats": client.stats(), "result": prune_for_model(payload)}

    tool_names = ["search_api", "get_api_details", "tripletex_get", "tripletex_write"]
    tool_map: dict[str, Callable[..., dict[str, Any]]] = {
        "search_api": search_api,
        "get_api_details": get_api_details,
        "tripletex_get": tripletex_get,
        "tripletex_write": tripletex_write,
    }

    def _run_model(prompt_text: str, allowed_tools: list[Any], allowed_tool_names: list[str], remote_call_budget: int) -> str:
        config = types.GenerateContentConfig(
            systemInstruction=build_system_instruction(),
            temperature=0.0,
            topP=0.8,
            maxOutputTokens=2_048,
            toolConfig=types.ToolConfig(
                functionCallingConfig=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.ANY,
                    allowedFunctionNames=allowed_tool_names,
                )
            ),
            tools=allowed_tools,
        )
        history: list[Any] = build_contents(prompt_text, attachments)
        remote_calls_used = 0
        read_only_rounds = 0
        no_call_rounds = 0
        last_text = ""

        while remote_calls_used < remote_call_budget:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=history,
                config=config,
            )

            candidate = response.candidates[0] if response.candidates else None
            content = candidate.content if candidate and candidate.content else None
            parts = content.parts if content and content.parts else []

            function_calls: list[Any] = []
            text_parts: list[str] = []
            for part in parts:
                if getattr(part, "text", None):
                    text_parts.append(part.text)
                function_call = getattr(part, "function_call", None)
                if function_call:
                    function_calls.append(function_call)

            if text_parts:
                last_text = "\n".join(text_parts).strip()

            if not function_calls:
                no_call_rounds += 1
                if client.write_calls == 0 and no_call_rounds <= 2:
                    history.append(
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text=(
                                        "This task requires a Tripletex state change. "
                                        "Use the available tools and make the needed write now."
                                    )
                                )
                            ],
                        )
                    )
                    continue
                return last_text

            if content:
                history.append(content)

            function_response_parts: list[types.Part] = []
            writes_before_round = client.write_calls
            for function_call in function_calls:
                if remote_calls_used >= remote_call_budget:
                    break

                tool = tool_map[function_call.name]
                try:
                    result = tool(**(function_call.args or {}))
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "error": str(exc),
                        "stats": client.stats(),
                    }
                function_response_parts.append(
                    types.Part.from_function_response(
                        name=function_call.name,
                        response=result,
                    )
                )
                remote_calls_used += 1

            if function_response_parts:
                history.append(types.Content(role="user", parts=function_response_parts))

            if client.write_calls == writes_before_round:
                read_only_rounds += 1
            else:
                read_only_rounds = 0

            if client.write_calls == 0 and read_only_rounds >= 2 and remote_calls_used < remote_call_budget:
                history.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=(
                                    "You are spending too many tool calls on reads. "
                                    "Stop exploring. If the endpoint family is clear, make the required write now. "
                                    "Use at most one more GET before the first write."
                                )
                            )
                        ],
                    )
                )

        return last_text

    summary = await asyncio.wait_for(
        asyncio.to_thread(
            _run_model,
            task_prompt,
            [search_api, get_api_details, tripletex_get, tripletex_write],
            tool_names,
            PRIMARY_TOOL_CALL_BUDGET,
        ),
        timeout=REQUEST_TIME_BUDGET_SECONDS,
    )

    if client.write_calls == 0:
        logger.warning("Primary pass completed without writes. Starting forced action pass.")
        remaining_budget = max(30, REQUEST_TIME_BUDGET_SECONDS // 3)
        action_prompt = build_action_prompt(request, attachments, catalog)
        forced_summary = await asyncio.wait_for(
            asyncio.to_thread(
                _run_model,
                action_prompt,
                [get_api_details, tripletex_get, tripletex_write],
                ["get_api_details", "tripletex_get", "tripletex_write"],
                ACTION_TOOL_CALL_BUDGET,
            ),
            timeout=remaining_budget,
        )
        if forced_summary:
            summary = forced_summary

    logger.info("Agent summary: %s", summary)
    logger.info("Tripletex call stats: %s", client.stats())
    return summary


@app.post("/solve", response_model=SolveResponse)
async def solve(
    request_body: SolveRequest,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    validate_endpoint_api_key(authorization)

    run_id = str(uuid.uuid4())[:8]
    attachments = [prepare_attachment(file) for file in request_body.files]
    tripletex_client = TripletexClient(
        base_url=request_body.tripletex_credentials.base_url,
        session_token=request_body.tripletex_credentials.session_token,
    )

    summary: str | None = None
    failure_reason: str | None = None
    try:
        summary = await execute_accounting_task(run_id, request_body, tripletex_client, attachments)
    except Exception:  # noqa: BLE001
        failure_reason = traceback.format_exc(limit=3)
        logger.exception("Unhandled execution error")

    success = tripletex_client.write_calls > 0 and tripletex_client.write_errors == 0
    if not success and not failure_reason:
        if tripletex_client.write_calls == 0:
            failure_reason = "The model completed without making any Tripletex write calls."
        elif tripletex_client.write_errors > 0:
            failure_reason = f"The model made {tripletex_client.write_errors} failing write call(s)."
        else:
            failure_reason = "The task did not reach a confident success state."

    log_task_report(
        TaskRunReport(
            run_id=run_id,
            task_prompt=request_body.prompt,
            write_calls=tripletex_client.write_calls,
            read_calls=tripletex_client.read_calls,
            write_errors=tripletex_client.write_errors,
            success=success,
            failure_reason=failure_reason,
            summary=summary,
        )
    )

    return JSONResponse(content=SolveResponse().model_dump())


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )
