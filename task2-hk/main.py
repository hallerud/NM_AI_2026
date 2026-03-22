import asyncio
import base64
import json
import os
import time
import traceback
import uuid
from datetime import date
from urllib.parse import parse_qsl

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

app = FastAPI()

DEFAULT_DATE_FROM = "2020-01-01"
DEFAULT_DATE_TO = "2030-12-31"
DEFAULT_INVOICE_QUERY = {"invoiceDateFrom": DEFAULT_DATE_FROM, "invoiceDateTo": DEFAULT_DATE_TO}
DEFAULT_ORDER_QUERY = {"orderDateFrom": DEFAULT_DATE_FROM, "orderDateTo": DEFAULT_DATE_TO}
DEFAULT_VOUCHER_QUERY = {"dateFrom": DEFAULT_DATE_FROM, "dateTo": DEFAULT_DATE_TO}

ACCOUNTING_HEAVY_TASKS = {"Bank Reconciliation", "Supplier Invoice", "Voucher", "Department", "Depreciation"}
INVOICE_TASKS = {"Invoice", "Credit Note", "Bank Reconciliation"}
SUPPLIER_TASKS = {"Supplier", "Supplier Invoice", "Voucher", "Bank Reconciliation", "Department"}
TRAVEL_TASKS = {"Travel Expense"}
PROJECT_TASKS = {"Project", "Timesheet"}
SALARY_TASKS = {"Salary/Payroll"}
ASSET_TASKS = {"Depreciation"}


# ── Tripletex API helper ──────────────────────────────────────────────────────

def _is_nullish(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() in ("", "null", "None"):
        return True
    return False


def _prune_nullish(value):
    """Remove null-like values so optional fields are omitted instead of sent as null."""
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            if _is_nullish(v):
                continue
            pruned = _prune_nullish(v)
            if pruned in ({}, []):
                continue
            cleaned[k] = pruned
        return cleaned
    if isinstance(value, list):
        cleaned = []
        for item in value:
            if _is_nullish(item):
                continue
            pruned = _prune_nullish(item)
            if pruned in ({}, []):
                continue
            cleaned.append(pruned)
        return cleaned
    return value


def _find_nullish_paths(value, prefix=""):
    paths = []
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            if _is_nullish(v):
                paths.append(path)
            else:
                paths.extend(_find_nullish_paths(v, path))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            path = f"{prefix}[{i}]"
            if _is_nullish(item):
                paths.append(path)
            else:
                paths.extend(_find_nullish_paths(item, path))
    return paths


def _decode_text_attachment(data: bytes) -> str | None:
    for encoding in ("utf-8", "utf-8-sig", "iso-8859-1", "cp1252"):
        try:
            return data.decode(encoding)
        except Exception:
            continue
    return None


def _split_endpoint_and_query(endpoint: str) -> tuple[str, dict]:
    path, sep, query = endpoint.partition("?")
    if not sep or not query:
        return endpoint, {}
    return path, dict(parse_qsl(query, keep_blank_values=True))


def _apply_safe_default_params(method: str, endpoint: str, params: dict) -> dict:
    """Auto-fill harmless required filters for free read endpoints."""
    params = dict(params or {})
    ep = endpoint.rstrip("/")
    if method.upper() != "GET":
        return params

    if ep.endswith("/ledger/voucher"):
        params.setdefault("dateFrom", DEFAULT_DATE_FROM)
        params.setdefault("dateTo", DEFAULT_DATE_TO)
    elif ep.endswith("/invoice"):
        params.setdefault("invoiceDateFrom", DEFAULT_DATE_FROM)
        params.setdefault("invoiceDateTo", DEFAULT_DATE_TO)
    elif ep.endswith("/order"):
        params.setdefault("orderDateFrom", DEFAULT_DATE_FROM)
        params.setdefault("orderDateTo", DEFAULT_DATE_TO)
    return params


def _missing_body_feedback(endpoint: str) -> dict:
    ep = endpoint.rstrip("/")
    guidance = {
        "/ledger/voucher": {
            "message": "POST /ledger/voucher requires a JSON payload in body with date, description, and balanced postings. Do not leave body empty.",
            "example": {
                "method": "POST",
                "endpoint": "/ledger/voucher",
                "body": {
                    "date": "2026-03-22",
                    "description": "Bank reconciliation or supplier invoice",
                    "postings": [
                        {"row": 1, "date": "2026-03-22", "account": {"id": "<ID of account #6340 from ledger_accounts>"}, "amountGross": 39800, "amountGrossCurrency": 39800},
                        {"row": 2, "date": "2026-03-22", "account": {"id": "<ID of account #2710 from ledger_accounts>"}, "amountGross": 9950, "amountGrossCurrency": 9950},
                        {"row": 3, "date": "2026-03-22", "account": {"id": "<ID of account #1920 from ledger_accounts>"}, "amountGross": -49750, "amountGrossCurrency": -49750},
                    ],
                },
            },
        },
        "/order": {
            "message": "POST /order requires a JSON payload in body with customer, orderDate, deliveryDate, VAT mode, and all orderLines.",
        },
        "/travelExpense": {
            "message": "POST /travelExpense requires a JSON payload in body. Do not leave body empty.",
        },
    }
    return guidance.get(ep, {
        "message": f"{ep} requires a JSON payload in the tool body. Do not put the payload in endpoint or params.",
    })

def tx(method: str, base_url: str, token: str, endpoint: str,
       params: dict = None, body: dict = None, bank_account: str = None) -> dict:
    m = method.upper()
    endpoint_path, endpoint_query = _split_endpoint_and_query(endpoint)
    ep = endpoint_path
    is_action = ep.split("/")[-1].startswith(":")

    # ── Local pre-flight checks — return early without hitting the API ──────────
    params = _prune_nullish({**endpoint_query, **(params or {})})
    params = _apply_safe_default_params(m, endpoint_path, params)
    body = _prune_nullish(body)

    def _missing_date(param_name: str) -> bool:
        v = params.get(param_name)
        return _is_nullish(v)

    # 1. POST/PUT to a non-action endpoint without a body
    if m in ("POST", "PUT") and not is_action and not body:
        return {"status_code": 422, "data": {
            "message": "Validation failed",
            "validationMessages": [{"field": "request body", "message": "Kan ikke være null."}],
        }}

    # 2. PUT /:invoice — needs bank account AND invoiceDate query param
    if m == "PUT" and is_action and ep.endswith(":invoice"):
        if not bank_account:
            return {"status_code": 422, "data": {
                "message": "Validering feilet.",
                "validationMessages": [{"message": "Faktura kan ikke opprettes før selskapet har registrert et bankkontonummer."}],
            }}
        if _missing_date("invoiceDate"):
            return {"status_code": 422, "data": {
                "message": "Validation failed",
                "validationMessages": [{"field": "invoiceDate", "message": "Kan ikke være null."}],
            }}

    # 3. PUT /:createCreditNote — requires ?date= query param
    if m == "PUT" and is_action and ep.endswith(":createCreditNote"):
        if _missing_date("date"):
            return {"status_code": 422, "data": {
                "message": "Validation failed",
                "validationMessages": [{"field": "date", "message": "Kan ikke være null. Add ?date=YYYY-MM-DD as a query param."}],
            }}

    # 4. GET /ledger/voucher — requires dateFrom and dateTo
    if m == "GET" and ep.rstrip("/").endswith("/ledger/voucher"):
        missing = [f for f in ("dateFrom", "dateTo") if _missing_date(f)]
        if missing:
            return {"status_code": 422, "data": {
                "message": "Validation failed",
                "validationMessages": [{"field": f, "message": "Kan ikke være null."} for f in missing],
            }}

    # 5. GET /invoice — requires invoiceDateFrom and invoiceDateTo
    if m == "GET" and ep.rstrip("/").endswith("/invoice"):
        missing = [f for f in ("invoiceDateFrom", "invoiceDateTo") if _missing_date(f)]
        if missing:
            return {"status_code": 422, "data": {
                "message": "Validation failed",
                "validationMessages": [{"field": f, "message": "Kan ikke være null."} for f in missing],
            }}

    # 6. GET /order — requires orderDateFrom and orderDateTo
    if m == "GET" and ep.rstrip("/").endswith("/order"):
        missing = [f for f in ("orderDateFrom", "orderDateTo") if _missing_date(f)]
        if missing:
            return {"status_code": 422, "data": {
                "message": "Validation failed",
                "validationMessages": [{"field": f, "message": "Kan ikke være null."} for f in missing],
            }}

    # 7. Bodyless action endpoints still require certain query params
    if m == "PUT" and is_action and ep.endswith(":payment"):
        missing = [f for f in ("paymentDate", "paymentTypeId", "paidAmount") if _is_nullish(params.get(f))]
        if missing:
            return {"status_code": 422, "data": {
                "message": "Validation failed",
                "validationMessages": [{"field": f, "message": "Kan ikke være null."} for f in missing],
            }}

    if m == "PUT" and is_action and ep.endswith(":send") and _is_nullish(params.get("sendType")):
        return {"status_code": 422, "data": {
            "message": "Validation failed",
            "validationMessages": [{"field": "sendType", "message": "Kan ikke være null."}],
        }}

    # ── HTTP request with 429 retry ───────────────────────────────────────────
    for attempt in range(3):
        try:
            r = requests.request(
                method=m,
                url=f"{base_url}{endpoint_path}",
                auth=("0", token),
                params=params,
                json=body if m in ("POST", "PUT") else None,
                timeout=60,
            )
            try:
                data = r.json()
            except Exception:
                data = r.text

            if r.status_code == 429 and attempt < 2:
                wait = int(r.headers.get("Retry-After", 20))
                print(f"  ⏳ 429 Too Many Requests — waiting {wait}s (attempt {attempt + 1}/3)…")
                time.sleep(wait)
                continue

            return {"status_code": r.status_code, "data": data}

        except Exception as e:
            return {"status_code": 0, "error": str(e)}

    return {"status_code": 429, "data": {"message": "Rate limit: all retries exhausted."}}


# ── Pre-fetch context ─────────────────────────────────────────────────────────

def prefetch(base_url: str, token: str, task_type: str = "Unknown") -> dict:
    auth = ("0", token)
    include_all = task_type == "Unknown"

    def get(path, params=None):
        try:
            r = requests.get(f"{base_url}{path}", auth=auth, params=params, timeout=15)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def vals(path, params):
        d = get(path, params)
        return d.get("values", []) if d else []

    ctx = {}

    who = get("/token/session/>whoAmI", {"fields": "employeeId,companyId"})
    if who:
        ctx["employee_id"] = who["value"]["employeeId"]
        ctx["company_id"] = who["value"]["companyId"]

    company = get(f"/company/{ctx.get('company_id', 0)}", {"fields": "id,name,bankAccountNumber"})
    if company:
        c = company["value"]
        ctx["company_name"] = c.get("name")
        ctx["bank_account"] = c.get("bankAccountNumber")

    emp = get(f"/employee/{ctx.get('employee_id', 0)}", {"fields": "firstName,lastName,email"})
    if emp:
        e = emp["value"]
        ctx["employee_name"] = f"{e.get('firstName', '')} {e.get('lastName', '')}".strip()

    ctx["vat_types"]       = [{"id": v["id"], "number": v.get("number"), "name": v.get("name"), "pct": v.get("percentage")}
                               for v in vals("/ledger/vatType", {"fields": "id,number,name,percentage", "count": 100})[:20]]
    ctx["customers"]       = vals("/customer",             {"fields": "id,name,organizationNumber,email", "count": 50})
    ctx["employees"]       = vals("/employee",             {"fields": "id,firstName,lastName,email", "count": 50})
    ctx["departments"]     = vals("/department",           {"fields": "id,name,departmentNumber", "count": 50})
    ctx["payment_types"]   = vals("/invoice/paymentType",  {"fields": "id,description,displayName,currencyCode,isInactive"}) if include_all or task_type in INVOICE_TASKS or task_type in TRAVEL_TASKS or task_type in ACCOUNTING_HEAVY_TASKS else []
    ctx["suppliers"]       = vals("/supplier",             {"fields": "id,name,organizationNumber,email", "count": 50}) if include_all or task_type in SUPPLIER_TASKS else []
    ctx["products"]        = vals("/product",              {"fields": "id,name,priceExcludingVatCurrency", "count": 50}) if include_all or task_type in {"Invoice", "Product"} else []
    ctx["invoices"]        = vals("/invoice",              {"fields": "id,invoiceNumber,customer,amount,amountOutstanding", "count": 30, **DEFAULT_INVOICE_QUERY}) if include_all or task_type in INVOICE_TASKS else []
    ctx["orders"]          = vals("/order",                {"fields": "id,number,customer,orderDate", "count": 30, **DEFAULT_ORDER_QUERY}) if include_all or task_type in INVOICE_TASKS else []
    ctx["vouchers"]        = vals("/ledger/voucher",       {"fields": "id,number,date,description", "count": 30, **DEFAULT_VOUCHER_QUERY}) if include_all or task_type in ACCOUNTING_HEAVY_TASKS else []
    ctx["currencies"]      = vals("/currency",             {"fields": "id,code,displayName", "count": 50}) if include_all or task_type in ACCOUNTING_HEAVY_TASKS or task_type in INVOICE_TASKS else []
    ctx["projects"]        = vals("/project",              {"fields": "id,name,number,customer,projectManager", "count": 50}) if include_all or task_type in PROJECT_TASKS else []
    ctx["travel_expenses"] = vals("/travelExpense",        {"fields": "id,title,employee,status", "count": 30}) if include_all or task_type in TRAVEL_TASKS else []
    ctx["assets"]          = vals("/asset",                {"fields": "id,name,accountNumber,acquisitionCost,depreciationRate", "count": 50}) if include_all or task_type in ASSET_TASKS else []
    ctx["ledger_accounts"] = vals("/ledger/account", {"fields": "id,number,name,isInactive", "count": 1000}) if include_all or task_type in ACCOUNTING_HEAVY_TASKS else []
    ctx["salary_types"]    = vals("/salary/type",    {"fields": "id,number,name", "count": 100}) if include_all or task_type in SALARY_TASKS else []

    return ctx


def _render_section(title: str, items: list, formatter) -> str:
    if not items:
        return f"### {title}\n  (none)"
    return f"### {title}\n" + "\n".join(f"  {formatter(item)}" for item in items)


def _build_context_text(prompt: str, ctx: dict, today: str, task_type: str) -> str:
    sections = [
        "## Task",
        prompt,
        "",
        "## Session",
        f"- Employee ID: {ctx.get('employee_id')} ({ctx.get('employee_name')})",
        f"- Company: {ctx.get('company_id')} — {ctx.get('company_name') or 'unknown'}",
        f"- Bank account: {ctx.get('bank_account') or 'not set'}",
        f"- Today: {today}",
        "",
        _render_section("VAT Types", ctx.get("vat_types", []), lambda v: f"ID {v['id']}: #{v['number']} {v['name']} ({v['pct']}%)"),
        "",
        _render_section("Customers", ctx.get("customers", []), lambda c: f"ID {c['id']}: {c.get('name')} (org: {c.get('organizationNumber')}, email: {c.get('email')})"),
        "",
        _render_section("Employees", ctx.get("employees", []), lambda e: f"ID {e['id']}: {e.get('firstName')} {e.get('lastName')} ({e.get('email')})"),
        "",
        _render_section("Departments", ctx.get("departments", []), lambda d: f"ID {d['id']}: {d.get('name')} (#{d.get('departmentNumber')})"),
    ]

    if task_type == "Unknown" or task_type in INVOICE_TASKS or task_type in TRAVEL_TASKS or task_type in ACCOUNTING_HEAVY_TASKS:
        sections.extend([
            "",
            _render_section("Payment Types", ctx.get("payment_types", []), lambda p: f"ID {p['id']}: {p.get('description') or p.get('displayName')} currency={p.get('currencyCode')} inactive={p.get('isInactive')}"),
        ])

    if task_type == "Unknown" or task_type in {"Invoice", "Product"}:
        sections.extend([
            "",
            _render_section("Products", ctx.get("products", []), lambda p: f"ID {p['id']}: {p.get('name')} (price excl: {p.get('priceExcludingVatCurrency')})"),
        ])

    if task_type == "Unknown" or task_type in INVOICE_TASKS:
        sections.extend([
            "",
            _render_section("Invoices", ctx.get("invoices", []), lambda i: f"ID {i['id']}: #{i.get('invoiceNumber')} amount={i.get('amount')} outstanding={i.get('amountOutstanding')}"),
            "",
            _render_section("Orders", ctx.get("orders", []), lambda o: f"ID {o['id']}: #{o.get('number')} customer={o.get('customer', {}).get('name') if isinstance(o.get('customer'), dict) else '?'}"),
        ])

    if task_type == "Unknown" or task_type in SUPPLIER_TASKS:
        sections.extend([
            "",
            _render_section("Suppliers", ctx.get("suppliers", []), lambda s: f"ID {s['id']}: {s.get('name')} (org: {s.get('organizationNumber')}, email: {s.get('email')})"),
        ])

    if task_type == "Unknown" or task_type in PROJECT_TASKS:
        sections.extend([
            "",
            _render_section("Projects", ctx.get("projects", []), lambda p: f"ID {p['id']}: {p.get('name')}"),
        ])

    if task_type == "Unknown" or task_type in TRAVEL_TASKS:
        sections.extend([
            "",
            _render_section("Travel Expenses", ctx.get("travel_expenses", []), lambda t: f"ID {t['id']}: {t.get('title')} status={t.get('status')}"),
        ])

    if task_type == "Unknown" or task_type in ASSET_TASKS:
        sections.extend([
            "",
            _render_section("Assets", ctx.get("assets", []), lambda a: f"ID {a['id']}: {a.get('name')} (acq cost: {a.get('acquisitionCost')})"),
        ])

    if task_type == "Unknown" or task_type in SALARY_TASKS:
        sections.extend([
            "",
            _render_section("Salary Types", ctx.get("salary_types", []), lambda s: f"ID {s['id']}: #{s.get('number')} {s.get('name')}"),
        ])

    if task_type == "Unknown" or task_type in ACCOUNTING_HEAVY_TASKS:
        sections.extend([
            "",
            _render_section("Vouchers", ctx.get("vouchers", []), lambda v: f"ID {v['id']}: #{v.get('number')} {v.get('date')} {v.get('description')}"),
            "",
            f"### Ledger Accounts ({len(ctx.get('ledger_accounts', []))} total)",
            "\n".join(f"  ID {a['id']}: #{a.get('number')} {a.get('name')}" for a in ctx.get("ledger_accounts", [])) or "  (none)",
        ])

    if task_type == "Unknown" or task_type in ACCOUNTING_HEAVY_TASKS or task_type in INVOICE_TASKS:
        sections.extend([
            "",
            _render_section("Currencies", ctx.get("currencies", []), lambda c: f"ID {c['id']}: {c.get('code')} {c.get('displayName')}"),
        ])

    sections.extend(["", "---", "Plan your steps, then execute efficiently."])
    return "\n".join(sections)


# ── Tool ──────────────────────────────────────────────────────────────────────

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="tripletex_api",
        description=(
            "Make a request to the Tripletex v2 REST API. "
            "endpoint must start with / (e.g. /employee, /invoice). "
            "Prefer passing query parameters in params instead of appending them to endpoint, "
            "but if query params are already present in endpoint they will still be used. "
            "Use Tripletex resource IDs in bodies; for voucher postings account.id must be the ledger account resource ID, not the account number. "
            "Action endpoints use colon syntax: PUT /order/{id}/:invoice, "
            "PUT /invoice/{id}/:send, PUT /invoice/{id}/:payment, "
            "PUT /invoice/{id}/:createCreditNote, PUT /asset/{id}/:depreciate."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "method":   types.Schema(
                    type=types.Type.STRING,
                    enum=["GET", "POST", "PUT", "DELETE"],
                    description="HTTP method. Use PUT for Tripletex updates/action endpoints; do not use PATCH."
                ),
                "endpoint": types.Schema(
                    type=types.Type.STRING,
                    description="API path starting with /. Prefer the path only, without query string. Example: /invoice/123/:payment"
                ),
                "params":   types.Schema(
                    type=types.Type.OBJECT,
                    description="Query parameters only. Use this for paymentDate, paymentTypeId, dateFrom/dateTo, sendType, invoiceDate, etc."
                ),
                "body":     types.Schema(
                    type=types.Type.OBJECT,
                    description="JSON body for POST/PUT to normal endpoints. Omit body for action endpoints like /:payment, /:send, /:invoice, /:createCreditNote unless explicitly required."
                ),
            },
            required=["method", "endpoint"],
        ),
    )
])]


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(today: str) -> str:
    return f"""You are an expert Tripletex (Norwegian ERP) accounting agent.
Tasks may be in Norwegian Bokmal, Nynorsk, English, German, Spanish, Portuguese, or French.
Today: {today}

## Goal
Maximize correctness first. Efficiency bonus depends on perfect correctness with few write calls and zero failed write calls.
GET calls are free. Use them to verify before writing.

## Core Rules
1. Read the task and any attachment carefully before writing anything.
2. Use pre-fetched context first to avoid duplicates and unnecessary calls.
3. Never send null/None/"null"/empty strings. Omit unknown optional fields entirely.
4. For normal POST/PUT requests, the JSON payload MUST go in body. Query parameters MUST go in params.
5. For action endpoints such as /:invoice, /:payment, /:send, /:createCreditNote, /:reverse: use query params and no body unless explicitly stated.
6. All dates use YYYY-MM-DD.
7. organizationNumber is always a STRING.
8. Bank account numbers must be 11 digits with separators removed.
9. If you are unsure of resource fields, do a free GET with fields=* before the write call.
10. If the first call returns 401 or an auth-related 403, stop and report the token problem.

## IDs And Updates
- Use Tripletex resource IDs, not human numbers, unless the API explicitly wants a number field.
- Important: account.id means the ledger account RESOURCE ID from pre-fetched ledger_accounts, not the account number like 1920 or 6340.
- For PUT updates to existing entities: GET first, then PUT with id, version, and the required fields plus your changes.

## Verified Query Defaults
- GET /invoice requires invoiceDateFrom and invoiceDateTo. If unknown, use {DEFAULT_DATE_FROM} to {DEFAULT_DATE_TO}.
- GET /order requires orderDateFrom and orderDateTo. If unknown, use {DEFAULT_DATE_FROM} to {DEFAULT_DATE_TO}.
- GET /ledger/voucher requires dateFrom and dateTo. If unknown, use {DEFAULT_DATE_FROM} to {DEFAULT_DATE_TO}.

## Verified Action Endpoints
- PUT /order/{{id}}/:invoice requires params.invoiceDate and no body.
- PUT /invoice/{{id}}/:payment requires params.paymentDate, params.paymentTypeId, params.paidAmount, and usually params.paidAmountCurrency. No body.
- PUT /invoice/{{id}}/:send requires params.sendType and no body.
- PUT /invoice/{{id}}/:createCreditNote requires params.date and no body.
- PUT /ledger/voucher/{{id}}/:reverse requires params.date and no body.

## Common Workflows
### Employee
- POST /employee with firstName, lastName, email, userType:"STANDARD", department:{{id}}.
- POST /employee/employment with employee:{{id}}, startDate, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver".
- POST /employee/employment/details with employment:{{id}}, date, percentageOfFullTimeEquivalent, annualSalary or hourlyWage, and occupationCode:{{id}}.
- occupationCode must be an object with a numeric id from GET /employee/employment/occupationCode.

### Customer And Supplier
- POST /customer with name, isCustomer:true, organizationNumber.
- POST /supplier with name, isSupplier:true, organizationNumber.

### Product
- POST /product with name and either priceExcludingVatCurrency or priceIncludingVatCurrency.

### Invoice And Payment
- To invoice from an order, create/find prerequisites, POST /order with all orderLines included in the initial body, then PUT /order/{{id}}/:invoice with params.invoiceDate={today}.
- VAT-exclusive prices use isPrioritizeAmountsIncludingVat:false with unitPriceExcludingVatCurrency.
- VAT-inclusive prices use isPrioritizeAmountsIncludingVat:true with unitPriceIncludingVatCurrency.

### Project And Timesheet
- POST /project with name, projectManager:{{id}}, startDate, customer:{{id}}, isInternal:false.
- Activities can be listed with GET /activity. POST /timesheet/entry needs employee, activity, project, date, hours.

### Voucher, Supplier Invoice, Bank Reconciliation
- POST /ledger/voucher with date, description, and balanced postings.
- Each posting needs row, date, account:{{id}}, amountGross, amountGrossCurrency.
- Postings can also include dimensions such as department:{{id}}, supplier:{{id}}, customer:{{id}}, project:{{id}} when the task requires them.
- If the task says the expense belongs to a department, include the correct department id on the relevant voucher posting(s).
- Postings must sum to zero. Positive is debit, negative is credit.
- Use ledger account RESOURCE IDs from pre-fetched ledger_accounts.
- Supplier invoice / expense voucher pattern: expense + VAT + bank/cash, balanced in one voucher.
- Bank reconciliation pattern: customer receipts should usually be matched with PUT /invoice/:payment. Supplier or other outgoing bank items without a Tripletex invoice usually become vouchers.
- Read CSV/text attachments carefully before creating vouchers.

### Travel Expense
- POST /travelExpense with employee, title, travelDetails.
- POST /travelExpense/perDiemCompensation with travelExpense and location.
- POST /travelExpense/cost with travelExpense, amountCurrencyIncVat, paymentType.

### Fixed Assets
- POST /asset with name, dateOfAcquisition, acquisitionCost, account:{{id}}, depreciationAccount:{{id}}, lifetime, depreciationMethod.
- PUT /asset/{{id}}/:depreciate uses a body with date and amount.

## Stop Conditions
- If a task appears unsupported or repeated GET attempts only return 404/403/500, stop and explain what you tried.
- Finish with DONE and a short summary when the task is complete.
"""


# ── Agent loop ────────────────────────────────────────────────────────────────

def _to_dict(obj):
    """Recursively convert proto-like objects to plain dicts."""
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=str))


def detect_task_type(prompt: str) -> str:
    p = prompt.lower()
    checks = [
        (["reiseregning", "travel expense", "note de frais", "gastos de viaje", "reisekost"], "Travel Expense"),
        (["kreditnota", "credit note", "gutschrift", "nota de crédito", "note de crédit"],     "Credit Note"),
        (["bankavst", "bank reconciliation", "bankutskrift", "avstem", "bankabstimmung", "conciliación bancaria"], "Bank Reconciliation"),
        (["leverandørfaktura", "leverandorfaktura", "supplier invoice", "lieferantenrechnung", "factura de proveedor", "facture fournisseur"], "Supplier Invoice"),
        (["avskrivning", "depreciation", "abschreibung", "amortissement", "depreciación"],     "Depreciation"),
        (["faktura", "invoice", "rechnung", "factura", "facture"],                             "Invoice"),
        (["ansatt", "tilsett", "employee", "empleado", "mitarbeiter", "employé", "ansette"],   "Employee"),
        (["prosjekt", "project", "proyecto", "projet", "projekt"],                             "Project"),
        (["avdeling", "department", "abteilung", "département", "departamento"],               "Department"),
        (["leverandør", "leverandor", "supplier", "lieferant", "fournisseur", "proveedor"],    "Supplier"),
        (["timesheet", "timeregistr", "timer ", "tidsregistr", "timeliste", "horas trabajadas"], "Timesheet"),
        (["lønn", "lonn", "payroll", "salary", "lohn", "salaire", "nómina"],                   "Salary/Payroll"),
        (["bilag", "voucher", "beleg", "pièce comptable"],                                     "Voucher"),
        (["produkt", "product", "producto", "produit"],                                        "Product"),
        (["kunde", "customer", "cliente", "client"],                                           "Customer"),
    ]
    for keywords, label in checks:
        if any(k in p for k in keywords):
            return label
    return "Unknown"


def task_verdict(stats: dict) -> tuple:
    """Returns (verdict_label, efficiency_note)."""
    stop = stats["stop_reason"]
    errors = stats["write_errors"]
    writes = stats["write_calls"]
    if "timeout" in stop:
        return "TIMED OUT ⏱", "no efficiency bonus"
    if stop == "max_iterations":
        return "MAX ITERATIONS HIT ⚠️", "likely incomplete — task may be wrong"
    if stop.startswith("gemini_error"):
        return "LLM ERROR ❌", stop
    # agent_done
    if errors == 0 and writes <= 2:
        return "CLEAN ✅", "full efficiency bonus likely"
    if errors == 0 and writes <= 5:
        return "COMPLETED ✅", f"minor efficiency loss ({writes} writes)"
    if errors == 0:
        return "COMPLETED ⚠️", f"efficiency loss — {writes} writes is excessive"
    return "SLOPPY ❌", f"{errors} failed or blocked write(s) — efficiency bonus likely lost"


def run_agent(prompt: str, files: list, base_url: str, token: str, ctx: dict, rid: str = "", task_type: str = ""):
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=api_key)
    start = time.time()
    today = date.today().isoformat()

    # File parts
    parts = []
    for f in files:
        mime = f.get("mime_type", "")
        data = base64.b64decode(f["content_base64"])
        if mime.startswith("image/") or mime == "application/pdf":
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=data)))
        else:
            text = _decode_text_attachment(data)
            if text is not None:
                parts.append(types.Part(text=f"File: {f.get('filename', '')}\n{text}"))

    ctx_text = _build_context_text(prompt, ctx, today, task_type)

    parts.append(types.Part(text=ctx_text))
    contents = [types.Content(role="user", parts=parts)]

    tag = f"[{rid}] " if rid else ""

    # ── Per-task stats tracking ───────────────────────────────────────────────
    stats = {
        "iterations": 0,
        "write_calls": 0,   # POST/PUT/DELETE — these cost efficiency score
        "write_errors": 0,  # Failed or blocked writes
        "read_errors": 0,   # Read-side 4xx/5xx for debugging only
        "calls": [],        # (method, endpoint, status, error_msg)
        "stop_reason": "max_iterations",
    }

    WRITE_METHODS = {"POST", "PUT", "DELETE"}
    consecutive_404s = 0
    consecutive_write_errors = 0
    fatal_stop = False
    read_cache = {}

    for i in range(15):
        elapsed = time.time() - start
        if elapsed > 200:
            stats["stop_reason"] = f"timeout ({elapsed:.0f}s)"
            print(f"{tag}  ⏱ {elapsed:.0f}s — hard stop.")
            break

        stats["iterations"] = i + 1
        print(f"\n{tag}── Iter {i+1} ({elapsed:.0f}s) " + "─" * 40)

        try:
            resp = client.models.generate_content(
                model="gemini-3.1-pro-preview",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=_build_system_prompt(today),
                    tools=TOOLS,
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(
                            mode="AUTO",
                        )
                    ),
                    max_output_tokens=8192,
                ),
            )
        except Exception as e:
            print(f"  Gemini error: {e}")
            if "quota" in str(e).lower() or "rate" in str(e).lower():
                print("  ⚠ Rate limited — waiting 30s...")
                time.sleep(30)
                continue
            stats["stop_reason"] = f"gemini_error: {e}"
            break

        candidate = resp.candidates[0]
        response_parts = candidate.content.parts

        for part in response_parts:
            if getattr(part, "text", None) and part.text.strip():
                # Show full agent reasoning — helps understand decisions
                print(f"{tag}  💭 {part.text.strip()[:300]}")

        # Collect function calls
        fn_calls = [p for p in response_parts if p.function_call]
        if not fn_calls:
            stats["stop_reason"] = "agent_done"
            print(f"\n{tag}  ✅ Agent signalled completion.")
            break

        # Add model turn to history
        contents.append(candidate.content)

        # Execute each tool call and collect responses
        tool_response_parts = []
        for part in fn_calls:
            fc = part.function_call
            inp = _to_dict(fc.args) or {}
            method   = inp.get("method", "GET")
            endpoint = inp.get("endpoint", "")
            endpoint, endpoint_query = _split_endpoint_and_query(endpoint)
            params   = _prune_nullish({**endpoint_query, **(_to_dict(inp.get("params")) or {})})
            params   = _apply_safe_default_params(method, endpoint, params)
            body     = _prune_nullish(_to_dict(inp.get("body")))

            is_write = method.upper() in WRITE_METHODS
            call_icon = "✏️ " if is_write else "🔍"
            print(f"{tag}  {call_icon} {method} {endpoint}", end="")
            if body:
                print(f"  body={json.dumps(body)[:200]}", end="")
            print()

            null_paths = _find_nullish_paths(_to_dict(inp.get("params"))) + _find_nullish_paths(_to_dict(inp.get("body")))
            if null_paths:
                print(f"{tag}  ℹ️ Stripped nullish fields before request: {', '.join(null_paths[:8])}")

            # Guard: POST/PUT without a body will cause 422 "Kan ikke være null"
            if method.upper() in ("POST", "PUT") and not body:
                # Exception: action endpoints like /:invoice, /:payment, /:send etc. are bodyless by design
                is_action = endpoint.split("?")[0].split("/")[-1].startswith(":")
                if not is_action:
                    print(f"{tag}  🛑 Blocked {method} {endpoint} — body is null/empty (would 422)")
                    feedback = _missing_body_feedback(endpoint)
                    result = {
                        "status_code": 422,
                        "data": {
                            "message": "Validation failed",
                            "developerMessage": feedback["message"],
                            "validationMessages": [{"field": "body", "message": "Kan ikke være null. Put the JSON payload in body."}],
                            "example": feedback.get("example"),
                        },
                    }
                    stats["write_errors"] += 1
                    stats["calls"].append({
                        "method": method,
                        "endpoint": endpoint,
                        "status": 422,
                        "error": "Validation failed",
                        "is_write": is_write,
                    })
                    if is_write:
                        consecutive_write_errors += 1
                    tool_response_parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": json.dumps(result)},
                        )
                    ))
                    if consecutive_write_errors >= 3:
                        stats["stop_reason"] = "fatal: 3 consecutive local write validation failures"
                        print(f"{tag}  🛑 Fatal: 3 consecutive blocked writes — stopping to avoid a retry loop.")
                        fatal_stop = True
                        break
                    continue

            cache_key = None
            if method.upper() == "GET":
                cache_key = json.dumps({"endpoint": endpoint, "params": params}, ensure_ascii=False, sort_keys=True, default=str)
                cached = read_cache.get(cache_key)
                if cached is not None:
                    result = cached
                    status = result.get("status_code", 0)
                    print(f"{tag}     ♻️ cache hit [{status}]")
                else:
                    result = tx(method, base_url, token, endpoint, params=params, body=body,
                                bank_account=ctx.get("bank_account"))
                    read_cache[cache_key] = result
                    status = result.get("status_code", 0)
            else:
                result = tx(method, base_url, token, endpoint, params=params, body=body,
                            bank_account=ctx.get("bank_account"))
                status = result.get("status_code", 0)
                if status < 400:
                    read_cache.clear()

            if is_write:
                stats["write_calls"] += 1

            # Parse error details fully for human readability
            error_summary = None
            if status >= 400:
                if is_write:
                    stats["write_errors"] += 1
                else:
                    stats["read_errors"] += 1
                d = result.get("data", {})
                error_parts = []
                if isinstance(d, dict):
                    for key in ("message", "developerMessage"):
                        if d.get(key):
                            error_parts.append(d[key])
                    for v in (d.get("validationMessages") or []):
                        msg = v.get("message", "") if isinstance(v, dict) else str(v)
                        field = v.get("field", "") if isinstance(v, dict) else ""
                        error_parts.append(f"  field={field!r}: {msg}" if field else f"  {msg}")
                error_summary = " | ".join(error_parts[:1]) if error_parts else f"HTTP {status}"
                status_icon = "🚫" if status == 403 else "❌"
                print(f"{tag}     {status_icon} {status} — {error_parts[0][:300] if error_parts else ''}")
                for ep in error_parts[1:]:
                    print(f"{tag}        {ep[:200]}")
            else:
                # Show created/updated ID on success for write calls
                if is_write and status in (200, 201):
                    d = result.get("data", {})
                    created_id = None
                    if isinstance(d, dict):
                        v = d.get("value", {})
                        if isinstance(v, dict):
                            created_id = v.get("id")
                    id_str = f" → id={created_id}" if created_id else ""
                    print(f"{tag}     ✅ {status}{id_str}")
                else:
                    print(f"{tag}     ✅ {status}")

            stats["calls"].append({
                "method": method,
                "endpoint": endpoint,
                "status": status,
                "error": error_summary,
                "is_write": is_write,
            })

            # ── Early-exit checks ─────────────────────────────────────────────
            if status == 404 and not is_write:
                consecutive_404s += 1
            else:
                consecutive_404s = 0

            if is_write and status >= 400:
                consecutive_write_errors += 1
            elif is_write:
                consecutive_write_errors = 0

            if status >= 400:
                d = result.get("data", {})
                err_text = ""
                if isinstance(d, dict):
                    validation_msgs = " ".join(str(v.get("message", "")) for v in (d.get("validationMessages") or []) if isinstance(v, dict))
                    err_text = (str(d.get("message", "")) + " " + str(d.get("developerMessage", "")) + " " + validation_msgs).lower()

                # Fatal: 401 Unauthorized — token missing or expired, nothing will work
                if status == 401:
                    stats["stop_reason"] = "fatal: 401 unauthorized"
                    print(f"{tag}  🛑 Fatal: 401 Unauthorized — stopping.")
                    fatal_stop = True
                    break

                # Fatal: sandbox has no bank account — no point retrying
                if "bank account" in err_text or "bankkonto" in err_text:
                    stats["stop_reason"] = "fatal: no bank account"
                    print(f"{tag}  🛑 Fatal: bank account error — stopping.")
                    fatal_stop = True
                    break

                # Fatal: token invalid on first call (only if it's an auth error, not a feature-access error)
                if status == 403 and stats["iterations"] <= 1:
                    is_feature_403 = "permission to access this feature" in err_text or "feature" in err_text
                    if not is_feature_403:
                        stats["stop_reason"] = "fatal: 403 on first call"
                        print(f"{tag}  🛑 Fatal: 403 on first call — token invalid, stopping.")
                        fatal_stop = True
                        break

            # Fatal: 3+ consecutive 404s on unknown endpoints (e.g. dimensions)
            if consecutive_404s >= 3:
                stats["stop_reason"] = "fatal: repeated 404s — unknown endpoint"
                print(f"{tag}  🛑 Fatal: 3 consecutive 404s — stopping (unknown endpoint).")
                fatal_stop = True
                break

            # Fatal: 3+ consecutive write errors — agent is stuck in a retry loop
            if consecutive_write_errors >= 3:
                stats["stop_reason"] = "fatal: 3 consecutive write errors — stuck"
                print(f"{tag}  🛑 Fatal: 3 consecutive failed writes — stopping to avoid penalty.")
                fatal_stop = True
                break

            r_str = json.dumps(result, ensure_ascii=False, default=str)
            if len(r_str) > 6000:
                d = result.get("data")
                if isinstance(d, dict) and "values" in d:
                    trunc = {**d, "values": d["values"][:5],
                             "_note": f"Showing 5/{len(d['values'])}. Use filters."}
                    r_str = json.dumps({"status_code": status, "data": trunc}, ensure_ascii=False, default=str)
                r_str = r_str[:6000] + "…[TRUNCATED]" if len(r_str) > 6000 else r_str

            tool_response_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response={"result": r_str},
                )
            ))

        contents.append(types.Content(role="user", parts=tool_response_parts))

        if fatal_stop:
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    from collections import Counter
    elapsed = time.time() - start
    all_writes    = [c for c in stats["calls"] if c["is_write"]]
    failed_writes = [c for c in all_writes if c["status"] >= 400]
    ok_writes     = [c for c in all_writes if c["status"] < 400]
    verdict_label, efficiency_note = task_verdict(stats)

    W = 58
    sep = "═" * W
    thin = "─" * W
    print(f"\n{tag}{sep}")
    header = f" RESULT  [{rid}]  ·  {task_type or 'Task'}  ·  {elapsed:.0f}s  ·  {stats['iterations']}/15 iter"
    print(f"{tag}{header}")
    print(f"{tag}{thin}")
    print(f"{tag}  VERDICT   : {verdict_label}")
    print(f"{tag}  Efficiency: {efficiency_note}")
    print(f"{tag}  Writes    : {stats['write_calls']} total  ✅ {len(ok_writes)} ok  ❌ {len(failed_writes)} failed/blocked  |  write errors: {stats['write_errors']}  read errors: {stats['read_errors']}")

    if all_writes:
        print(f"{tag}{thin}")
        print(f"{tag}  Write log (in order):")
        for c in all_writes:
            icon = "✅" if c["status"] < 400 else "❌"
            ep = c["endpoint"][:50]
            err_str = f"  → {c['error'][:70]}" if c.get("error") else ""
            print(f"{tag}    {icon} {c['method']:<7} {ep:<50} [{c['status']}]{err_str}")

    # Repeated failures = missing knowledge in system prompt
    fail_eps = Counter(
        f"{c['method']} {c['endpoint'].split('?')[0]}"
        for c in stats["calls"] if c["status"] >= 400 and c["is_write"]
    )
    if fail_eps:
        print(f"{tag}{thin}")
        print(f"{tag}  FIX IN PROMPT (repeated failed writes):")
        for ep, cnt in fail_eps.most_common():
            print(f"{tag}    {cnt}×  {ep}")

    print(f"{tag}{sep}")


# ── FastAPI endpoint ──────────────────────────────────────────────────────────

@app.post("/solve")
async def solve(request: Request):
    try:
        body = await request.json()
        prompt   = body["prompt"]
        files    = body.get("files", [])
        creds    = body["tripletex_credentials"]
        base_url = creds["base_url"]
        token    = creds["session_token"]

        rid = uuid.uuid4().hex[:6]
        task_type = detect_task_type(prompt)
        short_prompt = prompt[:120].strip() + ("…" if len(prompt) > 120 else "")
        print(f"\n[{rid}] ══ NEW TASK · {task_type} ══")
        print(f"[{rid}]    \"{short_prompt}\"")
        print(f"[{rid}]    Files: {len(files)}")

        ctx = await asyncio.to_thread(prefetch, base_url, token, task_type)
        print(f"[{rid}] ── Context: {ctx.get('employee_name')} | {ctx.get('company_name')} | "
              f"customers={len(ctx.get('customers',[]))} employees={len(ctx.get('employees',[]))} "
              f"accounts={len(ctx.get('ledger_accounts',[]))}")

        await asyncio.to_thread(run_agent, prompt, files, base_url, token, ctx, rid, task_type)

        return JSONResponse({"status": "completed"})
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
        return JSONResponse({"status": "completed"})
