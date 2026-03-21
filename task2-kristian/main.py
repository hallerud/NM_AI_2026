import asyncio
import base64
import json
import os
import time
import traceback
import uuid
from datetime import date

import requests
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types

app = FastAPI()
TODAY = date.today().isoformat()


# ── Tripletex API helper ──────────────────────────────────────────────────────

def tx(method: str, base_url: str, token: str, endpoint: str,
       params: dict = None, body: dict = None) -> dict:
    try:
        r = requests.request(
            method=method.upper(),
            url=f"{base_url}{endpoint}",
            auth=("0", token),
            params=params,
            json=body if method.upper() in ("POST", "PUT", "PATCH") else None,
            timeout=60,
        )
        try:
            data = r.json()
        except Exception:
            data = r.text
        return {"status_code": r.status_code, "data": data}
    except Exception as e:
        return {"status_code": 0, "error": str(e)}


# ── Pre-fetch context ─────────────────────────────────────────────────────────

def prefetch(base_url: str, token: str) -> dict:
    auth = ("0", token)

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
    ctx["payment_types"]   = vals("/invoice/paymentType",  {"fields": "id,description"})
    ctx["customers"]       = vals("/customer",             {"fields": "id,name,organizationNumber,email", "count": 50})
    ctx["employees"]       = vals("/employee",             {"fields": "id,firstName,lastName,email", "count": 50})
    ctx["products"]        = vals("/product",              {"fields": "id,name,priceExcludingVatCurrency", "count": 50})
    ctx["invoices"]        = vals("/invoice",              {"fields": "id,invoiceNumber,customer,amount,amountOutstanding", "count": 30})
    ctx["orders"]          = vals("/order",                {"fields": "id,number,customer,orderDate", "count": 30})
    ctx["departments"]     = vals("/department",           {"fields": "id,name,departmentNumber", "count": 50})
    ctx["projects"]        = vals("/project",              {"fields": "id,name,number,customer,projectManager", "count": 50})
    ctx["travel_expenses"] = vals("/travelExpense",        {"fields": "id,title,employee,status", "count": 30})
    ctx["assets"]          = vals("/asset",                {"fields": "id,name,accountNumber,acquisitionCost,depreciationRate", "count": 50})

    # Fetch all ledger accounts at once — avoids one-by-one lookups during agent loop
    ctx["ledger_accounts"] = vals("/ledger/account", {"fields": "id,number,name,isInactive", "count": 1000})

    return ctx


# ── Tool ──────────────────────────────────────────────────────────────────────

TOOLS = [types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="tripletex_api",
        description=(
            "Make a request to the Tripletex v2 REST API. "
            "endpoint must start with / (e.g. /employee, /invoice). "
            "Action endpoints use colon syntax: PUT /order/{id}/:invoice, "
            "PUT /invoice/{id}/:send, PUT /invoice/{id}/:payment, "
            "PUT /invoice/{id}/:createCreditNote, PUT /asset/{id}/:depreciate."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "method":   types.Schema(type=types.Type.STRING, enum=["GET", "POST", "PUT", "DELETE"]),
                "endpoint": types.Schema(type=types.Type.STRING),
                "params":   types.Schema(type=types.Type.OBJECT),
                "body":     types.Schema(type=types.Type.OBJECT),
            },
            required=["method", "endpoint"],
        ),
    )
])]


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""You are an expert Tripletex (Norwegian ERP) accounting agent.
Complete tasks using the Tripletex v2 REST API. Tasks may be in any of: NO, EN, DE, ES, PT, FR.
Today: {TODAY}

## SCORING — READ THIS FIRST
Your score = correctness × tier multiplier × efficiency bonus.
The efficiency bonus (up to 2×) only applies on PERFECT correctness. It is hurt by:
  - Extra write calls (POST/PUT/DELETE/PATCH) — even successful ones
  - Any 4xx error on a write call

GET calls are FREE — read as much as needed, never guess.
Strategy: plan thoroughly using pre-fetched data → execute with minimum writes → zero retries.

## RULES
1. Plan ALL steps using pre-fetched context BEFORE making any write calls.
2. Use pre-fetched data — check it before creating anything (avoid duplicates).
3. organizationNumber is always a STRING.
4. Every write must succeed first time — if unsure of fields, GET with ?fields=* first (free).
5. Bank account numbers: strip ALL separators (dots, spaces, dashes) → exactly 11 digits. E.g. "1234.56.78901" → "12345678901".
6. If you get 403 on the very first call: the token is invalid — stop immediately, do not retry.
7. Say DONE with a summary when finished.

## API
- Single entity: response.data.value | Lists: response.data.values
- POST returns: response.data.value.id
- All dates: YYYY-MM-DD | Discover fields: ?fields=*

## PUT — UNIVERSAL RULE (applies to ALL PUT endpoints)
ANY PUT to an existing entity requires:
  1. GET the entity first to get the current version number and ALL existing field values
  2. Send ALL fields back (not just the ones you want to change) + the version number
  3. Omit version or any required field → 422 "Kan ikke være null"
This applies to: PUT /employee, PUT /project, PUT /customer, PUT /supplier, PUT /ledger/account, etc.
Exception: action endpoints (/:invoice, /:payment, /:depreciate, /:reverse) — these take NO body unless stated.

## LEDGER ACCOUNTS
The pre-fetched context contains ALL ledger accounts (active AND inactive). Search it before calling GET /ledger/account.
- If an account has isInactive:true → reactivate it first: PUT /ledger/account/{{id}} with {{id, version:0, isInactive:false, number, name, type}}
- If an account is missing entirely, create it: POST /ledger/account {{number (INTEGER, not string), name, type, vatType:{{id:0}}}}
Valid types: ASSETS, EQUITY_AND_DEBT, OPERATING_REVENUES, OPERATING_EXPENSES, FINANCIAL_INCOME_AND_EXPENSES, TAX_ON_ORDINARY_ACTIVITY

## ENDPOINTS

### Employee — FULL FLOW
Step 1 — POST /employee:
  Required: firstName, lastName, email, userType (always use "STANDARD"), department:{{id}}
  Optional: dateOfBirth (YYYY-MM-DD), nationalIdentityNumber (personnummer), bankAccountNumber, phoneNumberMobile

Step 2 — POST /employee/employment (creates the employment record):
  Required: employee:{{id}}, startDate, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver"
  Do NOT include salary or employment% here — those go in Step 3.

Step 3 — POST /employee/employment/details (adds salary + position to the employment):
  Required: employment:{{id}} (from Step 2), date (same as startDate)
  Fields: employmentPercentage (e.g. 80.0), annualWage (årslønn) OR monthlyWage (månedslønn),
          occupationCode (stillingskode, 7-digit string e.g. "2511000"), jobTitle
  IMPORTANT: use employmentPercentage (NOT percentageOfFullTimeEquivalent) and annualWage (NOT annualSalary).

Step 4 — Updating an employee: GET /employee/{{id}} first → PUT /employee/{{id}} with ALL fields + version.
  PUT body must include: {{id, version, firstName, lastName, email, userType, ...fields to update}}

Version conflicts (409): always re-fetch the entity to get latest version before PUT.

### Customer / Supplier
POST /customer — name, isCustomer:true; organizationNumber (STRING)
POST /supplier — name, isSupplier:true; organizationNumber (STRING)
NOTE: bankAccountNumber is NOT a valid field on supplier — omit it.
GET /invoice ALWAYS requires dateFrom AND dateTo — without them it returns 422. Use a wide range if unknown: dateFrom=2020-01-01&dateTo=2030-12-31.

### Product
POST /product — name; priceExcludingVatCurrency or priceIncludingVatCurrency; optional: number (product number as STRING), vatType:{{id}}
CHECK pre-fetched products list before creating — if name already exists, use the existing product.

### Customer Invoice — ACTION ENDPOINTS TAKE NO BODY
IMPORTANT: If PUT /order/{{id}}/:invoice fails with "bank account" error — stop immediately, do not retry. Sandbox limitation.
1. Find/create customer (check pre-fetched customers first)
2. Find/create product (check pre-fetched products first)
3. POST /order — customer, orderDate, deliveryDate, isPrioritizeAmountsIncludingVat,
   orderLines:[{{product:{{id}}, count, unitPriceExcludingVatCurrency|unitPriceIncludingVatCurrency}}]
4. PUT /order/{{id}}/:invoice — NO body, NO params needed → returns invoice
5. If "send": PUT /invoice/{{id}}/:send?sendType=EMAIL — NO body

### Payment
PUT /invoice/{{id}}/:payment — body: {{paymentDate, paymentTypeId, paidAmount, paidAmountCurrency}}
  paymentTypeId: use ID from pre-fetched payment_types (not a nested object, it's a plain integer)

### Credit Note
PUT /invoice/{{id}}/:createCreditNote — NO body

### Project — FULL FLOW
POST /project — name, projectManager:{{id}} (use logged-in employee ID), startDate, customer:{{id}}, isInternal:false
  Optional: budget (decimal), fixedprice (decimal), isFixedPrice:true/false

PUT /project/{{id}} to update (set fixed price, etc.):
  REQUIRED: GET /project/{{id}} first to get version + all fields, then PUT with ALL fields + version + your changes.
  Example body: {{id, version, name, startDate, customer:{{id}}, projectManager:{{id}}, isInternal, isFixedPrice:true, fixedprice:429500, ...}}

### Activity + Timesheets
Activities are GLOBAL — not sub-resources of projects:
  GET /activity?fields=id,name,isProjectActivity — list existing activities
  POST /activity — name, isProjectActivity:true
  Existing activity id:0 ("Generell") works for most cases — use it to avoid extra writes.

Hour registration: POST /timesheet/entry — employee:{{id}}, activity:{{id}}, project:{{id}}, date (YYYY-MM-DD), hours (decimal)

### Department
POST /department — name, departmentNumber (unique int)

### Voucher (manual journal / supplier invoice)
POST /ledger/voucher — REQUIRED top-level fields: date (YYYY-MM-DD), description, postings:[...]
MISSING date at voucher level → 422 "Kan ikke være null". Always set it.
Each posting MUST include "row" (1-indexed, never 0) and "date" matching the voucher date:
  {{"row":1, "date":"YYYY-MM-DD", "account":{{"id":X}}, "amountGross":N, "amountGrossCurrency":N}}
RULES:
- amountGross MUST equal amountGrossCurrency (always, even for NOK)
- Postings MUST sum to zero. Positive=debit, Negative=credit.
- NEVER include vatType in postings — Tripletex rejects it ("system-generated row")
- NEVER post directly to account 2400 (Leverandørgjeld) or 2600 (Skattetrekk) — system-managed control accounts.
- Post all lines manually: expense net + VAT input + bank/clearing account

Supplier invoice/payment example (net 39800, 25% VAT = 9950, total 49750):
  {{"row":1,"date":"YYYY-MM-DD","account":{{"id":<6340_id>}},"amountGross":39800,"amountGrossCurrency":39800}}
  {{"row":2,"date":"YYYY-MM-DD","account":{{"id":<2710_id>}},"amountGross":9950,"amountGrossCurrency":9950}}
  {{"row":3,"date":"YYYY-MM-DD","account":{{"id":<1920_id>}},"amountGross":-49750,"amountGrossCurrency":-49750}}

Bank reconciliation — customer payment already covered by PUT /invoice/:payment.
For supplier payments without existing supplier invoices, post expense vs 1920 (bank account).
For tax/skattetrekk entries, use account 2600 only as the CREDIT side, debiting 1920.

### Fixed Assets & Depreciation
GET /asset — list existing assets
POST /asset — name, acquisitionCost, acquisitionDate, accountNumber (asset account e.g. 1200),
  depreciationAccountNumber (accumulated depreciation e.g. 1209),
  depreciationRate (annual % e.g. 12.5 for 8yr linear = 100/8)
PUT /asset/{{id}}/:depreciate — body: {{date:"YYYY-MM-DD", amount:DEPRECIATION_AMOUNT}}
  This creates the voucher automatically. DO NOT post depreciation as manual vouchers.

### Travel Expense
POST /travelExpense — employee:{{id}}, title, travelDetails:{{"isForeignTravel":false}}

Per diem (diett): POST /travelExpense/perDiemCompensation
  Required: travelExpense:{{id}}, location (string e.g. "Oslo")
  DO NOT send: countryCode, numberOfDays, rateType, rateCategory — these fields don't exist

Costs (utgifter): POST /travelExpense/cost
  Required: travelExpense:{{id}}, amountCurrencyIncVat (number), paymentType:{{id}}
  Optional: date (YYYY-MM-DD), description
  DO NOT send: vatAmountCurrency, vatType, amountGross — wrong field names

### Ledger queries
GET /ledger/posting?dateFrom=X&dateTo=X&fields=* — fetch postings by date range

## VAT
excl. VAT / uten mva / hors TVA / ohne MwSt → isPrioritizeAmountsIncludingVat:false, unitPriceExcludingVatCurrency
incl. VAT / inkl. mva / avec TVA / mit MwSt → isPrioritizeAmountsIncludingVat:true, unitPriceIncludingVatCurrency

## TRANSLATIONS (key terms)
faktura/Rechnung/facture = invoice | leverandørfaktura/Lieferantenrechnung/facture fournisseur = SUPPLIER INVOICE (voucher!)
reiseregning = travel expense | avdeling/Abteilung = department | kreditnota/Gutschrift = credit note
bilag/Beleg/pièce = voucher | avskrivning/Abschreibung/amortissement = depreciation (use /asset/:depreciate)
fastpris/Festpreis/prix fixe = fixed price project | delbetaling/Teilzahlung/acompte = partial payment
"""


# ── Agent loop ────────────────────────────────────────────────────────────────

def _to_dict(obj):
    """Recursively convert proto-like objects to plain dicts."""
    if obj is None:
        return None
    return json.loads(json.dumps(obj, default=str))


def run_agent(prompt: str, files: list, base_url: str, token: str, ctx: dict, rid: str = ""):
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    start = time.time()

    # File parts
    parts = []
    for f in files:
        mime = f.get("mime_type", "")
        data = base64.b64decode(f["content_base64"])
        if mime.startswith("image/") or mime == "application/pdf":
            parts.append(types.Part(inline_data=types.Blob(mime_type=mime, data=data)))
        else:
            try:
                parts.append(types.Part(text=f"File: {f.get('filename', '')}\n{data.decode()}"))
            except Exception:
                pass

    # Context summary
    def fmt_list(items, fn):
        return "\n".join(f"  {fn(i)}" for i in items) if items else "  (none)"

    ctx_text = f"""## Task
{prompt}

## Session
- Employee ID: {ctx.get('employee_id')} ({ctx.get('employee_name')})
- Company: {ctx.get('company_id')} — {ctx.get('company_name')}
- Bank account: {ctx.get('bank_account') or 'not set'}
- Today: {TODAY}

### VAT Types
{fmt_list(ctx.get('vat_types', []), lambda v: f"ID {v['id']}: #{v['number']} {v['name']} ({v['pct']}%)")}

### Payment Types
{fmt_list(ctx.get('payment_types', []), lambda p: f"ID {p['id']}: {p.get('description')}")}

### Customers
{fmt_list(ctx.get('customers', []), lambda c: f"ID {c['id']}: {c.get('name')} (org: {c.get('organizationNumber')}, email: {c.get('email')})")}

### Employees
{fmt_list(ctx.get('employees', []), lambda e: f"ID {e['id']}: {e.get('firstName')} {e.get('lastName')} ({e.get('email')})")}

### Products
{fmt_list(ctx.get('products', []), lambda p: f"ID {p['id']}: {p.get('name')} (price excl: {p.get('priceExcludingVatCurrency')})")}

### Invoices
{fmt_list(ctx.get('invoices', []), lambda i: f"ID {i['id']}: #{i.get('invoiceNumber')} amount={i.get('amount')} outstanding={i.get('amountOutstanding')}")}

### Orders
{fmt_list(ctx.get('orders', []), lambda o: f"ID {o['id']}: #{o.get('number')} customer={o.get('customer', {}).get('name') if isinstance(o.get('customer'), dict) else '?'}")}

### Departments
{fmt_list(ctx.get('departments', []), lambda d: f"ID {d['id']}: {d.get('name')} (#{d.get('departmentNumber')})")}

### Projects
{fmt_list(ctx.get('projects', []), lambda p: f"ID {p['id']}: {p.get('name')}")}

### Travel Expenses
{fmt_list(ctx.get('travel_expenses', []), lambda t: f"ID {t['id']}: {t.get('title')} status={t.get('status')}")}

### Assets
{fmt_list(ctx.get('assets', []), lambda a: f"ID {a['id']}: {a.get('name')} (acq cost: {a.get('acquisitionCost')})")}

### All Ledger Accounts ({len(ctx.get('ledger_accounts', []))} total)
{fmt_list(ctx.get('ledger_accounts', []), lambda a: f"ID {a['id']}: #{a.get('number')} {a.get('name')}")}

---
Plan your steps, then execute efficiently."""

    parts.append(types.Part(text=ctx_text))
    contents = [types.Content(role="user", parts=parts)]

    tag = f"[{rid}] " if rid else ""

    # ── Per-task stats tracking ───────────────────────────────────────────────
    stats = {
        "iterations": 0,
        "write_calls": 0,   # POST/PUT/DELETE/PATCH — these cost efficiency score
        "errors_4xx": 0,    # Client errors — also hurt score
        "calls": [],        # (method, endpoint, status, error_msg)
        "stop_reason": "max_iterations",
    }

    WRITE_METHODS = {"POST", "PUT", "DELETE", "PATCH"}

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
                    system_instruction=SYSTEM_PROMPT,
                    tools=TOOLS,
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
                print(f"{tag}  💭 {part.text.strip()[:800]}")

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
            params   = _to_dict(inp.get("params"))
            body     = _to_dict(inp.get("body"))

            is_write = method.upper() in WRITE_METHODS
            call_icon = "✏️ " if is_write else "🔍"
            print(f"{tag}  {call_icon} {method} {endpoint}", end="")
            if body:
                print(f"  body={json.dumps(body)[:200]}", end="")
            print()

            result = tx(method, base_url, token, endpoint, params=params, body=body)
            status = result.get("status_code", 0)

            if is_write:
                stats["write_calls"] += 1

            # Parse error details fully for human readability
            error_summary = None
            if status >= 400:
                stats["errors_4xx"] += 1
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

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - start
    failed_writes = [c for c in stats["calls"] if c["is_write"] and c["status"] >= 400]
    ok_writes     = [c for c in stats["calls"] if c["is_write"] and c["status"] < 400]

    print(f"\n{tag}" + "═" * 54)
    print(f"{tag} SUMMARY  [{rid}]  stop={stats['stop_reason']}  time={elapsed:.0f}s")
    print(f"{tag}{'─' * 54}")
    print(f"{tag}  Iterations : {stats['iterations']}/15")
    print(f"{tag}  Write calls: {stats['write_calls']}  (✅ {len(ok_writes)} ok  ❌ {len(failed_writes)} failed)")
    print(f"{tag}  4xx errors : {stats['errors_4xx']}")

    if ok_writes:
        print(f"{tag}  ── Successful writes:")
        for c in ok_writes:
            print(f"{tag}     {c['method']} {c['endpoint']}  [{c['status']}]")

    if failed_writes:
        print(f"{tag}  ── Failed writes (hurt score!):")
        for c in failed_writes:
            print(f"{tag}     {c['method']} {c['endpoint']}  [{c['status']}] — {c['error'] or ''}")

    # Spot patterns: repeated failures on same endpoint = likely missing knowledge
    from collections import Counter
    fail_endpoints = Counter(
        f"{c['method']} {c['endpoint']}" for c in stats["calls"] if c["status"] >= 400
    )
    if fail_endpoints:
        print(f"{tag}  ── Error frequency (fix these in system prompt):")
        for ep, cnt in fail_endpoints.most_common():
            print(f"{tag}     {cnt}x  {ep}")

    print(f"{tag}" + "═" * 54)


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
        print(f"\n[{rid}] {'='*54}\n[{rid}] TASK: {prompt[:300]}\n[{rid}] Files: {len(files)}\n[{rid}] {'='*54}")
        print(f"[{rid}] --- Pre-fetch ---")

        ctx = await asyncio.to_thread(prefetch, base_url, token)
        print(f"[{rid}]   {ctx.get('employee_name')} | {ctx.get('company_name')}")
        print(f"[{rid}]   customers={len(ctx.get('customers',[]))} employees={len(ctx.get('employees',[]))} "
              f"products={len(ctx.get('products',[]))} accounts={len(ctx.get('ledger_accounts',[]))} "
              f"assets={len(ctx.get('assets',[]))}")

        print(f"[{rid}] --- Agent ---")
        await asyncio.to_thread(run_agent, prompt, files, base_url, token, ctx, rid)

        return JSONResponse({"status": "completed"})
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
        return JSONResponse({"status": "completed"})
