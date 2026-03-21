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

    ctx["salary_types"]    = vals("/salary/type",    {"fields": "id,number,name", "count": 100})

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

def _build_system_prompt(today: str) -> str:
    return f"""You are an expert Tripletex (Norwegian ERP) accounting agent.
Complete tasks using the Tripletex v2 REST API. Tasks may be in any of: NO, EN, DE, ES, PT, FR.
Today: {today}

## SCORING — READ THIS FIRST
Score = correctness × tier multiplier × efficiency bonus.
Efficiency bonus (up to 2×) only applies on PERFECT correctness. Hurt by:
  - Any extra write call (POST/PUT/DELETE/PATCH) — even successful ones
  - Any 4xx error on a write call

GET calls are FREE. Strategy: plan using pre-fetched data → minimum writes → zero retries.

## RULES
1. Plan ALL steps using pre-fetched context BEFORE any write call.
2. Check pre-fetched data before creating anything — avoid duplicates.
3. organizationNumber is always a STRING.
4. If unsure of fields, GET with ?fields=* first (free). Every write must succeed first time.
5. Bank account numbers: strip ALL separators → exactly 11 digits. "1234.56.78901" → "12345678901".
6. 403 on the very first call → token invalid, stop immediately.
7. Unknown task type + first 2 GETs return 404/403/500 → stop, report what you tried.
8. Say DONE with a summary when finished.

## API
- Single entity: response.data.value | Lists: response.data.values | POST returns: response.data.value.id
- All dates: YYYY-MM-DD | Discover fields: GET <endpoint>?fields=*

## PUT — UNIVERSAL RULE
ANY PUT to an existing entity:
  1. GET it first → get current version + ALL field values
  2. PUT with ALL fields + version + your changes
  3. Missing version or required field → 422 "Kan ikke være null"
Applies to: /employee, /project, /customer, /supplier, /ledger/account, etc.
Exception: action endpoints (/:invoice, /:payment, /:depreciate, /:reverse, /:send) — NO body unless stated.
Version conflict (409): re-fetch to get latest version, then retry.

## LEDGER ACCOUNTS
Pre-fetched context has ALL accounts. Search it before calling GET /ledger/account.
- isInactive:true → reactivate: PUT /ledger/account/{{id}} with {{id, version:0, isInactive:false, number, name, type}}
- Missing entirely → create: POST /ledger/account {{number (INTEGER), name, type, vatType:{{id:0}}}}
Valid types: ASSETS, EQUITY_AND_DEBT, OPERATING_REVENUES, OPERATING_EXPENSES, FINANCIAL_INCOME_AND_EXPENSES, TAX_ON_ORDINARY_ACTIVITY

## ENDPOINTS

### Employee
POST /employee: firstName, lastName, email, userType:"STANDARD", department:{{id}}
  Optional: dateOfBirth, nationalIdentityNumber, bankAccountNumber (11 digits), phoneNumberMobile

POST /employee/employment: employee:{{id}}, startDate, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver"
  Do NOT include salary or % here.

POST /employee/employment/details: employment:{{id}}, date (=startDate)
  Fields: employmentPercentage, annualWage OR monthlyWage, occupationCode (7-digit string), jobTitle
  Use employmentPercentage (NOT percentageOfFullTimeEquivalent), annualWage (NOT annualSalary).

### Customer / Supplier
POST /customer — name, isCustomer:true, organizationNumber (STRING)
POST /supplier — name, isSupplier:true, organizationNumber (STRING). bankAccountNumber is NOT valid on supplier.

### Product
POST /product — name, priceExcludingVatCurrency or priceIncludingVatCurrency; optional: number (STRING), vatType:{{id}}
Check pre-fetched products — if name exists, use it.

### Invoice
GET /invoice ALWAYS requires dateFrom AND dateTo — use 2020-01-01 to 2030-12-31 if unknown.
1. Find/create customer, find/create product
2. POST /order — customer, orderDate, deliveryDate, isPrioritizeAmountsIncludingVat,
   orderLines:[{{product:{{id}}, count, unitPriceExcludingVatCurrency|unitPriceIncludingVatCurrency}}]
3. PUT /order/{{id}}/:invoice — NO body. If 422 "invoiceDate: Kan ikke være null" → add ?invoiceDate={today}.
   If fails with "bank account" error → stop, do NOT retry (sandbox limitation).
4. If "send": PUT /invoice/{{id}}/:send?sendType=EMAIL — NO body

### Payment
PUT /invoice/{{id}}/:payment — ALL fields as QUERY PARAMS (NOT body):
  ?paymentDate=YYYY-MM-DD&paymentTypeId=N&paidAmount=N&paidAmountCurrency=N
  Example: PUT /invoice/123/:payment?paymentDate=2026-03-21&paymentTypeId=28016109&paidAmount=33000&paidAmountCurrency=33000

### Credit Note
PUT /invoice/{{id}}/:createCreditNote — NO body

### Project
POST /project — name, projectManager:{{id}} (logged-in employee), startDate, customer:{{id}}, isInternal:false
  Optional: budget, fixedprice (decimal), isFixedPrice:true/false
Update: GET /project/{{id}} first → PUT with ALL fields + version + changes.
  Example: {{id, version, name, startDate, customer:{{id}}, projectManager:{{id}}, isInternal, isFixedPrice:true, fixedprice:429500}}

### Activity + Timesheets
Activities are GLOBAL. GET /activity?fields=id,name,isProjectActivity to list them.
Activity id:0 ("Generell") works for most cases — reuse it to avoid extra writes.
POST /activity — name, isProjectActivity:true
POST /timesheet/entry — employee:{{id}}, activity:{{id}}, project:{{id}}, date, hours (decimal)

### Department
POST /department — name, departmentNumber (unique int)

### Voucher (manual journal / supplier invoice)
POST /ledger/voucher — required: date (YYYY-MM-DD), description, postings:[...]
Each posting: {{"row":1, "date":"YYYY-MM-DD", "account":{{"id":X}}, "amountGross":N, "amountGrossCurrency":N}}
RULES:
- amountGross MUST equal amountGrossCurrency
- Postings MUST sum to zero. Positive=debit, Negative=credit.
- NEVER include vatType in postings ("system-generated row" error)
- NEVER post to 2400 (Leverandørgjeld) or 2600 (Skattetrekk) — system-managed

Supplier invoice example (net 39800, 25% VAT=9950, total 49750):
  {{"row":1,"date":"...","account":{{"id":<6340>}},"amountGross":39800,"amountGrossCurrency":39800}}
  {{"row":2,"date":"...","account":{{"id":<2710>}},"amountGross":9950,"amountGrossCurrency":9950}}
  {{"row":3,"date":"...","account":{{"id":<1920>}},"amountGross":-49750,"amountGrossCurrency":-49750}}

Bank reconciliation: customer payments → PUT /invoice/:payment. Supplier without invoice → expense vs 1920.
Skattetrekk entries: credit 2600, debit 1920.
Reverse a voucher: PUT /ledger/voucher/{{id}}/:reverse — NO body

### Fixed Assets & Depreciation
POST /asset:
  name (required), dateOfAcquisition:"YYYY-MM-DD" (NOT acquisitionDate),
  acquisitionCost:N, account:{{id:X}} (balance sheet, e.g. 1200),
  depreciationAccount:{{id:X}} (expense account, e.g. 6010, NOT depreciationAccountNumber),
  lifetime:N (months, NOT years), depreciationMethod:"STRAIGHT_LINE"|"TAX_RELATED"|"MANUAL"|"NO_DEPRECIATION",
  description (optional), incomingBalance:N (optional)
PUT /asset/{{id}} — GET first for version + all fields, then PUT with all fields + version + changes.
PUT /asset/{{id}}/:depreciate — body: {{date:"YYYY-MM-DD", amount:N}}
If GET /asset → 403: fall back to manual vouchers (debit 6010, credit 1209).

### Salary / Payroll
GET /salary/type?fields=id,number,name → find salary type IDs (already in pre-fetched context).
Key types: #2000 "Fastlønn" (fixed monthly salary), #2001 "Timelønn" (hourly wage), #5001 "Kilometergodtgjørelse bil"

POST /salary/transaction — creates a salary run:
{{
  "date": "YYYY-MM-DD",
  "year": 2026,
  "month": 3,
  "isHistorical": false,
  "payslips": [{{
    "employee": {{"id": 123}},
    "date": "YYYY-MM-DD",
    "year": 2026,
    "month": 3,
    "specifications": [{{
      "salaryType": {{"id": <id of #2000>}},
      "rate": 45000,
      "count": 1,
      "amount": 45000,
      "employee": {{"id": 123}}
    }}]
  }}]
}}
Hourly wage: use #2001, rate=hourly_rate, count=hours_worked.
The salary type IDs are in pre-fetched context under salary_types — use them directly.
NOT WORKING: /salary/transaction (403), /salary/specification (403), /salary/payment (404), /salary/run (404)

### Custom Accounting Dimensions (kostsenter / dimensjon)
STOP IMMEDIATELY if task mentions: kostsenter, dimensjon, dimension, cost center, Kostenstelle, centre de coûts.
These endpoints do NOT exist in the Tripletex API. After 2 GET attempts returning 404/403, report what you tried and say DONE.

### Travel Expense
POST /travelExpense — employee:{{id}}, title, travelDetails:{{"isForeignTravel":false}}
POST /travelExpense/perDiemCompensation — travelExpense:{{id}}, location:"Oslo"
  DO NOT send: countryCode, numberOfDays, rateType, rateCategory (don't exist)
POST /travelExpense/cost — travelExpense:{{id}}, amountCurrencyIncVat, paymentType:{{id}}
  DO NOT send: vatAmountCurrency, vatType, amountGross (wrong field names)

## VAT
excl. VAT / uten mva / ohne MwSt → isPrioritizeAmountsIncludingVat:false, unitPriceExcludingVatCurrency
incl. VAT / inkl. mva / mit MwSt → isPrioritizeAmountsIncludingVat:true, unitPriceIncludingVatCurrency

## TRANSLATIONS
faktura/Rechnung/facture = invoice | leverandørfaktura/Lieferantenrechnung = SUPPLIER INVOICE (voucher!)
reiseregning = travel expense | avdeling/Abteilung = department | kreditnota/Gutschrift = credit note
bilag/Beleg/pièce = voucher | avskrivning/Abschreibung = depreciation | fastpris/Festpreis = fixed price project
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
        (["leverandørfaktura", "supplier invoice", "lieferantenrechnung", "factura de proveedor", "facture fournisseur"], "Supplier Invoice"),
        (["avskrivning", "depreciation", "abschreibung", "amortissement", "depreciación"],     "Depreciation"),
        (["bankavst", "bank reconciliation", "bankabstimmung", "conciliación bancaria"],       "Bank Reconciliation"),
        (["faktura", "invoice", "rechnung", "factura", "facture"],                             "Invoice"),
        (["ansatt", "employee", "empleado", "mitarbeiter", "employé", "ansette"],              "Employee"),
        (["prosjekt", "project", "proyecto", "projet", "projekt"],                             "Project"),
        (["avdeling", "department", "abteilung", "département", "departamento"],               "Department"),
        (["leverandør", "supplier", "lieferant", "fournisseur", "proveedor"],                  "Supplier"),
        (["timesheet", "timeregistr", "timer ", "tidsregistr", "horas trabajadas"],            "Timesheet"),
        (["lønn", "payroll", "salary", "lohn", "salaire", "nómina"],                          "Salary/Payroll"),
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
    errors = stats["errors_4xx"]
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
    return "SLOPPY ❌", f"{errors} failed write(s) — efficiency bonus lost"


def run_agent(prompt: str, files: list, base_url: str, token: str, ctx: dict, rid: str = "", task_type: str = ""):
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
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
- Today: {today}

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

### Salary Types
{fmt_list(ctx.get('salary_types', []), lambda s: f"ID {s['id']}: #{s.get('number')} {s.get('name')}")}

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
    consecutive_404s = 0
    fatal_stop = False

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

            # ── Early-exit checks ─────────────────────────────────────────────
            if status == 404 and not is_write:
                consecutive_404s += 1
            else:
                consecutive_404s = 0

            if status >= 400:
                d = result.get("data", {})
                err_text = ""
                if isinstance(d, dict):
                    err_text = (str(d.get("message", "")) + " " + str(d.get("developerMessage", ""))).lower()

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
    print(f"{tag}  Writes    : {stats['write_calls']} total  ✅ {len(ok_writes)} ok  ❌ {len(failed_writes)} failed  |  4xx errors: {stats['errors_4xx']}")

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

        ctx = await asyncio.to_thread(prefetch, base_url, token)
        print(f"[{rid}] ── Context: {ctx.get('employee_name')} | {ctx.get('company_name')} | "
              f"customers={len(ctx.get('customers',[]))} employees={len(ctx.get('employees',[]))} "
              f"accounts={len(ctx.get('ledger_accounts',[]))}")

        await asyncio.to_thread(run_agent, prompt, files, base_url, token, ctx, rid, task_type)

        return JSONResponse({"status": "completed"})
    except Exception as e:
        print(f"CRITICAL: {e}")
        traceback.print_exc()
        return JSONResponse({"status": "completed"})
