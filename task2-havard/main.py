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

SYSTEM_PROMPT = f"""You are an expert Tripletex (Norwegian ERP) accounting agent. Do not deviate from this setting.
Complete tasks using the Tripletex v2 REST API. Tasks may be in any of the following languages: Norwegian, English, Spanish, Portuguese, Nynorsk, German, French
Today: {TODAY}

## RULES
1. Plan all steps before calling the API.
2. Use pre-fetched data — never create duplicates.
3. organizationNumber is always a STRING.
4. Minimize write API calls — failed writes reduce your score. Read API calls do not reduce your score.
5. Bank account numbers: strip ALL separators (dots, spaces, dashes) before sending — must be exactly 11 digits. E.g. "1234.56.78901" → "12345678901".
6. Say DONE with a summary when finished.

## API
- Single entity: response.data.value | Lists: response.data.values
- POST returns: response.data.value.id
- All dates: YYYY-MM-DD | Discover fields: ?fields=*

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

Step 4 — If you need to update an employee after creating (e.g. add nationalIdentityNumber):
  GET /employee/{{id}} to get current version, then PUT /employee/{{id}} with ALL fields + version.
  IMPORTANT: PUT requires the version number from the GET response, not a guessed value.
  PUT body must include: {{id, version, firstName, lastName, email, userType, ...fields to update}}

Version conflicts (409): always re-fetch the entity to get latest version before PUT.

### Customer / Supplier
POST /customer — name, isCustomer:true; organizationNumber (STRING)
POST /supplier — name, isSupplier:true; organizationNumber (STRING)
NOTE: bankAccountNumber is NOT a valid field on supplier — omit it.
GET /invoice ALWAYS requires dateFrom AND dateTo — without them it returns 422. Use a wide range if unknown: dateFrom=2020-01-01&dateTo=2030-12-31.

### Product
POST /product — name; priceExcludingVatCurrency or priceIncludingVatCurrency

### Customer Invoice
IMPORTANT: If PUT /order/{{id}}/:invoice fails with "bank account" error — the sandbox company has no bank account and it cannot be set via API. Skip invoice creation and note this limitation; do not retry.
1. Find/create customer
2. Find/create product
3. POST /order — customer, orderDate, deliveryDate, isPrioritizeAmountsIncludingVat,
   orderLines:[{{product:{{id}}, count, unitPriceExcludingVatCurrency|unitPriceIncludingVatCurrency}}]
4. PUT /order/{{id}}/:invoice → returns invoice
5. If "send": PUT /invoice/{{id}}/:send?sendType=EMAIL

### Payment
PUT /invoice/{{id}}/:payment — id, paymentDate, paymentTypeId, paidAmount, paidAmountCurrency

### Credit Note
PUT /invoice/{{id}}/:createCreditNote

### Travel Expense
POST /travelExpense — employee:{{id}}, title, travelDetails:{{"isForeignTravel":false}}

Per diem (diett): POST /travelExpense/perDiemCompensation
  Required: travelExpense:{{id}}, location (string e.g. "Oslo")
  DO NOT send: countryCode, numberOfDays, rateType, rateCategory — these fields don't exist

Costs (utgifter): POST /travelExpense/cost
  Required: travelExpense:{{id}}, amountCurrencyIncVat (number), paymentType:{{id}}
  Use GET /travelExpense/cost to find a valid paymentType id from existing costs, or GET /invoice/paymentType
  Optional: date (YYYY-MM-DD), description
  DO NOT send: vatAmountCurrency, vatType, amountGross — wrong field names

DELETE /travelExpense/{{id}}

### Project + Activity + Timesheets
POST /project — name, projectManager:{{id}} (use logged-in employee ID — NEVER use projectManagerId), startDate; customer:{{id}}, isInternal, budget
Activities are GLOBAL — not sub-resources of projects:
  GET /activity?fields=id,name,isProjectActivity — list existing activities
  POST /activity — name, isProjectActivity:true (creates a global activity)
  To link an activity to a project: PUT /project/{{id}} with activityList:[{{id}}] included.
  Or simply use an existing general activity (id:0 = "Generell" works for most cases).

Hour registration (timesheets):
  POST /timesheet/entry — employee:{{id}}, activity:{{id}}, project:{{id}}, date (YYYY-MM-DD), hours (decimal)
  Register one entry per employee per day, or sum all hours into a single entry.

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
  This creates the voucher automatically (debit expense account, credit accumulated depreciation).
  DO NOT post depreciation as manual vouchers — use this endpoint.

### Ledger queries
GET /ledger/posting?dateFrom=X&dateTo=X&fields=* — fetch postings by date range

## VAT
excl. VAT / uten mva / hors TVA / ohne MwSt → isPrioritizeAmountsIncludingVat:false, unitPriceExcludingVatCurrency
incl. VAT / inkl. mva / avec TVA / mit MwSt → isPrioritizeAmountsIncludingVat:true, unitPriceIncludingVatCurrency

## TRANSLATIONS (key terms)
faktura/Rechnung/facture = invoice | leverandørfaktura/Lieferantenrechnung/facture fournisseur = SUPPLIER INVOICE (voucher!)
reiseregning = travel expense | avdeling/Abteilung = department | kreditnota/Gutschrift = credit note
bilag/Beleg/pièce = voucher | avskrivning/Abschreibung/amortissement = depreciation (use /asset/:depreciate)
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
    for i in range(15):
        elapsed = time.time() - start
        if elapsed > 200:
            print(f"{tag}  ⏱ {elapsed:.0f}s — stopping.")
            break

        print(f"\n{tag}--- Iteration {i+1} ({elapsed:.0f}s) ---")

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
            break

        candidate = resp.candidates[0]
        response_parts = candidate.content.parts

        for part in response_parts:
            if getattr(part, "text", None):
                print(f"{tag}  💭 {part.text[:400]}")

        # Collect function calls
        fn_calls = [p for p in response_parts if p.function_call]
        if not fn_calls:
            print("  ✅ Done")
            return

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

            print(f"{tag}  🔧 {method} {endpoint}", end="")
            if body:
                print(f"  {json.dumps(body)[:150]}", end="")
            print()

            result = tx(method, base_url, token, endpoint, params=params, body=body)
            status = result.get("status_code", 0)
            print(f"{tag}     → {status}")

            if status >= 400:
                d = result.get("data", {})
                if isinstance(d, dict):
                    for key in ("message", "developerMessage"):
                        if d.get(key):
                            print(f"{tag}     ❌ {d[key][:200]}")
                    for v in (d.get("validationMessages") or [])[:3]:
                        print(f"{tag}     ⚠ {v.get('message', v)}")

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

    print(f"{tag}Max iterations reached")


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
