# Tripletex AI Accounting Agent — Progress Report

## Project Overview

AI agent for the **Tripletex AI Accounting Agent competition** (https://app.ainm.no/submit/tripletex).
Agent receives accounting tasks in 7 languages, calls the Tripletex API, scored on correctness + efficiency.

- **30 task types**, 56 variants each (7 languages × 8 data sets)
- **Scoring:** correctness × tier multiplier. Tier 1 ×1 (max 2.0), Tier 2 ×2 (max 4.0), Tier 3 ×3 (max 6.0)
- **Efficiency bonus:** only applies on perfect (1.0) correctness. Fewer write calls + zero 4xx errors = up to 2× score
- **GET calls are free** — only POST/PUT/DELETE/PATCH count toward efficiency
- **Best score per task kept** — bad runs never lower your score
- **Timeout:** 300s per submission. Agent stops at 200s (100s buffer for response).
- **Rate limits:** 10 submissions per task per day, 300 total daily, 3 concurrent

## Architecture

**Stack:** Python + FastAPI + Google Gemini (`gemini-3.1-pro-preview`) + Tripletex v2 REST API

**Flow:**
1. POST /solve receives task
2. `prefetch()` fetches all reference data including all 457+ ledger accounts + salary types at once
3. Full context + task prompt + files sent to Gemini
4. Gemini calls `tripletex_api` tool in agentic loop (max 15 iterations, 200s hard stop)
5. Returns `{"status": "completed"}`

**Key file:** `main.py` — single file
**Run with:** `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 3` ← MUST use --workers 3 for true concurrency

## Running the Agent
```bash
export GOOGLE_API_KEY=...
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 3
# Expose via ngrok (NOT cloudflare — 120s timeout too short)
ngrok http 8000
# Submit URL: https://xxxx.ngrok-free.app/solve
```

## Scoring Strategy — KEY FOCUS

The efficiency bonus can **double** your score, but ONLY on perfect (1.0) correctness.
A Tier 2 task: perfect + efficient = 4.0. Perfect + sloppy = ~2.1. Imperfect = 1.6 flat.

**To maximize score:**
1. GET everything needed before writing — GETs are free, build full picture first
2. Every write must succeed first time — 4xx errors hurt even on eventually-correct tasks
3. Zero retries — if an endpoint fails, fix the approach in the system prompt, don't retry same body

**When a task scores badly — how to investigate:**
1. Read the write log in the terminal output — find which exact endpoint failed and what error it gave
2. Check if the error is a wrong field name, wrong param location (body vs query), or missing required field
3. If unknown: use `test_reverse.py` pattern — write a quick script to call the endpoint manually and check OpenAPI spec
4. Fetch OpenAPI: `GET /v2/openapi.json` — search for the path, check `parameters[].in` (query vs path) and `requestBody`
5. Fix the system prompt with the confirmed correct syntax before re-running

## Task Types & Results

| Task | Tier | Result | Notes |
|------|------|--------|-------|
| Create departments | 1 | ✅ 100% | Clean, 1 write |
| Create employee (basic) | 1 | ✅ 100% | Needs userType, department, employment record |
| Create employee (from PDF) | 1 | ~45-59% | PDF parsing hit-or-miss |
| Register payment on invoice | 1 | ✅ fixed | `:payment` now uses query params — was failing with body |
| Create product | 1 | ✅ 100% | 1 write, 12s — perfect efficiency |
| Create supplier | 1 | ✅ 100% | 1 write, clean |
| Create invoice + send | 2 | ❌ partial | Sandbox bank account blocker — order/invoice OK, invoice send fails |
| Credit note | 2 | ✅ working | PUT /invoice/{id}/:createCreditNote, no body |
| Reverse voucher | 2 | ✅ fixed | PUT /ledger/voucher/{id}/:reverse?date=YYYY-MM-DD — was failing 4× per task |
| Project with fixed price + partial invoice | 2/3 | ✅ working | GET-first rule fixed PUT /project failures |
| Travel expense | 2 | ~partial | perDiemCompensation + cost endpoints documented |
| Bank reconciliation (CSV) | 3 | ~50% | Customer payments work; supplier vouchers still flaky |
| Ledger error correction | 3 | ~partial | 403 early-exit now smarter (feature 403 ≠ token 403) |
| Supplier invoice (voucher) | 2 | working | Voucher structure confirmed |
| Salary/payroll | 2/3 | ✅ added | POST /salary/transaction documented — was 0% |
| Depreciation | 2 | improved | 403 on /asset now falls back to manual vouchers instead of stopping |
| Custom accounting dimensions | ? | ❌ 0% | No API endpoints exist — agent now stops early |

## Confirmed Working API Patterns

### UNIVERSAL PUT RULE (applies everywhere)
```
ANY PUT to an existing entity:
  1. GET /{endpoint}/{id} first → get current version number + all fields
  2. PUT with ALL fields (including unchanged ones) + version number
  3. Missing any required field or version → 422 "Kan ikke være null"
Applies to: PUT /employee, PUT /project, PUT /customer, PUT /ledger/account, etc.
```

### Action Endpoints — param location matters, verified against OpenAPI
```
PUT /order/{id}/:invoice              — NO body. Add ?invoiceDate=YYYY-MM-DD if 422
PUT /invoice/{id}/:send               — query param only: ?sendType=EMAIL
PUT /invoice/{id}/:createCreditNote   — NO body, NO params
PUT /invoice/{id}/:payment            — ALL params as QUERY PARAMS (NOT body):
                                         ?paymentDate=YYYY-MM-DD&paymentTypeId=N&paidAmount=N&paidAmountCurrency=N
PUT /ledger/voucher/{id}/:reverse     — ?date=YYYY-MM-DD as QUERY PARAM (REQUIRED), NO body
                                         Confirmed via OpenAPI (in=query) and sandbox test
PUT /asset/{id}/:depreciate           — body: {date, amount}
```

### Employee — FULL FLOW
```
POST /employee:
  firstName, lastName, email, userType:"STANDARD", department:{id}
  Optional: dateOfBirth, nationalIdentityNumber, bankAccountNumber (11 digits, no separators)

POST /employee/employment:
  employee:{id}, startDate, isMainEmployer:true, taxDeductionCode:"loennFraHovedarbeidsgiver"
  ← do NOT put salary/% here

POST /employee/employment/details:
  employment:{id}, date (=startDate)
  employmentPercentage, annualWage OR monthlyWage, occupationCode (7-digit string), jobTitle
  ← NOT percentageOfFullTimeEquivalent, NOT annualSalary

PUT /employee: GET first for version, send ALL fields + version
```

### Project — FULL FLOW
```
POST /project: name, projectManager:{id}, startDate, customer:{id}, isInternal:false
  Optional: fixedprice (decimal), isFixedPrice:true, budget

PUT /project/{id} to update:
  GET /project/{id} first → grab version + all current fields
  PUT with ALL fields + version + your changes
  e.g.: {id, version, name, startDate, customer:{id}, projectManager:{id},
         isInternal:false, isFixedPrice:true, fixedprice:429500}
```

### Voucher (manual journal / supplier invoice)
```
POST /ledger/voucher — CRITICAL rules:
- Each posting: row (1-indexed), date, account:{id}, amountGross, amountGrossCurrency
- amountGross MUST equal amountGrossCurrency
- Postings MUST sum to zero. Positive=debit, Negative=credit.
- NEVER include vatType in postings ("system-generated row" error)
- NEVER post to 2400 (Leverandørgjeld) or 2600 (Skattetrekk) — system-managed
- GET /ledger/voucher requires dateFrom + dateTo params

Supplier invoice example (net 39800, 25% VAT=9950, total 49750):
  row 1: account 6340 (expense),  amountGross=39800
  row 2: account 2710 (VAT in),   amountGross=9950
  row 3: account 1920 (bank),     amountGross=-49750
```

### Salary / Payroll
```
GET /salary/type?fields=id,number,name → in pre-fetched context as salary_types
Key types: #2000 "Fastlønn" (monthly), #2001 "Timelønn" (hourly), #5001 "Kilometergodtgjørelse"

POST /salary/transaction:
{
  "date": "YYYY-MM-DD", "year": 2026, "month": 3, "isHistorical": false,
  "payslips": [{
    "employee": {"id": X}, "date": "YYYY-MM-DD", "year": 2026, "month": 3,
    "specifications": [{
      "salaryType": {"id": <#2000 id>}, "rate": 45000, "count": 1, "amount": 45000,
      "employee": {"id": X}
    }]
  }]
}

NOT WORKING: /salary/transaction (403), /salary/specification (403),
             /salary/payment (404), /salary/run (404)
```

### Fixed Assets & Depreciation
```
GET /asset → 403 "no permission to feature" in some sandbox accounts (module not enabled).
  → Fall back to manual vouchers: debit 6010, credit 1209

POST /asset (when module available):
  name, dateOfAcquisition (NOT acquisitionDate), acquisitionCost,
  account:{id} (balance sheet, e.g. 1200),
  depreciationAccount:{id} (expense, e.g. 6010 — NOT depreciationAccountNumber),
  lifetime (months, NOT years),
  depreciationMethod: "STRAIGHT_LINE"|"TAX_RELATED"|"MANUAL"|"NO_DEPRECIATION"

PUT /asset/{id}/:depreciate — body: {date, amount}
```

### Travel Expense
```
POST /travelExpense: employee:{id}, title, travelDetails:{"isForeignTravel":false}
POST /travelExpense/perDiemCompensation: travelExpense:{id}, location:"Oslo"
  DO NOT send: countryCode, numberOfDays, rateType, rateCategory (don't exist)
POST /travelExpense/cost: travelExpense:{id}, amountCurrencyIncVat, paymentType:{id}
  DO NOT send: vatAmountCurrency, vatType, amountGross (wrong field names)
```

### Timesheets
```
POST /timesheet/entry: employee:{id}, activity:{id}, project:{id}, date, hours (decimal)
Activities are GLOBAL — GET /activity?fields=id,name,isProjectActivity to list them
Activity id "Generell" works for most cases — reuse it to avoid extra writes
POST /activity: name, isProjectActivity:true
```

### Other Confirmed Patterns
```
POST /supplier — name, isSupplier:true, organizationNumber (STRING)
  ← bankAccountNumber NOT a valid field on supplier

POST /customer — name, isCustomer:true, organizationNumber (STRING)

Bank account numbers from PDFs: strip ALL dots/spaces → must be exactly 11 digits

GET /invoice — requires dateFrom AND dateTo params (use 2020-01-01 to 2030-12-31 if unknown)
GET /ledger/voucher — requires dateFrom AND dateTo params
```

## Known Blockers

1. **Invoice creation blocked in sandbox** — `PUT /order/{id}/:invoice` fails with "company has no bank account". Cannot be fixed via API. Agent now stops immediately on this error.
2. **Custom accounting dimensions** — no API endpoints exist at all. Agent now stops after 2 failed GETs.
3. **"Endpoint unreachable"** — caused by running without `--workers 3`. Single worker = tasks queue and timeout.
4. **Salary/payroll** — POST /salary/transaction now documented but not yet tested live.
5. **Fixed assets module** — GET /asset returns 403 in sandbox (module not enabled). Agent falls back to manual vouchers instead of stopping (fixed — was incorrectly triggering fatal early-exit).

## Early-Exit Rules (in agent loop)

The agent loop now stops immediately on:
- **Bank account error** — "company has no bank account" / "bankkonto" in error → sandbox limitation, no retry
- **Token 403 on first call** — only if error is NOT "permission to access this feature" (feature-gated 403s are normal, let agent continue)
- **3+ consecutive 404s on GETs** — unknown endpoint pattern, stops wasted iterations

## Still Unknown / To Discover

1. **Employee admin role** — "kontoadministrator" / administrator role assignment fields unknown.
2. **~15 task types unseen** — keep submitting to discover them. Log new task prompts here when seen.
3. **Salary correctness** — POST /salary/transaction is now in the prompt but not yet confirmed working in live runs.

## API Quick Reference
- Auth: Basic Auth, username `0`, password = session_token
- All calls go through provided base_url (proxy)
- Single entity: `response["value"]` | Lists: `response["values"]`
- POST returns created entity: `response["value"]["id"]`
- OpenAPI spec: `GET /v2/openapi.json` — check `parameters[].in` for query vs path, `requestBody` for body
- Sandbox web UI: https://kkpqfuj-amager.tripletex.dev
- Session token expires: March 31, 2026
