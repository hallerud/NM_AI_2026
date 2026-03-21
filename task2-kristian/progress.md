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
2. `prefetch()` fetches all reference data including all 457+ ledger accounts at once
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
3. Zero retries — if an endpoint fails, fix your approach, don't retry the same body

**Observed efficiency in recent runs:**
- Best case: 1 write, 0 errors, 12s (product creation) → full efficiency bonus
- Worst case: 12 writes, 8 errors, 123s (project task) → nearly zero efficiency bonus despite correct output

**Highest-leverage fixes (do these first):**
- `PUT /project` was failing 6× per run → fixed with GET-first rule in system prompt
- `PUT /order/:invoice` with body → 422, retry → wasted write → fixed (action endpoints take NO body)
- Agent retrying after bank account error → fixed (stop on first failure)
- Product duplicate creation → fixed (check pre-fetched list before POST)

## Task Types & Results

| Task | Tier | Result | Notes |
|------|------|--------|-------|
| Create departments | 1 | ✅ 100% | Clean, 1 write |
| Create employee (basic) | 1 | ✅ 100% | Needs userType, department, employment record |
| Create employee (from PDF) | 1 | ~45-59% | PDF parsing + employment/details still uncertain |
| Register payment on invoice | 1 | ✅ 100% | PUT /invoice/{id}/:payment — fast, 2 iterations |
| Create product | 1 | ✅ 100% | 1 write, 12s — perfect efficiency |
| Create supplier | 1 | ✅ 100% | 1 write, clean |
| Create invoice + send | 2 | ❌ 0% | Blocked by sandbox bank account — order created but invoice fails |
| Credit note | 2 | ✅ working | PUT /invoice/{id}/:createCreditNote, no body |
| Reverse payment (bounced) | 2 | ✅ working | PUT /ledger/voucher/{id}/:reverse |
| Project with fixed price + partial invoice | 2/3 | ✅ 7/7 | Works but 8 failed writes → poor efficiency |
| Travel expense | 2 | ~partial | perDiemCompensation + cost endpoints now known |
| Bank reconciliation (CSV) | 3 | ~50% | Customer payments work; supplier vouchers still flaky |
| Ledger error correction | 3 | ❌ 0% | 403 proxy token errors — token may expire per task |
| Supplier invoice (voucher) | 2 | unknown | Voucher structure now fixed in prompt |
| Salary/payroll | ? | ❌ 0% | Endpoint completely unknown — needs discovery |

## Confirmed Working API Patterns

### UNIVERSAL PUT RULE (applies everywhere)
```
ANY PUT to an existing entity:
  1. GET /{endpoint}/{id} first → get current version number + all fields
  2. PUT with ALL fields (including unchanged ones) + version number
  3. Missing any required field or version → 422 "Kan ikke være null"
Applies to: PUT /employee, PUT /project, PUT /customer, PUT /ledger/account, etc.
Exception: action endpoints (/:invoice, /:payment, /:depreciate, /:reverse, /:send) — NO body unless stated
```

### Action Endpoints — NO BODY
```
PUT /order/{id}/:invoice       — no body, no params
PUT /invoice/{id}/:send        — query param only: ?sendType=EMAIL
PUT /invoice/{id}/:createCreditNote — no body
PUT /ledger/voucher/{id}/:reverse   — no body
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

PUT /project/{id} to update (e.g. set fixed price):
  GET /project/{id} first → grab version + all current fields
  PUT with ALL fields + version + your changes
  e.g.: {id, version, name, startDate, customer:{id}, projectManager:{id},
         isInternal:false, isFixedPrice:true, fixedprice:429500}
```

### Voucher (manual journal / supplier invoice)
```
POST /ledger/voucher — CRITICAL rules:
- Each posting needs: row (1-indexed, never 0), date, account:{id}, amountGross, amountGrossCurrency
- amountGross MUST equal amountGrossCurrency
- NEVER use vatType in postings (triggers "system-generated row" error)
- NEVER post to 2400 (Leverandørgjeld) or 2600 (Skattetrekk) — system-managed, will reject
- Post 3 lines: expense (debit) + VAT input 2710 (debit) + bank 1920 (credit)
```

### Travel Expense
```
POST /travelExpense: employee:{id}, title, travelDetails:{"isForeignTravel":false}

POST /travelExpense/perDiemCompensation:
  travelExpense:{id}, location:"Oslo"  ← ONLY these fields
  DO NOT send: countryCode, numberOfDays, rateType, rateCategory (don't exist)

POST /travelExpense/cost:
  travelExpense:{id}, amountCurrencyIncVat, paymentType:{id}
  DO NOT send: vatAmountCurrency, vatType, amountGross (wrong field names)
```

### Timesheets / Hour Registration ✅ CONFIRMED WORKING
```
POST /timesheet/entry: employee:{id}, activity:{id}, project:{id}, date, hours
Activities are GLOBAL — GET /activity to find existing ones
POST /activity: name, isProjectActivity:true
```

### Invoice / Payment
```
PUT /invoice/{id}/:payment — body: {paymentDate, paymentTypeId (plain int!), paidAmount, paidAmountCurrency}
GET /invoice — requires dateFrom and dateTo params (use 2020-01-01 to 2030-12-31 if unknown)
PUT /order/{id}/:invoice — FAILS in sandbox ("company has no bank account", unfixable via API)
  → stop after first failure, do NOT retry
```

### Other Confirmed Patterns
```
POST /supplier — name, isSupplier:true, organizationNumber (STRING)
  ← bankAccountNumber NOT a valid field on supplier

PUT /asset/{id}/:depreciate — body: {date, amount}
  ← never post depreciation as manual voucher

PUT /ledger/voucher/{id}/:reverse — reverses a payment voucher cleanly

Bank account numbers from PDFs: strip ALL dots/spaces → must be exactly 11 digits
```

## Known Blockers

1. **Invoice creation blocked in sandbox** — `PUT /order/{id}/:invoice` fails with "company has no bank account". Cannot be fixed via API. Affects all invoice/project lifecycle tasks — accept partial score.
2. **Voucher to 2400/2600** — system-managed accounts. Use 1920 (bank) as credit side for supplier payments.
3. **"Endpoint unreachable"** — caused by running without `--workers 3`. Single worker = tasks queue and timeout.
4. **Salary/payroll** — endpoints completely unknown. Agent guesses randomly and fails. Zero score until discovered.
5. **Ledger error correction** — sometimes hits 403 (invalid/expired proxy token per-task). Nothing the agent can do.

## What To Focus On Next (priority order)

### 1. Discover salary/payroll endpoints (HIGH VALUE)
The agent currently scores 0% on salary tasks. Need to manually explore:
- `GET /salary/transaction?fields=*`
- `GET /salary/payslip?fields=*`
- `GET /salary/settings/pensionScheme?fields=*`
Use the sandbox UI (https://kkpqfuj-amager.tripletex.dev) to run a payroll and inspect what API calls it makes.

### 2. Fix efficiency on tasks we already get right (HIGH VALUE)
Tasks we score 100% correctness on but waste writes:
- Project tasks: now fixed in system prompt (GET-first rule for PUT /project)
- Invoice tasks: now fixed (action endpoints take no body)
- Re-run these to capture the efficiency bonus (can be 2× the base score)

### 3. Discover remaining unknown task types (~15 unseen)
Keep submitting to see new task types. When a new one appears, log the prompt + what worked/failed here.
Known unseen: custom accounting dimensions, employee admin role, year-end closing, budget tasks, dimension values.

### 4. Bank reconciliation reliability (MEDIUM)
Currently ~50%. Supplier voucher postings still sometimes fail. Review voucher rules and test more.

### 5. Employee from PDF (MEDIUM)
~45-59%. The PDF parsing is hit-or-miss. Could improve by being more explicit in the prompt about
which fields to extract and how to handle missing data.

## Still Unknown / To Discover

1. **Salary/payroll** — entire flow unknown. `/salary/transaction`, `/salary/payslip` return 500/404.
2. **Custom accounting dimensions** — endpoint unknown. Task: create dimension "Region" with values.
3. **Employee admin role** — "kontoadministrator" / administrator role assignment fields unknown.
4. **~15 task types unseen** — keep submitting to discover them.

## API Quick Reference
- Auth: Basic Auth, username `0`, password = session_token
- All calls go through provided base_url (proxy)
- Single entity: `response["value"]` | Lists: `response["values"]`
- POST returns created entity: `response["value"]["id"]`
- Sandbox web UI: https://kkpqfuj-amager.tripletex.dev
- Session token expires: March 31, 2026
