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

## Task Types & Results

| Task | Tier | Result | Notes |
|------|------|--------|-------|
| Create departments | 1 | ✅ 100% | Clean |
| Create employee (basic) | 1 | ✅ 100% | Needs userType, department, employment record |
| Create employee (from PDF) | 1 | ~45-59% | PDF parsing + employment/details still uncertain |
| Register payment on invoice | 1 | ✅ 100% | PUT /invoice/{id}/:payment — fast, 2 iterations |
| Create product | 1 | ✅ 100% | 1 iteration |
| Reverse payment (bounced) | 2 | ✅ working | PUT /ledger/voucher/{id}/:reverse |
| Travel expense | 2 | ~partial | perDiemCompensation + cost endpoints now known |
| Bank reconciliation | 3 | ~20% | Customer payments work; supplier vouchers fail |
| Project full lifecycle | 3 | ~partial | Timesheets work; invoice blocked by bank account |
| Supplier invoice (voucher) | 2 | unknown | Voucher structure now fixed in prompt |

## Confirmed Working API Patterns

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

PUT /employee: always GET first for version, send ALL fields + version
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
PUT /invoice/{id}/:payment — works reliably
GET /invoice — requires dateFrom and dateTo params
PUT /order/{id}/:invoice — FAILS in sandbox ("company has no bank account", unfixable via API)
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

1. **Invoice creation blocked in sandbox** — `PUT /order/{id}/:invoice` fails with "company has no bank account". Cannot be fixed via API (PUT /company returns 405). Affects project lifecycle scoring.
2. **Voucher to 2400/2600** — these accounts are system-managed. Use 1920 (bank) as the credit side for supplier payments instead.
3. **"Endpoint unreachable"** — caused by running without `--workers 3`. Single worker = sequential = tasks queue up and timeout.

## Still Unknown / To Discover

1. **Custom accounting dimensions** — endpoint unknown. Task: create dimension "Region" with values.
2. **Employee admin role** — "kontoadministrator" role fields unknown.
3. **~18 task types unseen** — keep submitting to discover them.

## API Quick Reference
- Auth: Basic Auth, username `0`, password = session_token
- All calls go through provided base_url (proxy)
- Single entity: `response["value"]` | Lists: `response["values"]`
- POST returns created entity: `response["value"]["id"]`
- Sandbox web UI: https://kkpqfuj-amager.tripletex.dev
- Session token expires: March 31, 2026
