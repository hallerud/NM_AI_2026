# Tripletex AI Agent — Troubleshooting Summary for Claude Code

## Context
This is a competition agent (main.py) that receives accounting tasks in 7 languages, calls the Tripletex API via an LLM (Google Gemini), and gets scored on correctness × efficiency. Current score: ~50-56%. The goal is to fix specific failure modes to reach 80%+.

## Current Architecture
- Single file: `main.py` (596 lines)
- Stack: Python + FastAPI + Google Gemini (`gemini-3.1-pro-preview`) + Tripletex v2 REST API
- Flow: POST /solve → prefetch all reference data → send to Gemini with system prompt → agentic tool loop (max 15 iterations, 200s hard stop) → return {"status": "completed"}

---

## BUG #1: requirements.txt is wrong
`requirements.txt` lists `anthropic` but the code uses `google-genai`. Fix:
```
fastapi
uvicorn
requests
google-genai
```

## BUG #2: TODAY is computed at import time (line 17)
```python
TODAY = date.today().isoformat()
```
If server runs overnight, date is stale. Move into the request handler or `run_agent()`.

---

## HIGH-PRIORITY FIXES (score impact)

### FIX 1: Salary/Payroll — currently 0%, endpoints now discovered

From sandbox API exploration, these endpoints are CONFIRMED WORKING:

```
GET  /salary/type?fields=*&count=100          → returns salary types (Fastlønn #2000, Timelønn #2001, etc.)
GET  /salary/payslip?fields=*&employeeId=X    → returns payslips (empty until payroll run)
GET  /salary/settings?fields=*                → returns {municipality, payrollTaxCalcMethod}
GET  /salary/compilation?employeeId=X&fields=* → returns {employee, year, vacationPayBasis, wages:[], expenses:[], taxDeductions:[]}
POST /salary/payslip                          → CREATE payslip. Requires at minimum: employee:{id}
```

CONFIRMED NOT WORKING:
```
/salary/transaction  → 403 (no permission)
/salary/specification → 403 (no permission)  
/salary/payment      → 404 (doesn't exist)
/salary/run          → 404 (doesn't exist)
```

The POST /salary/payslip schema is NOT YET FULLY KNOWN. The OpenAPI spec is available at:
`GET /v2/openapi.json` — search for "salary" paths and the payslip request body schema to get exact fields.

Key salary types discovered:
- #1000 "Gjeld til ansatte" (debt to employees)
- #2000 "Fastlønn" (fixed salary) — this is the main one for monthly salary
- #2001 "Timelønn" (hourly wage)
- #5001 "Kilometergodtgjørelse bil" (mileage allowance)

**Action needed:** Add salary endpoint documentation to the system prompt. The agent currently has ZERO knowledge of salary endpoints and guesses randomly, scoring 0%.

### FIX 2: Custom Accounting Dimensions — currently 0%, 15 wasted iterations

The agent tried 9 different endpoint paths (all 404): /dimension, /accountingDimension, /ledger/dimension, /projectCategory, /customDimension, /ledger/costCenter, /costCenter, etc.

**Action needed:** The correct endpoints are NOT YET DISCOVERED. The OpenAPI spec at `/v2/openapi.json` should be searched for "dimension" to find the real path. Until discovered, add an early-exit rule: if the task mentions "kostsenter", "dimensjon", "dimension", or "cost center" and the first 2 GET attempts return 404, STOP immediately instead of burning 15 iterations.

### FIX 3: Efficiency losses on tasks that already score 100% correctness

The efficiency bonus can DOUBLE your score but only on perfect correctness. These tasks get 100% but waste writes:

**Travel expense:** Should be ~3 clean writes. Agent does 8 writes with 2 failures.
- Root cause: Agent tries wrong field names, gets 422, retries. The system prompt already documents correct fields but the agent doesn't follow them reliably.

**Project tasks:** Agent was doing 12 writes, 8 failures on PUT /project.
- Root cause: Not doing GET-first before PUT. Progress notes say this was "fixed in system prompt" but efficiency is still poor.

**Action needed:** For these known task types, the system prompt instructions need to be more forceful/explicit. Consider adding worked examples showing the exact sequence of calls.

### FIX 4: Early exit on known-fatal errors

The agent loop has no programmatic detection of fatal errors. When it hits:
- "company has no bank account" on PUT /order/:invoice → should STOP, not retry
- 403 on first API call → should STOP (token invalid)
- Repeated 404s on unknown endpoint pattern → should STOP after 2-3 attempts

Currently the agent just feeds errors back to Gemini and hopes it stops. Add code in the agent loop (around line 467) to detect these patterns and break early:

```python
# After getting result, check for fatal errors
if status >= 400 and is_write:
    error_data = result.get("data", {})
    error_msg = ""
    if isinstance(error_data, dict):
        error_msg = error_data.get("message", "") + " " + str(error_data.get("developerMessage", ""))
    
    # Fatal: bank account missing — sandbox limitation, no point retrying
    if "bank account" in error_msg.lower() or "bankkonto" in error_msg.lower():
        stats["stop_reason"] = "fatal: no bank account"
        break
    
    # Fatal: token/permission error on first call
    if status == 403 and stats["iterations"] <= 1:
        stats["stop_reason"] = "fatal: 403 on first call"
        break
```

---

## MEDIUM-PRIORITY FIXES

### FIX 5: Invoice creation blocked in sandbox
`PUT /order/{id}/:invoice` always fails with "company has no bank account." This is a sandbox limitation, not a code bug. The agent should detect this and stop instead of retrying.

Already covered by FIX 4 early-exit logic.

### FIX 6: Employee from PDF — scoring ~45-59%
PDF parsing is unreliable. The agent sometimes misses fields from the PDF attachment. Consider being more explicit in the system prompt about extracting: firstName, lastName, email, dateOfBirth, nationalIdentityNumber, bankAccountNumber, phoneNumberMobile from PDF content.

### FIX 7: Bank reconciliation — scoring ~50%
Supplier voucher postings still fail sometimes. The voucher rules in the system prompt are correct but complex. The agent sometimes includes vatType in postings (which triggers "system-generated row" error) or posts to system-managed accounts (2400, 2600).

---

## LOWER-PRIORITY (architecture improvements, do after fixing the above)

### Context size optimization
All 457+ ledger accounts are dumped into every prompt, even for "create department" tasks. The `detect_task_type()` function already classifies tasks but the result is only used for logging, not for filtering context. Could reduce token usage by only including relevant context per task type.

### Prefetch optimization  
`prefetch()` makes ~12 sequential HTTP calls for every task. Could skip irrelevant ones based on task type (e.g., don't fetch assets/travel_expenses for a "create department" task) or parallelize with asyncio.gather.

### Parallel GET execution
When Gemini returns multiple GET calls in one turn, they're executed sequentially. GETs are free — executing them in parallel would save time against the 200s hard stop.

---

## IMPORTANT: What NOT to change
- The core architecture (single file, Gemini agentic loop, prefetch pattern) is working fine at 50-56%
- Don't refactor for refactoring's sake — focus on the specific fixes above
- The system prompt is large but mostly correct — add to it, don't rewrite it
- `--workers 3` in uvicorn is required for concurrency, don't remove it

## API Discovery Still Needed
The full OpenAPI spec is available at the sandbox: `GET https://kkpqfuj-amager.tripletex.dev/v2/openapi.json`
This should be fetched and searched for:
1. `salary/payslip` POST schema (exact required fields)
2. Any path containing "dimension" (for custom accounting dimensions)
3. `asset` POST schema (current field names in system prompt are confirmed WRONG — acquisitionDate and depreciationAccountNumber don't exist)

## File Reference
- `main.py` — the single agent file, all code lives here
- `progress.md` — detailed task-by-task results and API patterns
- Run with: `uvicorn main:app --host 0.0.0.0 --port 8000 --workers 3`
- Expose with: `ngrok http 8000` (NOT cloudflare — 120s timeout too short)