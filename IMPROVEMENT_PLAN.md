# Improvement Plan — NSE Indicator-Based Trading Dashboard

A dependency-ordered, agent-executable plan covering all 50 improvements.
Each task has: **Goal**, **Files**, **Steps**, **Acceptance criteria**, **Depends on**, **Effort**.

## How to use this document (for the agent)

- Work **phase by phase, top to bottom**. Phases are ordered by dependency and risk.
- Complete every task in a phase before moving on, unless a task is marked `[parallel-ok]`.
- For each task: make the change on a feature branch named `task-<ID>-<slug>`, open a PR, ensure acceptance criteria pass, then merge.
- Never commit secrets. Never push `.env`, `.fyers_token`, or `analysis_data.db`.
- After each phase, run the full test suite and the app locally to confirm nothing regressed.
- Effort key: **S** = <2h, **M** = half-day, **L** = 1–2 days.

## Conventions to establish before starting

- Python: format with `black`, lint with `ruff`, type-check with `mypy` (added in Phase 6 but follow style from the start).
- Branch names: `task-01-remove-secrets`, etc.
- Commit style: matches existing history (imperative, e.g. "Add ...", "Fix ...").
- Keep PRs small and focused on a single task ID where possible.

---

# Phase 0 — Critical Security & Hygiene (do first)

> These block safe deployment. Nothing else should ship before these.

## Task 1 — Remove hardcoded secrets, load from env only
- **Goal:** No credentials in source. App reads `FYERS_APP_ID`, `FYERS_SECRET_ID`, `SECRET_KEY`, `GITHUB_TOKEN`, `GEMINI_API_KEY` from environment; in production, missing values raise a clear startup error.
- **Files:** `app.py`, `daily_runner.py`, `auto_trade.py`, `main.py`, `flyers_integration.py`, new `config.py`.
- **Steps:**
  1. Create `config.py` with a single `Settings` loader (see Task 14) that reads env vars.
  2. Replace all literal defaults like `"JIHLRUYWGE-100"` / `"DZQQB3O1GS"` with `settings.fyers_app_id` etc.
  3. If `PRODUCTION=1` and any required secret is empty, `raise RuntimeError` with a descriptive message at startup.
  4. Keep dev fallback only behind `PRODUCTION=0` and print a visible warning.
- **Acceptance criteria:** `rg "JIHLRUYWGE|DZQQB3O1GS|abcd@1234" --hidden` returns no matches in tracked files; app refuses to boot in production without env vars set.
- **Depends on:** none (but coordinate with Task 14).
- **Effort:** M

## Task 2 — Rotate exposed credentials & password
- **Goal:** The currently-leaked App ID/Secret and dashboard password are invalidated and replaced.
- **Files:** none (operational) + update `.env.example` placeholders.
- **Steps:**
  1. In the Fyers dashboard, regenerate the App Secret (and App ID if possible).
  2. Choose a new dashboard password; store its hash via Task 3.
  3. Update local `.env` and the deploy host's env vars.
  4. Note in PR description that old values are compromised and must not be reused.
- **Acceptance criteria:** Old secret no longer authenticates; `.env.example` contains only placeholders, never real values.
- **Depends on:** Task 1.
- **Effort:** S

## Task 3 — Server-side authentication
- **Goal:** Replace the JavaScript `user === 'Trader'` check with real server-side auth.
- **Files:** `app.py` (new `/auth/dashboard-login`, `/auth/logout`), `templates/index.html`, new `auth.py`.
- **Steps:**
  1. Store a single dashboard user with a **hashed** password (`werkzeug.security.generate_password_hash`), hash provided via env var `DASHBOARD_PASSWORD_HASH`.
  2. Add a login POST endpoint that sets a signed Flask session cookie on success.
  3. Remove the hardcoded credential check from the frontend; the login form now POSTs to the backend.
  4. Add a `logout` route that clears the session.
- **Acceptance criteria:** Viewing page source reveals no password; API returns 401 without a valid session; login/logout works end to end.
- **Depends on:** Task 1.
- **Effort:** M

## Task 4 — Protect all `/api/*` routes server-side
- **Goal:** Every data/action endpoint requires an authenticated session.
- **Files:** `app.py`, `auth.py`.
- **Steps:**
  1. Create a `@login_required` decorator that checks the session and returns JSON 401 otherwise.
  2. Apply it to all `/api/*` routes except a public `/healthz`.
  3. Keep Fyers `/auth/*` routes available but still session-gated where appropriate.
- **Acceptance criteria:** Hitting any `/api/*` route without a session returns 401; authenticated requests succeed.
- **Depends on:** Task 3.
- **Effort:** S

## Task 5 — Rate limiting
- **Goal:** Throttle abuse-prone endpoints.
- **Files:** `app.py`, `requirements.txt`.
- **Steps:**
  1. Add `flask-limiter`.
  2. Apply limits: `/auth/*` (e.g. 10/min), `/api/chat` (e.g. 20/min), `/api/analyze` & `/api/run-daily` (e.g. 3/min).
  3. Return JSON 429 with a friendly message.
- **Acceptance criteria:** Exceeding the limit returns 429; normal use is unaffected.
- **Depends on:** Task 4.
- **Effort:** S

## Task 6 — Security headers + HTTPS enforcement
- **Goal:** Standard hardening headers in production.
- **Files:** `app.py`, `requirements.txt`.
- **Steps:**
  1. Add `flask-talisman` (or set headers manually).
  2. Enable HSTS, X-Frame-Options=DENY, X-Content-Type-Options=nosniff, Referrer-Policy.
  3. Add a CSP allowing only required sources (self + chart CDN if used). Start in report-only, then enforce.
- **Acceptance criteria:** Response headers include the above in production; app still renders correctly.
- **Depends on:** none. `[parallel-ok]`
- **Effort:** S

## Task 7 — CSRF protection
- **Goal:** Protect state-changing POSTs.
- **Files:** `app.py`, `templates/index.html`, `requirements.txt`.
- **Steps:**
  1. Add `flask-wtf` CSRF or a custom token tied to the session.
  2. Inject a CSRF token into the page; send it via header on every POST `fetch`.
  3. Validate on the server for all POST routes.
- **Acceptance criteria:** POST without a valid CSRF token is rejected; UI flows still work.
- **Depends on:** Task 3.
- **Effort:** M

## Task 8 — Secure token storage
- **Goal:** Stop storing the Fyers access token as plaintext in a repo-relative file.
- **Files:** `flyers_integration.py`, `app.py`, `daily_runner.py`, new storage helper.
- **Steps:**
  1. Move token storage to the DB (Task 30) or an encrypted file using `cryptography.Fernet` with a key from env.
  2. Remove any logging of token prefixes (e.g. `token[:8]`).
  3. Ensure token file path is outside the repo and gitignored.
- **Acceptance criteria:** No plaintext token in repo; no token fragments in logs; auth still persists across restarts.
- **Depends on:** ideally Task 30 (or do encrypted-file interim).
- **Effort:** M

## Task 9 — Sanitize chat input / output
- **Goal:** Reduce prompt-injection and XSS via the chatbot.
- **Files:** `app.py` (`/api/chat`), `templates/index.html` (chat render).
- **Steps:**
  1. Enforce a max input length and strip control characters server-side.
  2. Render AI replies as sanitized HTML (allowlist tags) or escape and convert known-safe formatting only.
  3. Add a system-prompt guard reminding the model to ignore instructions in user data.
- **Acceptance criteria:** Injecting `<script>` in chat does not execute; oversized inputs are rejected.
- **Depends on:** Task 4.
- **Effort:** S

---

# Phase 1 — Backend Architecture Foundation

## Task 14 — Typed config object
- **Goal:** One validated settings source.
- **Files:** new `config.py`, all modules reading env.
- **Steps:**
  1. Add `pydantic-settings`; define `Settings` with typed fields + defaults + `PRODUCTION` flag.
  2. Instantiate once (`settings = Settings()`); import everywhere instead of `os.environ.get`.
  3. Validate required-in-prod fields.
- **Acceptance criteria:** All env reads go through `settings`; invalid/missing prod config fails fast with a clear message.
- **Depends on:** Task 1.
- **Effort:** M

## Task 15 — Replace `print()` with logging
- **Goal:** Structured, level-based logs.
- **Files:** all `.py` files.
- **Steps:**
  1. Configure root logger in `config.py`/app factory (format, level via env `LOG_LEVEL`).
  2. Replace `print(...)` with `logger.info/debug/warning/error`.
  3. Keep emoji-free log messages; use levels appropriately.
- **Acceptance criteria:** No `print(` remains in app code (scripts may keep CLI output); logs show timestamps + levels.
- **Depends on:** Task 14.
- **Effort:** M

## Task 10 — Flask app factory + Blueprints
- **Goal:** Modular, testable backend.
- **Files:** `app.py` → split into `app/__init__.py` (factory), `app/routes/auth.py`, `app/routes/analysis.py`, `app/routes/data.py`, `app/routes/chat.py`; update `wsgi.py`.
- **Steps:**
  1. Create `create_app()` that builds the Flask app, registers blueprints, error handlers, extensions.
  2. Move routes into blueprints by domain.
  3. Update `wsgi.py` and `__main__` to use the factory.
- **Acceptance criteria:** App runs identically; routes respond as before; `create_app()` usable in tests.
- **Depends on:** Tasks 14, 15.
- **Effort:** L

## Task 11 — Remove global mutable Fyers client
- **Goal:** Thread/worker-safe client access.
- **Files:** `flyers_integration.py`, `app/routes/*`.
- **Steps:**
  1. Introduce a `FyersClientProvider` that loads token from storage (Task 8/30) and builds a client on demand.
  2. Replace `global fyers_client` usage with provider calls.
  3. Ensure each request gets a valid client without shared mutable state.
- **Acceptance criteria:** No `global fyers_client`; concurrent requests work with 2+ gunicorn workers.
- **Depends on:** Tasks 8/30, 10.
- **Effort:** L

## Task 16 — Centralized error handling
- **Goal:** Consistent JSON error responses.
- **Files:** `app/__init__.py`, new `app/errors.py`.
- **Steps:**
  1. Define custom exceptions (`AuthError`, `FyersError`, `ValidationError`).
  2. Register Flask error handlers returning `{ "error": ..., "code": ... }` with proper status.
  3. Replace ad-hoc `jsonify({"error":...}), 500` blocks with raised exceptions.
- **Acceptance criteria:** All API errors share one shape; unexpected exceptions return 500 with a safe message (details only in logs).
- **Depends on:** Task 10.
- **Effort:** M

## Task 13 — Health-check endpoint
- **Goal:** Platform liveness/readiness probe.
- **Files:** `app/routes/data.py` (or a `meta` blueprint).
- **Steps:**
  1. Add public `GET /healthz` returning `{status:"ok", db:bool, fyers_token:bool, time:...}`.
  2. Wire into `render.yaml`/Dockerfile healthcheck.
- **Acceptance criteria:** `/healthz` returns 200 without auth and reflects DB/token state.
- **Depends on:** Task 10. `[parallel-ok]`
- **Effort:** S

## Task 17 — Fix `auto_trade.py` portability
- **Goal:** Remove hardcoded local `sys.path`.
- **Files:** `auto_trade.py`.
- **Steps:**
  1. Delete the `sys.path.insert(... Google Drive ...)` line.
  2. Use relative imports / package imports consistent with the new structure.
  3. Read credentials from `settings`.
- **Acceptance criteria:** Script runs from a fresh clone on any machine without path edits.
- **Depends on:** Tasks 10, 14.
- **Effort:** S

## Task 18 — Rename `flyers_*` → `fyers_*`
- **Goal:** Consistent naming with the brand.
- **Files:** `flyers_integration.py` → `fyers_integration.py`; all imports; `.env.example` (`FLYERS_*` leftovers).
- **Steps:**
  1. Rename file and update imports.
  2. Replace residual `FLYERS_` strings.
  3. Grep to confirm no stale references.
- **Acceptance criteria:** `rg -i "flyers"` returns no matches; app runs.
- **Depends on:** Task 10 (do during refactor to limit churn).
- **Effort:** S

---

# Phase 2 — Trading Logic & Correctness

> Add tests (Task 27) alongside each change so correctness is verifiable.

## Task 27 — Indicator unit tests (do first in this phase)
- **Goal:** Reference-checked tests for every indicator.
- **Files:** new `tests/test_indicators.py`, `tests/data/` fixtures.
- **Steps:**
  1. Build small fixed OHLCV fixtures with known indicator values (compute via `pandas-ta`/TA-Lib as reference).
  2. Assert each `_rsi/_atr/_adx/_macd/stoch/...` within tolerance.
  3. Add scoring tests for representative inputs.
- **Acceptance criteria:** `pytest tests/test_indicators.py` passes; tests fail if formulas regress.
- **Depends on:** none. `[parallel-ok]`
- **Effort:** M

## Task 19 — Wilder smoothing for RSI/ATR/ADX
- **Goal:** Standard-accurate indicators.
- **Files:** `indicators.py`.
- **Steps:**
  1. Replace simple-mean RSI with Wilder's smoothed averages.
  2. Use Wilder smoothing in ATR and ADX (or adopt `pandas-ta`).
  3. Re-baseline tests from Task 27.
- **Acceptance criteria:** Values match reference library within tolerance; tests pass.
- **Depends on:** Task 27.
- **Effort:** M

## Task 26 — Vectorize indicators
- **Goal:** Speed up analysis for 2100 stocks.
- **Files:** `indicators.py`.
- **Steps:**
  1. Replace Python `for` loops (EMA, ATR, OBV, ADX) with vectorized NumPy/pandas or `pandas-ta`.
  2. Benchmark before/after on a 200-stock run.
- **Acceptance criteria:** Per-stock analysis time drops meaningfully (target ≥3× on full set); outputs unchanged within tolerance (tests pass).
- **Depends on:** Tasks 19, 27.
- **Effort:** M

## Task 22 — Fix Stochastic %D
- **Goal:** Correct, readable %K/%D.
- **Files:** `indicators.py`.
- **Steps:**
  1. Compute %K series cleanly, then %D as a 3-period SMA of %K.
  2. Remove the `-i or None` slicing hack.
- **Acceptance criteria:** %D matches reference; no edge-case index errors on short series; tests pass.
- **Depends on:** Task 27.
- **Effort:** S

## Task 23 — Session-anchored VWAP
- **Goal:** Meaningful VWAP.
- **Files:** `indicators.py`.
- **Steps:**
  1. Anchor VWAP to each trading session (reset cumulative sums daily) or use a rolling window; document the choice.
  2. Update tests and any UI labels.
- **Acceptance criteria:** VWAP no longer drifts across the whole history; tests verify reset behavior.
- **Depends on:** Task 27.
- **Effort:** M

## Task 20 — Symmetric bullish/bearish scoring
- **Goal:** Score reflects both buy and sell pressure.
- **Files:** `indicators.py`, possibly `data_store.py` (new fields), UI.
- **Steps:**
  1. Add negative points / a separate `bear_score` for bearish conditions (overbought RSI, death cross, below cloud, DI- dominance, etc.).
  2. Produce a net signal and/or a `BUY/NEUTRAL/SELL` label independent of the LLM.
  3. Persist new fields; expose in API and UI filters.
- **Acceptance criteria:** Clearly bearish fixtures get low/negative scores; tests cover both directions.
- **Depends on:** Tasks 19, 27.
- **Effort:** M

## Task 21 — Mutually exclusive contradictory signals
- **Goal:** Prevent simultaneously firing opposite signals.
- **Files:** `indicators.py`.
- **Steps:**
  1. Group related rules (e.g. 52W-high vs 52W-low proximity) and pick the dominant one.
  2. Add tests asserting both can't trigger together.
- **Acceptance criteria:** No fixture produces contradictory `reasons`; tests pass.
- **Depends on:** Task 20.
- **Effort:** S

## Task 25 — Configurable periods & weights
- **Goal:** No magic numbers; tunable strategy.
- **Files:** `indicators.py`, new `strategy.yaml`/config section.
- **Steps:**
  1. Extract indicator periods and score weights into a config file loaded at startup.
  2. Reference config in scoring; validate ranges.
- **Acceptance criteria:** Changing a weight in config changes scoring without code edits; tests use a known config.
- **Depends on:** Tasks 14, 20.
- **Effort:** M

## Task 24 — Don't swallow exceptions in `analyze_stock`
- **Goal:** Surface real errors.
- **Files:** `indicators.py`.
- **Steps:**
  1. Catch narrowly; log `symbol` + exception with stack at debug level.
  2. Return `None` only after logging; add a counter for skipped/errored symbols.
- **Acceptance criteria:** Induced error in one symbol is logged with its name; run continues.
- **Depends on:** Task 15.
- **Effort:** S

---

# Phase 3 — Performance & Data Layer

## Task 31 — Remove `analysis_data.db` from git
- **Goal:** Stop committing the DB.
- **Files:** `.gitignore`, repo.
- **Steps:**
  1. `git rm --cached analysis_data.db`; add to `.gitignore`.
  2. Provide a seed script `scripts/seed_db.py` or document how to regenerate via `daily_runner.py`.
- **Acceptance criteria:** DB no longer tracked; fresh clone builds an empty DB on first run.
- **Depends on:** none. `[parallel-ok]`
- **Effort:** S

## Task 30 — Migrate SQLite → Postgres
- **Goal:** Durable, concurrent storage that survives redeploys.
- **Files:** `data_store.py` → `db.py` with SQLAlchemy; `requirements.txt`; `render.yaml`.
- **Steps:**
  1. Introduce SQLAlchemy models for `daily_runs`, `stock_analysis`, `ai_recommendations`.
  2. Support `DATABASE_URL` (Postgres in prod, SQLite fallback in dev).
  3. Add Alembic migrations; port existing read/write functions.
- **Acceptance criteria:** App works against Postgres; data persists across redeploys; existing endpoints unchanged.
- **Depends on:** Tasks 14, 16.
- **Effort:** L

## Task 32 — Connection pooling / batch writes
- **Goal:** Efficient DB usage.
- **Files:** `db.py`.
- **Steps:**
  1. Use SQLAlchemy engine pool; reuse sessions.
  2. Batch-insert stock rows in one transaction (executemany / bulk_save).
- **Acceptance criteria:** Saving a full run uses a single transaction; no per-row connect/close.
- **Depends on:** Task 30.
- **Effort:** S

## Task 29 — Persistent symbol-master cache
- **Goal:** Avoid re-downloading the NSE master on every process start.
- **Files:** `fyers_integration.py`.
- **Steps:**
  1. Cache the parsed symbol list to a local JSON/parquet with a timestamp.
  2. Refresh only when older than 24h; fall back to cache on download failure.
- **Acceptance criteria:** Second startup within 24h uses cache (no network); stale cache refreshes.
- **Depends on:** none. `[parallel-ok]`
- **Effort:** S

## Task 28 — Parallel Fyers fetching
- **Goal:** Faster data download within rate limits.
- **Files:** `fyers_integration.py`.
- **Steps:**
  1. Replace sequential loop with a bounded `ThreadPoolExecutor` (e.g. 6–8 workers) plus a shared rate limiter (token bucket ≤180/min, ≤10/s).
  2. Keep retry/backoff for failures.
  3. Benchmark on 200 stocks.
- **Acceptance criteria:** Full-set fetch is substantially faster; stays within Fyers limits (no sustained 429s).
- **Depends on:** Task 11.
- **Effort:** M

## Task 12 — Background job queue for analysis
- **Goal:** Long runs don't block/timeout web requests.
- **Files:** new `worker.py`, `app/routes/analysis.py`, `requirements.txt`, `render.yaml` (worker service).
- **Steps:**
  1. Add RQ (or Celery) + Redis. `/api/analyze` and `/api/run-daily` enqueue a job and return a `job_id`.
  2. Add `/api/jobs/<id>` for status/progress; worker updates progress in Redis/DB.
  3. Frontend polls (or SSE/WebSocket) for progress.
- **Acceptance criteria:** A 2100-stock run completes via worker without HTTP timeout; UI shows live progress.
- **Depends on:** Tasks 10, 11, 28, 30.
- **Effort:** L

## Task 33 — Response caching for stored data
- **Goal:** Cheap repeat reads.
- **Files:** `app/routes/data.py`.
- **Steps:**
  1. Cache `/api/stored-dates` and stored-data responses (Flask-Caching, short TTL + invalidate on new run).
  2. Add ETag/Last-Modified where sensible.
- **Acceptance criteria:** Repeated reads hit cache; new runs invalidate it.
- **Depends on:** Task 30.
- **Effort:** S

## Task 34 — Pin dependency versions
- **Goal:** Reproducible builds.
- **Files:** `requirements.txt` (+ optional `requirements.lock`/`pyproject.toml`).
- **Steps:**
  1. Pin exact versions for all deps (and new ones added in this plan).
  2. Optionally adopt `pip-tools` or `uv` for lockfiles.
- **Acceptance criteria:** Fresh install resolves identical versions; CI build is deterministic.
- **Depends on:** done late so all new deps are captured.
- **Effort:** S

---

# Phase 4 — Frontend / UI / UX

> Big refactor: turning one HTML file into a maintainable, modern UI.

## Task 35 — Split monolithic `index.html`
- **Goal:** Separate concerns; enable tooling.
- **Files:** `templates/index.html` → `templates/` partials + `static/css/*.css` + `static/js/*.js`. Consider a light framework (Vue 3 or React + Vite) if scope allows.
- **Steps:**
  1. Extract CSS to stylesheets and JS to modules.
  2. Split markup into logical partials (header, controls, grid, modals).
  3. Decide: keep vanilla JS modules, or migrate to Vite + Vue/React (recommended for the later tasks).
- **Acceptance criteria:** No inline `<style>`/`<script>` blobs; app renders identically; assets served from `static/`.
- **Depends on:** none. `[parallel-ok]` (coordinate with backend route names)
- **Effort:** L

## Task 39 — Accessibility pass
- **Goal:** Not color/emoji-only; keyboard + screen-reader friendly.
- **Files:** frontend.
- **Steps:**
  1. Add text labels alongside color/emoji for BUY/SELL/NEUTRAL; add `aria-label`s.
  2. Ensure focus states, tab order, and modal focus trapping.
  3. Check contrast ratios (WCAG AA).
- **Acceptance criteria:** Lighthouse a11y ≥ 90; keyboard-only navigation works; signals distinguishable without color.
- **Depends on:** Task 35.
- **Effort:** M

## Task 40 — Responsive layout
- **Goal:** Usable on mobile/tablet.
- **Files:** frontend CSS.
- **Steps:**
  1. Replace fixed `min-width` controls/grid with responsive grid + breakpoints.
  2. Make header controls wrap/stack on small screens.
- **Acceptance criteria:** No horizontal scroll at 375px; grid reflows across breakpoints.
- **Depends on:** Task 35.
- **Effort:** M

## Task 41 — Theme toggle (light/dark)
- **Goal:** User-selectable theme, persisted.
- **Files:** frontend.
- **Steps:**
  1. Define light theme variables; add a toggle.
  2. Persist choice in `localStorage`; respect `prefers-color-scheme` default.
- **Acceptance criteria:** Toggle switches themes instantly and persists across reloads.
- **Depends on:** Task 35.
- **Effort:** S

## Task 42 — Replace `alert()` with toasts
- **Goal:** Non-blocking notifications.
- **Files:** frontend.
- **Steps:**
  1. Add a small toast component (success/error/info).
  2. Replace all `alert(...)` calls.
- **Acceptance criteria:** No `alert(` remains; errors/success show as toasts.
- **Depends on:** Task 35.
- **Effort:** S

## Task 43 — Real progress + skeletons
- **Goal:** Honest loading feedback tied to backend jobs.
- **Files:** frontend, `app/routes/analysis.py`.
- **Steps:**
  1. Replace the fake progress bar with progress from `/api/jobs/<id>` (Task 12).
  2. Add skeleton placeholders for the grid while loading.
- **Acceptance criteria:** Progress reflects real job state; skeletons show during load.
- **Depends on:** Tasks 12, 35.
- **Effort:** M

## Task 37 — Virtualized stock grid
- **Goal:** Smooth rendering of thousands of cards.
- **Files:** frontend.
- **Steps:**
  1. Implement list virtualization (windowing) for the grid.
  2. Render only visible cards; recycle on scroll.
- **Acceptance criteria:** Rendering 2000+ results stays responsive (no multi-second jank); scroll is smooth.
- **Depends on:** Task 35.
- **Effort:** M

## Task 38 — Sortable/filterable table view
- **Goal:** Power-user data view.
- **Files:** frontend.
- **Steps:**
  1. Add a table toggle with sortable columns (score, %chg, RSI, ADX, etc.).
  2. Combine with existing filter tabs and search.
- **Acceptance criteria:** Columns sort ascending/descending; filters + search compose correctly.
- **Depends on:** Tasks 35, 37.
- **Effort:** M

## Task 36 — Interactive candlestick charts
- **Goal:** Biggest UX upgrade — visual price + indicators.
- **Files:** frontend; new backend route `/api/history/<symbol>`.
- **Steps:**
  1. Add a backend endpoint returning OHLCV (+ selected indicator series) for a symbol/date range.
  2. Integrate TradingView Lightweight Charts; overlay SMA/EMA/Bollinger; subpanels for RSI/MACD.
  3. Lazy-load chart data on demand.
- **Acceptance criteria:** Clicking a stock shows an interactive candlestick chart with at least 2 overlays and 1 oscillator panel.
- **Depends on:** Tasks 11, 35.
- **Effort:** L

## Task 44 — Stock detail view/modal
- **Goal:** Deep dive per stock.
- **Files:** frontend.
- **Steps:**
  1. Build a modal/page: full indicator table, the chart (Task 36), AI reasoning, target/stop-loss visualized on the chart.
  2. Link from each card/table row.
- **Acceptance criteria:** Detail view shows all indicators + chart + AI rec + target/SL markers.
- **Depends on:** Tasks 36, 38.
- **Effort:** M

---

# Phase 5 — Product Features

## Task 48 — Financial disclaimer
- **Goal:** Compliance + trust.
- **Files:** frontend (footer + first-run modal).
- **Steps:**
  1. Add a persistent "Not investment advice" disclaimer in the footer.
  2. Show a one-time acknowledgment modal.
- **Acceptance criteria:** Disclaimer visible on all pages; acknowledgment stored.
- **Depends on:** Task 35. `[parallel-ok]`
- **Effort:** S

## Task 47 — Export & shareable reports
- **Goal:** Get data out / share results.
- **Files:** `app/routes/data.py`, frontend.
- **Steps:**
  1. Add CSV/Excel export of the current result set (`/api/export?date=&category=&format=`).
  2. Add shareable read-only links for a stored run (token or slug).
- **Acceptance criteria:** Export downloads a correct CSV/XLSX; share link renders the run without login (read-only) if enabled.
- **Depends on:** Task 30.
- **Effort:** M

## Task 46 — Watchlists, portfolio & alerts
- **Goal:** Personalization + notifications.
- **Files:** `db.py` (new tables), `app/routes/*`, frontend, `worker.py`.
- **Steps:**
  1. Add `watchlist` and optional `portfolio` tables tied to the user.
  2. Add alert rules (e.g. score ≥ X, RSI < Y) evaluated by the daily worker.
  3. Send alerts via email (SMTP) or Telegram bot.
- **Acceptance criteria:** User can add/remove watchlist items; a triggered rule sends a notification on the next run.
- **Depends on:** Tasks 12, 30, 3.
- **Effort:** L

## Task 45 — Backtesting module
- **Goal:** Validate the scoring strategy historically.
- **Files:** new `backtest.py`, `app/routes/*`, frontend report.
- **Steps:**
  1. For a date range, simulate: enter when score ≥ threshold, exit on rule (target/SL/time), using stored/fetched history.
  2. Compute win rate, avg return, max drawdown, equity curve.
  3. Surface a backtest report page with charts.
- **Acceptance criteria:** Running a backtest over a period returns win rate + equity curve; results are reproducible.
- **Depends on:** Tasks 19–26 (correct indicators), 30, 36 (charts).
- **Effort:** L

---

# Phase 6 — DevOps, Testing & Docs

## Task 49 — CI pipeline
- **Goal:** Automated quality gates.
- **Files:** `.github/workflows/ci.yml`, `pyproject.toml`.
- **Steps:**
  1. Add `pyproject.toml` configuring `ruff`, `black`, `mypy`, `pytest`.
  2. CI: install pinned deps, run lint + type-check + tests, build Docker image on push/PR.
  3. Add a coverage report.
- **Acceptance criteria:** PRs run CI; failing lint/tests block merge; Docker build succeeds in CI.
- **Depends on:** Tasks 27, 34.
- **Effort:** M

## Task 50 — Rewrite README + docs
- **Goal:** Accurate, complete documentation.
- **Files:** `README.md`, `docs/` (optional), screenshots.
- **Steps:**
  1. Document the Flask dashboard, stored vs live modes, env setup, Fyers app config (redirect `https://fyersapiapp.com`), local run, worker, deploy.
  2. Add an architecture diagram and screenshots.
  3. Remove outdated CLI-only instructions.
- **Acceptance criteria:** A new dev can set up and run the app end to end using only the README.
- **Depends on:** most features stable (do near the end).
- **Effort:** M

---

# Suggested execution order (summary)

1. **Phase 0** (Tasks 1, 14, 2, 3, 4, 5, 6, 7, 8, 9) — security & config foundation.
2. **Phase 1** (15, 10, 16, 13, 11, 17, 18) — architecture.
3. **Phase 2** (27, 19, 26, 22, 23, 20, 21, 25, 24) — correct logic, test-first.
4. **Phase 3** (31, 29, 30, 32, 28, 33, 12, 34) — data & performance.
5. **Phase 4** (35, 39, 40, 41, 42, 37, 38, 36, 43, 44) — UI/UX.
6. **Phase 5** (48, 47, 46, 45) — features.
7. **Phase 6** (49, 50) — CI & docs.

## Global definition of done
- All acceptance criteria met; CI green.
- No secrets, tokens, or the DB committed.
- App boots locally and in production config; `/healthz` green.
- README lets a fresh dev reproduce the setup.
