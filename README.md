# NSE Indicator-Based Trading Dashboard

A Flask web dashboard that screens NSE-listed stocks using ~25 technical indicators,
scores each one for bullish/bearish pressure, and optionally layers an LLM
"buy / hold / avoid" recommendation on top. Daily runs are persisted to a database so
you can browse historical analyses, chart any symbol, set alerts, export results, and
backtest the scoring strategy. Market data comes from the **Fyers** broker API.

> ⚠️ **Not investment advice.** This is an educational/analytical tool. See the
> [disclaimer](#disclaimer) before using any output to make trading decisions.

---

## Features

- **Two run modes**
  - **Stored mode** — browse, chart, filter, and export previously computed runs. No
    Fyers connection required.
  - **Live mode** — connect a Fyers account to fetch fresh OHLCV data and run analysis
    on demand (single category synchronously, or the full NSE universe as a background job).
- **~25 technical indicators** (`indicators.py`): RSI, MACD, Bollinger Bands, Stochastic
  %K/%D, ADX/DI, ATR, OBV, MFI, VWAP, multiple SMAs/EMAs, 52-week high/low proximity,
  golden/death cross detection, and more — with Wilder smoothing for RSI/ATR/ADX.
- **Symmetric scoring** — a `bull_score`, a `bear_score`, and a net `BUY / NEUTRAL / SELL`
  signal computed independently of the LLM. Periods and weights are configurable in
  `strategy.py`.
- **AI recommendations** (`llm_analyzer.py`) — optional pass over scored stocks via
  GitHub Models, producing `BUY / HOLD / AVOID` with reasoning, targets, and stop-losses.
- **Interactive charts** — candlestick view with indicator overlays and oscillator panels
  (served from `/api/history/<symbol>`).
- **Watchlists & alerts** — track symbols and define rules (e.g. `score >= 8`, `RSI < 30`)
  evaluated by the daily worker, with email/Telegram notifications.
- **Backtesting** — simulate the scoring strategy over a date range (win rate, average
  return, drawdown, equity curve).
- **Export & sharing** — download the current result set as CSV or XLSX.
- **Production-grade backend** — app factory + blueprints, server-side session auth,
  rate limiting, CSP/security headers, CSRF setup, encrypted token storage at rest, a
  health-check endpoint, and a Redis-backed background job queue with an in-process fallback.

---

## Architecture

The browser talks to a Flask app (an app factory registering domain blueprints). Routes
delegate to a shared service layer, which fans out to the Fyers API for data, the
indicator/scoring engine, the optional LLM, and the database. Long analysis runs are
handed off to a background worker.

```mermaid
flowchart LR
    Browser["Browser (templates/ + static/)"]

    subgraph Flask["Flask app (create_app)"]
        direction TB
        Auth["auth_routes<br/>(login, Fyers OAuth)"]
        Analysis["analysis<br/>(/api/analyze, /api/run-daily)"]
        Data["data<br/>(stored, history, export, watchlist, alerts, backtest)"]
        Chat["chat (/api/chat)"]
        Meta["meta (/, /healthz)"]
    end

    subgraph Services["Service layer"]
        direction TB
        Svc["analysis_service"]
        Ind["indicators + strategy"]
        LLM["llm_analyzer"]
    end

    Fyers["Fyers broker API"]
    DB[("Database<br/>SQLite dev / Postgres prod")]
    Redis[("Redis + RQ worker")]

    Browser -->|HTTPS / JSON| Flask
    Flask --> Services
    Svc --> Ind
    Svc --> LLM
    Svc -->|OHLCV| Fyers
    Svc -->|read/write runs| DB
    Analysis -->|enqueue long runs| Redis
    Redis -->|worker.py| Svc
    Meta --> DB
```

For a deeper component-by-component breakdown, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Tech stack

| Area              | Choices |
| ----------------- | ------- |
| Language          | Python 3.11 |
| Web framework     | Flask 3 (app factory + blueprints), Gunicorn |
| Data / analysis   | pandas, NumPy (optional `pandas-ta` for indicator test references) |
| Config            | pydantic-settings (typed `Settings`, env-driven) |
| Security          | flask-limiter, flask-talisman (CSP/HSTS), flask-wtf (CSRF), `cryptography` (Fernet) |
| Persistence       | SQLAlchemy 2 (SQLite dev / Postgres prod), Alembic |
| Jobs / cache      | Redis + RQ (in-process fallback), Flask-Caching |
| Broker data       | `fyers-apiv3` |
| LLM               | GitHub Models (PAT with `models:read`) |
| Tooling           | ruff, black, mypy, pytest (+coverage) — see `pyproject.toml` |

---

## Prerequisites

- **Python 3.11**
- **A Fyers app** (create one at <https://myapi.fyers.in/dashboard>) with the redirect URI
  set to **`https://fyersapiapp.com`**. You'll get an `APP_ID` and `SECRET_ID`.
- *(Optional)* **Redis** — enables the background job queue and Redis-backed caching.
  Without it, jobs run in-process.
- *(Optional)* **PostgreSQL** — recommended for production; SQLite is the dev default.
- *(Optional)* **A GitHub token** with `models:read` scope — enables the LLM
  recommendations via GitHub Models. Without it, the indicator scoring still works.

---

## Setup

```bash
# 1. Clone
git clone <your-fork-url> indicator_based_trading
cd indicator_based_trading

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
# The Fyers SDK declares the obsolete `asyncio` PyPI backport as a dependency,
# which shadows stdlib asyncio and breaks installs — it MUST be installed --no-deps:
pip install --no-deps fyers-apiv3==3.1.7

# 4. Create your local environment file
cp .env.example .env
```

### Generate secrets

Use the bundled management CLI to produce the values for `.env`:

```bash
python manage.py gen-secret-key                 # -> SECRET_KEY
python manage.py hash-password 'your-password'  # -> DASHBOARD_PASSWORD_HASH
python manage.py gen-encryption-key             # -> TOKEN_ENCRYPTION_KEY
```

Then edit `.env` and fill in at least:

- `SECRET_KEY`, `DASHBOARD_PASSWORD_HASH`, `TOKEN_ENCRYPTION_KEY` (from the commands above)
- `DASHBOARD_USER` (defaults to `Trader`)
- `FYERS_APP_ID`, `FYERS_SECRET_ID` (from your Fyers app)
- `GITHUB_TOKEN` (optional, for AI recommendations)

Initialize the database (creates the schema; SQLite file by default):

```bash
python manage.py init-db
```

---

## Running locally

```bash
python wsgi.py
# or, production-style:
gunicorn wsgi:app --bind 0.0.0.0:5000 --workers 2 --timeout 120
```

Open <http://127.0.0.1:5000>, then:

1. **Log in** with `DASHBOARD_USER` and the password you hashed into `DASHBOARD_PASSWORD_HASH`.
2. **Stored mode** works immediately — browse any previously stored run.
3. **Live mode**: click *Connect Fyers*, complete the OAuth flow (you'll be redirected to
   `https://fyersapiapp.com` with an `auth_code`), and paste it back into the dashboard.
   The access token is encrypted at rest with your `TOKEN_ENCRYPTION_KEY`. Once connected,
   you can run live analysis and view live charts.

Health check: `GET /healthz` returns `{ status, db, fyers_token, jobs, time }` without auth.

---

## Background jobs

Full-universe analysis (thousands of symbols) is long-running and is offloaded to a worker
so web requests never time out.

- **With Redis** (recommended): set `REDIS_URL` in `.env`, then run the worker alongside the web app:

```bash
python worker.py
```

  `/api/run-daily` enqueues a job and returns a `job_id`; poll `/api/jobs/<job_id>` for
  live progress.

- **Without Redis**: if `REDIS_URL` is unset, jobs run in an in-process background thread
  (fine for development and small universes).

---

## CLI usage

All CLIs read configuration from `.env` via the typed settings object.

### `daily_runner.py` — scheduled analysis + AI, stored to the DB

```bash
python daily_runner.py                      # all NSE stocks, today
python daily_runner.py --date 2026-02-18    # specific date (YYYY-MM-DD)
python daily_runner.py --category nifty50   # nifty50 | nifty100 | nifty200 | nifty500 | all
python daily_runner.py --min-score 3        # minimum score to qualify (default 2)
python daily_runner.py --skip-ai            # indicators only, skip the LLM pass
python daily_runner.py --force              # overwrite an existing run for that date
```

Example cron (weekdays, after market hours):

```cron
30 16 * * 1-5 cd /path/to/project && /path/to/.venv/bin/python daily_runner.py --force >> logs/daily.log 2>&1
```

> `daily_runner.py` requires a saved Fyers token — connect once via the dashboard first.

### `main.py` — quick NIFTY 50 console report

```bash
python main.py
```

Fetches NIFTY 50 via Fyers, scores each stock, and prints a ranked table (name, price,
score, signal, RSI). Prompts for Fyers auth if no token is saved.

### `auto_trade.py` — analyze and optionally place orders

```bash
python auto_trade.py
```

Scores NIFTY 50, lists the top candidates (score ≥ 4), and **interactively** asks how many
to trade before placing any orders through Fyers. It never trades without explicit confirmation.

---

## Deployment

### Render (blueprint)

`render.yaml` defines a complete Render deployment:

- a **web** service (Gunicorn) with the health check wired to `/healthz`,
- a **worker** service running `python worker.py`,
- a managed **Postgres** database, and
- a **Redis** instance.

Both services use the build command
`pip install -r requirements.txt && pip install --no-deps fyers-apiv3==3.1.7`.
`SECRET_KEY` is auto-generated; set the remaining secrets
(`DASHBOARD_PASSWORD_HASH`, `FYERS_APP_ID`, `FYERS_SECRET_ID`, `TOKEN_ENCRYPTION_KEY`,
`GITHUB_TOKEN`, `DOMAIN`) in the Render dashboard. `DATABASE_URL` and `REDIS_URL` are wired
automatically from the managed services. The `release` command runs `python manage.py init-db`.

Connect the repo in Render and it will auto-detect `render.yaml`.

### Docker

A `Dockerfile` is provided (works on Railway / Koyeb / Fly.io free tiers). It installs
dependencies (with the Fyers SDK `--no-deps`), copies the app, and starts Gunicorn:

```bash
docker build -t indicator-trading .
docker run -p 5000:5000 --env-file .env indicator-trading
```

The container honours `$PORT` (defaults to 5000).

---

## Environment variables

All configuration is read from the environment (or `.env` locally) through the typed
`Settings` object in `config.py`. In **production** (`PRODUCTION=1`), missing required
secrets cause a fail-fast startup error.

| Variable                  | Description |
| ------------------------- | ----------- |
| `PRODUCTION`              | `1` enables prod hardening (HTTPS/HSTS, secure cookies) and required-secret validation; `0` for local dev. |
| `LOG_LEVEL`               | Root log level (`DEBUG`, `INFO`, `WARNING`, ...). Default `INFO`. |
| `DOMAIN`                  | Public host (e.g. `localhost:5000` or `app.example.com`); used to build the base URL. |
| `SECRET_KEY`              | Flask session signing key. Generate with `python manage.py gen-secret-key`. **Required in prod.** |
| `DASHBOARD_USER`          | Dashboard login username. Default `Trader`. |
| `DASHBOARD_PASSWORD_HASH` | Werkzeug password **hash** (never the plaintext). Generate with `python manage.py hash-password '...'`. **Required in prod.** |
| `FYERS_APP_ID`            | Fyers app ID. **Required in prod.** |
| `FYERS_SECRET_ID`         | Fyers app secret. **Required in prod.** |
| `FYERS_REDIRECT_URI`      | OAuth redirect URI; must match the Fyers app. Default `https://fyersapiapp.com`. |
| `TOKEN_ENCRYPTION_KEY`    | Fernet key used to encrypt the saved Fyers token at rest. Generate with `python manage.py gen-encryption-key`. **Required in prod.** |
| `GITHUB_TOKEN`            | GitHub PAT with `models:read` for LLM recommendations (optional). |
| `GEMINI_API_KEY`          | Optional Gemini API key (alternative LLM provider). |
| `DATABASE_URL`            | SQLAlchemy URL. Dev default `sqlite:///analysis_data.db`; prod e.g. `postgresql+psycopg2://user:pass@host:5432/db`. |
| `REDIS_URL`               | Redis URL for the job queue + cache. If unset, jobs run in-process. |
| `MIN_SCORE_THRESHOLD`     | Default minimum score for a stock to "qualify". Default `2`. |
| `ANALYSIS_PERIOD_DAYS`    | Lookback window (days) of history fetched per symbol. Default `365`. |

See `.env.example` for a ready-to-copy template.

---

## Development & quality

```bash
pip install -r requirements-dev.txt
pip install --no-deps fyers-apiv3==3.1.7

ruff check .            # lint + import order
black --check .         # formatting
mypy .                  # (lenient) type checks
pytest -q               # tests (tests/)
pytest -q --cov=.       # tests with coverage
```

Tooling is configured in `pyproject.toml` (ruff/black at line length 100, target Python
3.11; mypy lenient; pytest discovers `tests/`). CI runs all of the above plus a Docker
image build on every push and pull request — see `.github/workflows/ci.yml`.

---

## Project layout

```
.
├── wsgi.py                 # Gunicorn / local entry point (create_app)
├── app/
│   ├── __init__.py         # app factory: extensions, blueprints, error handlers
│   ├── auth.py             # session auth helpers + @login_required
│   ├── errors.py           # custom exceptions + JSON error handlers
│   ├── extensions.py       # limiter, csrf, cache, rate_limit/cached helpers
│   └── routes/             # meta, auth_routes, analysis, data, chat blueprints
├── analysis_service.py     # shared analyze() + run_and_store() pipeline
├── indicators.py           # ~25 indicators + scoring (bull/bear/signal)
├── strategy.py             # configurable periods & weights
├── llm_analyzer.py         # GitHub Models BUY/HOLD/AVOID recommendations
├── fyers_integration.py    # Fyers client provider, parallel fetch, symbol cache
├── db.py                   # SQLAlchemy models + read/write helpers
├── security.py             # password hashing + encrypted token storage
├── jobs.py / worker.py     # RQ background jobs (+ in-process fallback)
├── daily_runner.py         # CLI: scheduled analysis -> DB
├── main.py / auto_trade.py # CLI: console report / interactive trading
├── manage.py               # secrets + DB init CLI
├── templates/ , static/    # frontend
├── render.yaml , Dockerfile, Procfile
└── pyproject.toml , .github/workflows/ci.yml
```

---

## Disclaimer

**This software is for educational and informational purposes only and is _not_
investment, financial, or trading advice.** The technical indicators, scores, signals, and
AI-generated recommendations it produces may be inaccurate, incomplete, or wrong, and must
not be relied upon to make any financial decision. Trading and investing in securities
carries substantial risk, including the loss of principal. Past performance and backtested
results do not guarantee future returns. The `auto_trade.py` script can place **real
orders** through your broker — use it entirely at your own risk. Always do your own
research and consult a qualified, licensed financial advisor. The authors and contributors
accept no liability for any loss or damage arising from the use of this software.
