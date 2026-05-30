"""
Notification half of Task 46 — evaluate alert rules against a fresh analysis run
and dispatch digest notifications over email (SMTP) or Telegram.

This module is intentionally **best-effort and fully defensive**: any failure
(missing credentials, network errors, malformed rules) is logged and swallowed
so it can never break the analysis pipeline that calls it.

Configuration is read from the environment directly (the typed ``settings``
object deliberately does not carry SMTP/Telegram secrets):

  Email (SMTP):
    SMTP_HOST       e.g. smtp.gmail.com
    SMTP_PORT       e.g. 587  (default 587)
    SMTP_USER       SMTP username
    SMTP_PASSWORD   SMTP password / app password
    SMTP_FROM       From address (defaults to SMTP_USER)

  Telegram:
    TELEGRAM_BOT_TOKEN   Bot API token; the chat id comes from each rule's target.

Rules come from ``db.list_alert_rules(enabled_only=True)`` and have the shape:
    {id, name, metric, operator, threshold, channel, target, enabled}
where ``metric`` is one of {score, rsi, signal, bear_score, change_pct, ...},
``operator`` is one of {">=", "<=", "==", ">", "<"} and ``channel`` is
{email, telegram}.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage
from typing import Dict, List, Optional

from db import list_alert_rules

logger = logging.getLogger(__name__)

# The actionable signal we match when a rule targets the categorical `signal`.
_TARGET_SIGNAL = "BUY"

_OPERATORS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


# ─── Matching ──────────────────────────────────────────────────────────────

def _matches(stock: Dict, metric: str, operator: str, threshold: float) -> bool:
    """Return True if ``stock`` satisfies ``metric operator threshold``."""
    try:
        if metric == "signal":
            # Categorical metric: notify on the actionable BUY signal.
            return str(stock.get("signal", "")).upper() == _TARGET_SIGNAL
        value = stock.get(metric)
        if value is None:
            return False
        op = _OPERATORS.get(operator)
        if op is None:
            return False
        return bool(op(float(value), float(threshold)))
    except (TypeError, ValueError):
        return False


def _find_matches(results: List[Dict], rule: Dict) -> List[Dict]:
    metric = rule.get("metric", "score")
    operator = rule.get("operator", ">=")
    threshold = rule.get("threshold", 0)
    return [s for s in results if _matches(s, metric, operator, threshold)]


def _build_digest(rule: Dict, matches: List[Dict]) -> str:
    metric = rule.get("metric", "score")
    operator = rule.get("operator", ">=")
    threshold = rule.get("threshold", 0)
    name = rule.get("name") or f"{metric} {operator} {threshold}"

    if metric == "signal":
        condition = f"signal == {_TARGET_SIGNAL}"
    else:
        condition = f"{metric} {operator} {threshold}"

    lines = [
        f"Alert: {name}",
        f"Condition: {condition}",
        f"{len(matches)} stock(s) matched:",
        "",
    ]
    for s in matches[:50]:
        lines.append(
            f"  {s.get('name', s.get('symbol', '?')):<14} "
            f"price={s.get('price', 'NA')}  "
            f"score={s.get('score', 'NA')}  "
            f"signal={s.get('signal', 'NA')}  "
            f"rsi={s.get('rsi', 'NA')}  "
            f"chg%={s.get('change_pct', 'NA')}"
        )
    if len(matches) > 50:
        lines.append(f"  ... and {len(matches) - 50} more")
    return "\n".join(lines)


# ─── Senders ─────────────────────────────────────────────────────────────────

def send_email(to: str, subject: str, body: str) -> bool:
    """Send a plain-text email via SMTP. Returns True on success.

    Reads SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASSWORD / SMTP_FROM from the
    environment. Missing config logs a warning and returns False (never raises).
    """
    if not to:
        logger.warning("send_email: no recipient; skipping.")
        return False

    host = os.environ.get("SMTP_HOST")
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    sender = os.environ.get("SMTP_FROM") or user
    try:
        port = int(os.environ.get("SMTP_PORT", "587"))
    except ValueError:
        port = 587

    if not host or not sender:
        logger.warning("send_email: SMTP not configured (need SMTP_HOST/SMTP_FROM); skipping.")
        return False

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        msg.set_content(body)

        with smtplib.SMTP(host, port, timeout=20) as server:
            try:
                server.starttls()
            except smtplib.SMTPException:
                # Server may not support STARTTLS (e.g. local relay); continue.
                logger.debug("send_email: STARTTLS unavailable, sending unencrypted.")
            if user and password:
                server.login(user, password)
            server.send_message(msg)
        logger.info("Alert email sent to %s", to)
        return True
    except Exception as exc:
        logger.warning("send_email failed (to=%s): %s", to, exc)
        return False


def send_telegram(chat_id: str, text: str) -> bool:
    """Send a Telegram message via the Bot API. Returns True on success.

    Reads TELEGRAM_BOT_TOKEN from the environment; ``chat_id`` is the rule's
    target. Missing token logs a warning and returns False (never raises).
    """
    if not chat_id:
        logger.warning("send_telegram: no chat_id; skipping.")
        return False

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.warning("send_telegram: TELEGRAM_BOT_TOKEN not configured; skipping.")
        return False

    try:
        import requests  # local import keeps module import light/clean

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(
            url,
            json={"chat_id": str(chat_id), "text": text, "disable_web_page_preview": True},
            timeout=20,
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            logger.info("Alert telegram sent to chat %s", chat_id)
            return True
        logger.warning("send_telegram failed (chat=%s): HTTP %s %s",
                       chat_id, resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        logger.warning("send_telegram failed (chat=%s): %s", chat_id, exc)
        return False


def _dispatch(rule: Dict, subject: str, body: str) -> bool:
    channel = (rule.get("channel") or "email").lower()
    target = rule.get("target") or ""
    if channel == "telegram":
        return send_telegram(target, f"{subject}\n\n{body}")
    if channel == "email":
        return send_email(target, subject, body)
    logger.warning("Alert rule %s has unknown channel %r; skipping.", rule.get("id"), channel)
    return False


# ─── Public entry point ──────────────────────────────────────────────────────

def evaluate_and_notify(results: List[Dict]) -> int:
    """Evaluate all enabled alert rules against ``results`` and send digests.

    ``results`` is the list of stock dicts produced by a run. Returns the number
    of notifications successfully sent. Never raises.
    """
    if not results:
        return 0

    try:
        rules = list_alert_rules(enabled_only=True)
    except Exception as exc:
        logger.warning("evaluate_and_notify: could not load alert rules: %s", exc)
        return 0

    sent = 0
    for rule in rules:
        try:
            matches = _find_matches(results, rule)
            if not matches:
                continue
            subject = f"[Trading Alert] {rule.get('name') or rule.get('metric', 'rule')} — {len(matches)} match(es)"
            body = _build_digest(rule, matches)
            if _dispatch(rule, subject, body):
                sent += 1
        except Exception as exc:
            logger.warning("evaluate_and_notify: rule %s failed: %s", rule.get("id"), exc)

    if sent:
        logger.info("evaluate_and_notify: sent %d notification(s)", sent)
    return sent
