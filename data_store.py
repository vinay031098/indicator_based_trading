"""
Backward-compatibility shim.

The hand-rolled sqlite3 implementation was replaced by the SQLAlchemy layer in
db.py (Tasks 30-32). This module re-exports the same public functions so older
imports (`from data_store import ...`) keep working.
"""

from db import (  # noqa: F401
    add_alert_rule,
    add_to_watchlist,
    create_run,
    delete_alert_rule,
    get_available_dates,
    get_run_by_date,
    get_stored_analysis,
    init_db,
    list_alert_rules,
    list_watchlist,
    remove_from_watchlist,
    save_ai_recommendations,
    save_stock_analysis,
    update_run_status,
)
