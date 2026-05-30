#!/usr/bin/env python3
"""
Small management CLI for one-off setup tasks.

  python manage.py hash-password 'mySecret'   # -> DASHBOARD_PASSWORD_HASH value
  python manage.py gen-secret-key             # -> SECRET_KEY value
  python manage.py gen-encryption-key         # -> TOKEN_ENCRYPTION_KEY value
  python manage.py init-db                    # create database tables
"""

from __future__ import annotations

import secrets
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1]

    if cmd == "hash-password":
        if len(sys.argv) < 3:
            print("Usage: python manage.py hash-password '<password>'")
            return 1
        from security import make_password_hash

        print(make_password_hash(sys.argv[2]))
    elif cmd == "gen-secret-key":
        print(secrets.token_hex(32))
    elif cmd == "gen-encryption-key":
        from security import generate_encryption_key

        print(generate_encryption_key())
    elif cmd == "init-db":
        from db import init_db

        init_db()
        print("Database initialized.")
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
