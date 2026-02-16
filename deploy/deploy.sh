#!/bin/bash
# ═══════════════════════════════════════════════════════════
#  Deploy NIFTY 50 Trading Dashboard to belezabrasileiro.com
#  Run this on your VPS (Ubuntu 22.04+)
# ═══════════════════════════════════════════════════════════

set -e

DOMAIN="belezabrasileiro.com"
APP_DIR="/opt/indicators_trade"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "═══════════════════════════════════════════════════════"
echo "  Deploying to $DOMAIN"
echo "═══════════════════════════════════════════════════════"

# ─── Step 1: Install system dependencies ─────────────────
echo -e "\n[1/7] Installing system packages..."
sudo apt update -qq
sudo apt install -y python3 python3-pip python3-venv nginx certbot python3-certbot-nginx

# ─── Step 2: Create app directory ─────────────────────────
echo -e "\n[2/7] Setting up app directory..."
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR

# Copy project files
cp -r "$REPO_DIR"/*.py "$APP_DIR/"
cp -r "$REPO_DIR"/templates "$APP_DIR/"
cp -r "$REPO_DIR"/requirements.txt "$APP_DIR/" 2>/dev/null || true

# ─── Step 3: Python virtual environment ──────────────────
echo -e "\n[3/7] Setting up Python environment..."
cd $APP_DIR
python3 -m venv venv
source venv/bin/activate
pip install -q flask fyers-apiv3 pandas numpy gunicorn requests python-dotenv

# ─── Step 4: Production .env file ────────────────────────
echo -e "\n[4/7] Creating .env config..."
if [ ! -f "$APP_DIR/.env" ]; then
    SECRET=$(python3 -c "import secrets; print(secrets.token_hex(32))")
    cat > "$APP_DIR/.env" << EOF
PRODUCTION=1
DOMAIN=$DOMAIN
SECRET_KEY=$SECRET
FYERS_APP_ID=HTEDSURO6P-100
FYERS_SECRET_ID=6E0U40KRQT
EOF
    echo "  Created .env with random SECRET_KEY"
else
    echo "  .env already exists, skipping"
fi

# Copy token if it exists
if [ -f "$REPO_DIR/.fyers_token" ]; then
    cp "$REPO_DIR/.fyers_token" "$APP_DIR/.fyers_token"
    echo "  Copied saved Fyers token"
fi

# ─── Step 5: Systemd service ────────────────────────────
echo -e "\n[5/7] Setting up systemd service..."
sudo cp "$REPO_DIR/deploy/indicators.service" /etc/systemd/system/
sudo chown -R www-data:www-data $APP_DIR
sudo systemctl daemon-reload
sudo systemctl enable indicators
sudo systemctl restart indicators
echo "  Service started!"

# ─── Step 6: Nginx config ───────────────────────────────
echo -e "\n[6/7] Configuring Nginx..."
sudo cp "$REPO_DIR/deploy/nginx.conf" /etc/nginx/sites-available/indicators
sudo ln -sf /etc/nginx/sites-available/indicators /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
echo "  Nginx configured!"

# ─── Step 7: SSL Certificate ────────────────────────────
echo -e "\n[7/7] Setting up SSL..."
echo "  Make sure DNS A records point to this server:"
echo "    $DOMAIN → $(curl -s ifconfig.me)"
echo "    www.$DOMAIN → $(curl -s ifconfig.me)"
echo ""
read -p "  DNS ready? Run certbot now? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    sudo certbot --nginx -d $DOMAIN -d www.$DOMAIN --non-interactive --agree-tos --email admin@$DOMAIN
    echo "  SSL certificate installed!"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  ✅ DEPLOYED SUCCESSFULLY!"
echo ""
echo "  Dashboard:  https://$DOMAIN"
echo "  Login:      https://$DOMAIN/auth/login"
echo "  Status:     sudo systemctl status indicators"
echo "  Logs:       sudo journalctl -u indicators -f"
echo ""
echo "  ⚠️  IMPORTANT: Update Fyers redirect URL to:"
echo "     https://$DOMAIN/auth/callback"
echo "     at: https://myapi.fyers.in/dashboard"
echo "═══════════════════════════════════════════════════════"
