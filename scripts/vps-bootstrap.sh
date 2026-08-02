#!/usr/bin/env bash
#
# One-time setup for a fresh OVH VPS (Debian 12, "Docker" image).
#
#   ssh debian@<vps-host>
#   curl -fsSL https://raw.githubusercontent.com/Aminpatra/Itqan/main/scripts/vps-bootstrap.sh | sudo bash
#
# Written down rather than typed from memory, because the alternative is a box
# whose configuration exists only in somebody's shell history — and the swap file
# below is not optional on 4 GB.
#
# Idempotent: safe to re-run after a rebuild or a snapshot restore.
set -euo pipefail

DEPLOY_USER="${DEPLOY_USER:-itqan}"
APP_DIR="/opt/itqan"
SWAP_SIZE="${SWAP_SIZE:-2G}"

log() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run with sudo"; exit 1; }

log "Packages"
apt-get update -qq
apt-get install -y -qq --no-install-recommends \
    ca-certificates curl git ufw unattended-upgrades postgresql-client

log "Docker"
# The OVH "Docker" image ships it, but the compose v2 plugin is not always there
# and this script also has to work on a plain Debian install.
if ! command -v docker >/dev/null; then
    curl -fsSL https://get.docker.com | sh
fi
docker compose version >/dev/null 2>&1 || apt-get install -y -qq docker-compose-plugin
systemctl enable --now docker

log "Swap (${SWAP_SIZE})"
# THE most important line in this file.
#
# 4 GB total, and the peaks overlap: Postgres ~700 MB, the API 145 MB idle, and
# OCR ~1.2 GB (measured) while it reads a scanned page — about 2.3 GB with
# everything busy. That fits, but not with room to be careless, and Linux resolves
# an overshoot by killing the largest process, which is the one doing the user's
# work. Swap turns an OOM kill into a slow request.
if ! swapon --show | grep -q '/swapfile'; then
    fallocate -l "$SWAP_SIZE" /swapfile
    chmod 600 /swapfile
    mkswap /swapfile >/dev/null
    swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
    # Prefer RAM, but use swap before killing anything.
    sysctl -qw vm.swappiness=10
    grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=10' >> /etc/sysctl.conf
fi

log "Firewall"
# Default deny inbound. 80 is required — Let's Encrypt's HTTP-01 challenge lands
# there — and Postgres is deliberately absent: it is reachable only on the
# compose network, and an exposed 5432 on a public IP is found by scanners within
# hours.
ufw --force reset >/dev/null
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'ssh'
ufw allow 80/tcp comment 'http (acme challenge + redirect to https)'
ufw allow 443/tcp comment 'https'
ufw --force enable

log "Automatic security updates"
dpkg-reconfigure -f noninteractive unattended-upgrades

log "Deploy user"
if ! id "$DEPLOY_USER" >/dev/null 2>&1; then
    adduser --disabled-password --gecos "" "$DEPLOY_USER"
fi
usermod -aG docker "$DEPLOY_USER"
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$APP_DIR"
# Where the frontend repo's deploy job rsyncs the built site. Created up front
# and owned by the deploy user, because Caddy bind-mounts it read-only and a
# missing directory would make the container fail to start rather than serve an
# empty site.
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" "$APP_DIR/web"

log "Done"
cat <<EOF

  Next, as ${DEPLOY_USER}:

    git clone https://github.com/Aminpatra/Itqan.git ${APP_DIR}
    cd ${APP_DIR}
    cp .env.example .env && \$EDITOR .env     # secrets — see DEPLOY.md
    docker compose up -d

  Then seed the database and install the ingestion cron; both are in DEPLOY.md.

  Add this box's public key to the GitHub repo as a deploy secret so pushes
  deploy themselves:

    sudo -u ${DEPLOY_USER} ssh-keygen -t ed25519 -N '' -f /home/${DEPLOY_USER}/.ssh/id_ed25519

EOF
