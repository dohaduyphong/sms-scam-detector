#!/usr/bin/env bash
# Update the production API from GitHub (no Docker):
#   git pull --ff-only -> pip install (only if requirements.txt changed)
#   -> systemctl --user restart -> health check
#
# Run on the server as the deploy user that owns the checkout (no sudo):
#   deploy/update.sh
#
# Never resets, cleans or deletes local files: if tracked files have local
# changes or the pull is not a fast-forward, it stops and changes nothing.
#
# Optional environment variables:
#   SERVICE_NAME    systemd unit (default: sms-scam-detector)
#   SYSTEMD_SCOPE   user (default): systemctl --user, as the deploy user
#                   system: optional system-level unit, restarted via `sudo -n`
#   HEALTH_URL      default: http://127.0.0.1:8000/health
#   HEALTH_TIMEOUT  seconds to wait for /health after restart (default: 180)
#   FORCE_PIP=1     run pip install even if requirements.txt did not change
#   FORCE_RESTART=1 restart even if there were no new commits
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${SERVICE_NAME:-sms-scam-detector}"
SYSTEMD_SCOPE="${SYSTEMD_SCOPE:-user}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-180}"
# CPU-only torch wheels (the default PyPI torch on Linux bundles CUDA, several GB).
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

log() { printf '[update] %s\n' "$*"; }
die() { printf '[update] ERROR: %s\n' "$*" >&2; exit 1; }

case "$SYSTEMD_SCOPE" in
    user)
        # Non-login shells (su, cron, some SSH setups) may lack this, and
        # `systemctl --user` then fails with "Failed to connect to bus".
        export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
        SYSTEMCTL=(systemctl --user)
        JOURNAL_HINT="journalctl --user -u $SERVICE_NAME -n 100 --no-pager"
        ;;
    system)
        SYSTEMCTL=(sudo -n systemctl)
        JOURNAL_HINT="journalctl -u $SERVICE_NAME -n 100 --no-pager"
        ;;
    *) die "SYSTEMD_SCOPE must be 'user' or 'system' (got '$SYSTEMD_SCOPE')." ;;
esac

cd "$APP_DIR"

[ -x .venv/bin/pip ] || die "$APP_DIR/.venv not found -- create it first (see DEPLOY.md)."
command -v curl >/dev/null || die "curl is required for the health check."
if [ "$SYSTEMD_SCOPE" = user ] && ! systemctl --user cat "$SERVICE_NAME" >/dev/null 2>&1; then
    die "User unit '$SERVICE_NAME' not found (XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR). See DEPLOY.md section 4."
fi

if [ "$SYSTEMD_SCOPE" = user ] && [ ! -d models/visobert_full ] &&
    ! systemctl --user cat "$SERVICE_NAME" 2>/dev/null | grep -q '^Environment=VISOBERT_SOURCE='; then
    log "WARNING: neither models/visobert_full/ nor VISOBERT_SOURCE in the unit -- ViSoBERT"
    log "         will fail to load. Copy it first, e.g. from your machine:"
    log "         rsync -avz models/visobert_full/ <user>@<host>:$APP_DIR/models/visobert_full/"
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
    git status --short
    die "Tracked files have local changes. Commit or stash them, then re-run."
fi

old_rev="$(git rev-parse HEAD)"
log "Pulling (fast-forward only)..."
git pull --ff-only
new_rev="$(git rev-parse HEAD)"

if [ "$old_rev" = "$new_rev" ]; then
    log "Already up to date ($new_rev)."
    if [ "${FORCE_PIP:-0}" != 1 ] && [ "${FORCE_RESTART:-0}" != 1 ]; then
        exit 0
    fi
else
    log "Updated ${old_rev:0:7} -> ${new_rev:0:7}:"
    git --no-pager log --oneline "$old_rev..$new_rev"
fi

if [ "${FORCE_PIP:-0}" = 1 ] || ! git diff --quiet "$old_rev" "$new_rev" -- requirements.txt; then
    log "Installing requirements..."
    .venv/bin/pip install --extra-index-url "$TORCH_INDEX_URL" -r requirements.txt
else
    log "requirements.txt unchanged -- skipping pip install."
fi

# user scope: no root needed. system scope: `sudo -n` never prompts, so it
# only works with a sudoers rule for this command (optional, see DEPLOY.md).
log "Restarting $SERVICE_NAME (${SYSTEMCTL[*]})..."
"${SYSTEMCTL[@]}" restart "$SERVICE_NAME" ||
    die "Restart failed. Run manually: ${SYSTEMCTL[*]} restart $SERVICE_NAME, then: $JOURNAL_HINT"

log "Waiting for $HEALTH_URL (up to ${HEALTH_TIMEOUT}s, models take a while to load)..."
deadline=$((SECONDS + HEALTH_TIMEOUT))
until body="$(curl -fsS --max-time 5 "$HEALTH_URL" 2>/dev/null)"; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        die "Health check failed. Inspect: $JOURNAL_HINT"
    fi
    sleep 3
done
log "Healthy: $body"
