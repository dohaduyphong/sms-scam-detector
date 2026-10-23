#!/usr/bin/env bash
# Cập nhật API trên máy chủ từ GitHub (không dùng Docker): git pull --ff-only,
# pip install (chỉ khi requirements.txt thay đổi), systemctl --user restart, rồi kiểm tra /health.
#
# Chạy trên máy chủ bằng chính tài khoản deploy sở hữu thư mục mã nguồn (không cần sudo):
#   deploy/update.sh
#
# Script không bao giờ reset, clean hay xoá file local: nếu file đang được theo dõi có thay đổi
# chưa commit, hoặc lần pull không phải fast-forward, script dừng lại và không đổi gì.
#
# Biến môi trường tùy chọn:
#   SERVICE_NAME    tên unit systemd (mặc định: sms-scam-detector)
#   SYSTEMD_SCOPE   user (mặc định): dùng systemctl --user bằng tài khoản deploy
#                   system: dùng unit cấp hệ thống (tùy chọn), khởi động lại bằng `sudo -n`
#   HEALTH_URL      mặc định: http://127.0.0.1:8000/health
#   HEALTH_TIMEOUT  số giây chờ /health sau khi khởi động lại (mặc định: 420; lần khởi động đầu
#                   sau khi đổi model phải tải ViSoBERT và PhoBERT, khoảng 930 MB, từ Hugging Face)
#   FORCE_PIP=1     chạy pip install kể cả khi requirements.txt không đổi
#   FORCE_RESTART=1 khởi động lại kể cả khi không có commit mới
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${SERVICE_NAME:-sms-scam-detector}"
SYSTEMD_SCOPE="${SYSTEMD_SCOPE:-user}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-420}"
# Dùng bản torch chỉ chạy CPU (bản torch mặc định trên PyPI cho Linux kèm CUDA, nặng vài GB).
TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"

log() { printf '[update] %s\n' "$*"; }
die() { printf '[update] ERROR: %s\n' "$*" >&2; exit 1; }

case "$SYSTEMD_SCOPE" in
    user)
        # Shell không đăng nhập (su, cron, một số cấu hình SSH) có thể thiếu biến này,
        # khi đó `systemctl --user` báo lỗi "Failed to connect to bus".
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

# Transformer (không có trong git): ưu tiên thư mục local trong models/, sau đó đến <NAME>_SOURCE trong unit,
# cuối cùng là repo HF được ghim trong scam_detector/config.py (tải về ở lần khởi động đầu).
for m in visobert:VISOBERT phobert_base_v2:PHOBERT; do
    dir="${m%%:*}" var="${m##*:}"
    if [ -d "models/$dir" ]; then
        log "$var: local models/$dir/"
    elif [ "$SYSTEMD_SCOPE" = user ] &&
        systemctl --user cat "$SERVICE_NAME" 2>/dev/null | grep -q "^Environment=${var}_SOURCE="; then
        log "$var: ${var}_SOURCE from the unit (pinned ${var}_REVISION in config.py if it is a HF repo)"
    else
        log "$var: Hugging Face repo pinned in scam_detector/config.py"
    fi
done

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

# Scope user không cần quyền root. Scope system dùng `sudo -n` nên không bao giờ hỏi mật khẩu,
# vì vậy chỉ chạy được khi có rule sudoers cho lệnh này (tùy chọn, xem DEPLOY.md).
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
