#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/setup_ssh.sh -h HOST -u REMOTE_USER [-p PORT] [-k KEY_PATH] [-a ALIAS]

Examples:
  bash scripts/setup_ssh.sh -h 203.0.113.10 -u ubuntu -p 22 -a myserver
  HOST=203.0.113.10 REMOTE_USER=ubuntu PORT=22 bash scripts/setup_ssh.sh

Notes:
  - Run this on the client (macOS).
  - It creates the key if missing and installs the public key on the server.
  - If ALIAS is set, it appends a host block to ~/.ssh/config.
EOF
}

HOST="${HOST:-}"
REMOTE_USER="${REMOTE_USER:-}"
PORT="${PORT:-22}"
KEY_PATH="${KEY_PATH:-$HOME/.ssh/id_ed25519_ai}"
ALIAS="${ALIAS:-}"

while getopts ":h:u:p:k:a:" opt; do
  case "$opt" in
    h) HOST="$OPTARG" ;;
    u) REMOTE_USER="$OPTARG" ;;
    p) PORT="$OPTARG" ;;
    k) KEY_PATH="$OPTARG" ;;
    a) ALIAS="$OPTARG" ;;
    *) usage; exit 2 ;;
  esac
done

if [ -z "$HOST" ] || [ -z "$REMOTE_USER" ]; then
  usage
  exit 1
fi

mkdir -p "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

if [ ! -f "$KEY_PATH" ]; then
  ssh-keygen -t ed25519 -a 100 -f "$KEY_PATH" -N ""
fi

if [ ! -f "${KEY_PATH}.pub" ]; then
  ssh-keygen -y -f "$KEY_PATH" > "${KEY_PATH}.pub"
fi

if command -v ssh-copy-id >/dev/null 2>&1; then
  ssh-copy-id -i "${KEY_PATH}.pub" -p "$PORT" "${REMOTE_USER}@${HOST}"
else
  cat "${KEY_PATH}.pub" | ssh -p "$PORT" "${REMOTE_USER}@${HOST}" \
    'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys'
fi

if [ -n "$ALIAS" ]; then
  SSH_CONFIG="$HOME/.ssh/config"
  touch "$SSH_CONFIG"
  chmod 600 "$SSH_CONFIG"
  if ! grep -qE "^[[:space:]]*Host[[:space:]]+${ALIAS}([[:space:]]|$)" "$SSH_CONFIG"; then
    cat >> "$SSH_CONFIG" <<EOF
Host ${ALIAS}
  HostName ${HOST}
  User ${REMOTE_USER}
  Port ${PORT}
  IdentityFile ${KEY_PATH}
  IdentitiesOnly yes
EOF
  fi
fi

echo "Done. Test with: ssh -i \"$KEY_PATH\" -p \"$PORT\" \"$REMOTE_USER@$HOST\""
if [ -n "$ALIAS" ]; then
  echo "Or: ssh \"$ALIAS\""
fi
