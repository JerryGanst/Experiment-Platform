# SSH Config Guide (macOS client -> Ubuntu server)

This is a generic template. Replace HOST, USER, and PORT with your own values.

Prerequisites:
- You must know the server login user and SSH port.
- If unsure, ask the admin. Guessing can lock you out.

## 1) Generate a dedicated key

```bash
ssh-keygen -t ed25519 -a 100 -f ~/.ssh/id_ed25519_ai
```

## 2) Install the public key on the server

Preferred:

```bash
ssh-copy-id -i ~/.ssh/id_ed25519_ai.pub -p PORT USER@HOST
```

Fallback (no ssh-copy-id):

```bash
cat ~/.ssh/id_ed25519_ai.pub | ssh -p PORT USER@HOST \
  'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys'
```

## 3) Add a host alias (optional but recommended)

Create or edit `~/.ssh/config`:

```sshconfig
Host myserver
  HostName HOST
  User USER
  Port PORT
  IdentityFile ~/.ssh/id_ed25519_ai
  IdentitiesOnly yes
```

## 4) Test

```bash
ssh myserver
# or:
ssh -i ~/.ssh/id_ed25519_ai -p PORT USER@HOST
```

## 5) Troubleshooting

- Permissions: `~/.ssh` is 700, `~/.ssh/authorized_keys` is 600
- Server-side: check `/etc/ssh/sshd_config` for `PubkeyAuthentication yes`
- Debug: `ssh -vvv -i ~/.ssh/id_ed25519_ai -p PORT USER@HOST`

## 6) Security notes

- Use a dedicated key for automation/AI.
- Keep the private key local; never commit it.
- Optional: restrict `authorized_keys` with `from="IP"` or `command="..."`.

## Template script

See `scripts/setup_ssh.sh` (edit HOST/REMOTE_USER/PORT or pass flags).
