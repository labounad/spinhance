#!/usr/bin/env bash
# setup_aws_login.sh — configure and refresh the hack-scripps SSO token
#
# Usage:
#   ./context/setup_aws_login.sh          # configure (if needed) + login
#   ./context/setup_aws_login.sh --verify # also verify identity after login

set -euo pipefail

PROFILE="hack-scripps"
SESSION="scripps-hackathon"
ACCOUNT_ID="127696279288"
SSO_START_URL="https://d-9267e96a16.awsapps.com/start"
SSO_REGION="us-west-2"
REGION="us-west-2"
ROLE="Hackathon"
CONFIG="$HOME/.aws/config"

VERIFY=false
[[ "${1:-}" == "--verify" ]] && VERIFY=true

# ── helpers ──────────────────────────────────────────────────────────────────
ok()  { echo "  [ok]  $*"; }
info(){ echo "  [--]  $*"; }
err() { echo "  [!!]  $*" >&2; }

# ── 1. check aws CLI ──────────────────────────────────────────────────────────
echo
echo "==> Step 1  Check aws CLI"
if ! command -v aws &>/dev/null; then
    err "aws CLI not found. Install it:"
    err "  Linux:   https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    err "  macOS:   brew install awscli"
    exit 1
fi
ok "$(aws --version)"

# ── 2. write ~/.aws/config entries if missing ─────────────────────────────────
echo
echo "==> Step 2  Configure ~/.aws/config"
mkdir -p "$HOME/.aws"
touch "$CONFIG"

write_sso_session=false
write_profile=false

if ! grep -q "\[sso-session ${SESSION}\]" "$CONFIG" 2>/dev/null; then
    write_sso_session=true
fi
if ! grep -q "\[profile ${PROFILE}\]" "$CONFIG" 2>/dev/null; then
    write_profile=true
fi

if $write_sso_session || $write_profile; then
    # Ensure file doesn't end mid-line before we append
    [[ -s "$CONFIG" ]] && echo >> "$CONFIG"

    if $write_sso_session; then
        cat >> "$CONFIG" <<EOF

[sso-session ${SESSION}]
sso_start_url = ${SSO_START_URL}
sso_region = ${SSO_REGION}
sso_registration_scopes = sso:account:access
EOF
        ok "Wrote [sso-session ${SESSION}] to $CONFIG"
    fi

    if $write_profile; then
        cat >> "$CONFIG" <<EOF

[profile ${PROFILE}]
sso_session = ${SESSION}
sso_account_id = ${ACCOUNT_ID}
sso_role_name = ${ROLE}
region = ${REGION}
output = json
EOF
        ok "Wrote [profile ${PROFILE}] to $CONFIG"
    fi
else
    ok "Profile '${PROFILE}' already present in $CONFIG"
fi

# ── 3. sso login ─────────────────────────────────────────────────────────────
echo
echo "==> Step 3  SSO login (a browser window will open)"
info "If the browser doesn't open, copy the URL + code printed below."
echo
aws sso login --profile "$PROFILE"

# ── 4. optional verify ───────────────────────────────────────────────────────
if $VERIFY; then
    echo
    echo "==> Step 4  Verify identity"
    IDENTITY=$(aws sts get-caller-identity --profile "$PROFILE" --output json)
    GOT_ACCOUNT=$(echo "$IDENTITY" | python3 -c "import sys,json; print(json.load(sys.stdin)['Account'])")
    GOT_ARN=$(echo     "$IDENTITY" | python3 -c "import sys,json; print(json.load(sys.stdin)['Arn'])")

    if [[ "$GOT_ACCOUNT" != "$ACCOUNT_ID" ]]; then
        err "Wrong account: $GOT_ACCOUNT (expected $ACCOUNT_ID)"
        exit 1
    fi
    ok "Account : $GOT_ACCOUNT"
    ok "Identity: $GOT_ARN"

    echo
    echo "==> Step 5  Quick S3 listing"
    aws s3 ls --profile "$PROFILE" --region "$REGION" 2>/dev/null \
        && ok "S3 listing succeeded" \
        || info "No buckets yet (that's fine)"
fi

echo
echo "Done. Token cached under ~/.aws/sso/cache/"
echo "Re-run this script whenever you see: 'Token has expired and refresh failed'"
