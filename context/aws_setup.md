# AWS Setup — Scripps Hackathon Account

Account `127696279288`, region `us-west-2`, accessed via AWS IAM Identity Center (SSO).

---

## 1. Configure the SSO profile

Add the following two stanzas to `~/.aws/config` (create the file if it doesn't exist):

```ini
[sso-session scripps-hackathon]
sso_start_url = https://d-9267e96a16.awsapps.com/start
sso_region = us-west-2
sso_registration_scopes = sso:account:access

[profile hack-scripps]
sso_session = scripps-hackathon
sso_account_id = 127696279288
sso_role_name = Hackathon
region = us-west-2
output = json
```

Or run `setup_aws_login.sh` (in this directory) — it writes these stanzas automatically if they're missing, then logs you in.

---

## 2. Log in (run this every 8–12 hours when the token expires)

```bash
aws sso login --profile hack-scripps
```

A browser window opens. Sign in with your Scripps credentials (`smansfield@scripps.edu`) and approve the device authorization. The token is cached under `~/.aws/sso/cache/`.

If the browser doesn't open automatically, the CLI will print a URL and a short code — open the URL manually and enter the code.

---

## 3. Verify

```bash
aws sts get-caller-identity --profile hack-scripps
```

Expected output:
```json
{
    "UserId": "...",
    "Account": "127696279288",
    "Arn": "arn:aws:sts::127696279288:assumed-role/AWSReservedSSO_.../smansfield@scripps.edu"
}
```

Quick S3 check:
```bash
aws s3 ls --profile hack-scripps --region us-west-2
```

---

## 4. Shared hackathon data

Project datasets are in `s3://scrippsresearch-hackathon/`:

```bash
# Browse top-level prefixes
aws s3 ls s3://scrippsresearch-hackathon/ --profile hack-scripps

# Stream a README without downloading
aws s3 cp s3://scrippsresearch-hackathon/allen/PQBP1_Hexamer_IM_README.md - \
  --profile hack-scripps
```

| Prefix | Dataset |
|--------|---------|
| `allen/` | PQBP1 hexamer cryo-EM + docking |
| `heberling/` | Siglec6 structural work |

---

## 5. Your personal S3 bucket

Create once, use for outputs and shared data:

```bash
# Name format: scrippsresearch-<yourname>-hackathon
aws s3 mb s3://scrippsresearch-smansfield-hackathon \
  --region us-west-2 --profile hack-scripps
```

Upload / download:
```bash
aws s3 cp ./output.npy s3://scrippsresearch-smansfield-hackathon/output.npy \
  --profile hack-scripps --region us-west-2

aws s3 sync ./results/ s3://scrippsresearch-smansfield-hackathon/results/ \
  --profile hack-scripps --region us-west-2
```

Share your bucket with other participants in the same account (read-only):
```bash
cat > /tmp/bucket-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"AWS": "arn:aws:iam::127696279288:root"},
    "Action": ["s3:GetObject", "s3:ListBucket"],
    "Resource": [
      "arn:aws:s3:::scrippsresearch-smansfield-hackathon",
      "arn:aws:s3:::scrippsresearch-smansfield-hackathon/*"
    ]
  }]
}
EOF

aws s3api put-bucket-policy \
  --bucket scrippsresearch-smansfield-hackathon \
  --policy file:///tmp/bucket-policy.json \
  --profile hack-scripps --region us-west-2
```

Add `"s3:PutObject"` to the Action list if teammates also need to write.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `The SSO session associated with this profile has expired` | `aws sso login --profile hack-scripps` |
| `profile 'hack-scripps' not found` | Add the stanzas from step 1 to `~/.aws/config` |
| `AccessDenied` on a specific action | Role `Hackathon` may not include that permission — check with the hackathon admin |
| `aws: command not found` | Open a new terminal; if still missing, install the AWS CLI v2 |
