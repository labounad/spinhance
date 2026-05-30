import json
import boto3

PROFILE = "hack-scripps"
REGION = "us-west-2"

# Available models on this account (us-west-2, account 127696279288)
# Verified working — use the us.* inference profile IDs, not bare model IDs.
MODELS = {
    "haiku":       "us.anthropic.claude-haiku-4-5-20251001-v1:0",    # fastest / cheapest
    "sonnet-4-5":  "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "sonnet":      "us.anthropic.claude-sonnet-4-6",                  # latest sonnet
    "opus-4-1":    "us.anthropic.claude-opus-4-1-20250805-v1:0",
    "opus-4-5":    "us.anthropic.claude-opus-4-5-20251101-v1:0",
    "opus":        "us.anthropic.claude-opus-4-6-v1",                 # latest opus
}
MODEL = MODELS["opus"]

PROMPT = """\
List exactly 10 SMILES strings for small organic molecules that each have
exactly 8 magnetically distinct ¹H spin groups (i.e. 8 chemically non-equivalent
sets of protons after accounting for symmetry, homotopicity, and enantiotopicity).

Requirements for each molecule:
- Contains only C, H, N, O, S, F, Cl, Br (no metals, no radicals)
- Molecular weight between 100 and 400 Da
- Fewer than 50 heavy atoms
- Exactly 8 magnetically distinct proton environments
- Valid, canonical SMILES

Return ONLY a numbered list of 10 SMILES strings, one per line, no extra commentary.
"""

session = boto3.Session(profile_name=PROFILE, region_name=REGION)
bedrock = session.client("bedrock-runtime")

try:
    resp = bedrock.invoke_model(
        modelId=MODEL,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [{"role": "user", "content": PROMPT}],
        }),
    )
    text = json.loads(resp["body"].read())["content"][0]["text"].strip()
    print(text)
except Exception as e:
    print(f"Error: {e}")
