# EC2 on the Hackathon Account

Account `127696279288`, region `us-west-2`. Use `--profile hack-scripps` for all CLI commands.

---

## Network resources (permanent, already provisioned)

| Resource | ID | Notes |
|---|---|---|
| VPC | `vpc-0c7b131a02a1b3b85` | `172.31.0.0/16` |
| PrivateSubnet1A | `subnet-0096ffc9c05bebab3` | `172.31.64.0/20` — us-west-2a |
| PrivateSubnet2A | `subnet-071d3dbca8e9b7209` | `172.31.32.0/20` — us-west-2b |
| PrivateSubnet3A | `subnet-0e8c4795d00d2f2c6` | `172.31.80.0/20` — us-west-2c |
| PublicSubnet1A | `subnet-0e1bdfbc00c6b98d6` | `172.31.0.0/20` — us-west-2a |
| Internet Gateway | `igw-04019d688ed9e7748` | `hackathon-igw` |
| NAT Gateway | `nat-0561360099820e566` | EIP `54.186.216.143` — leave running between sessions |
| S3 gateway endpoint | `vpce-0ebf641ca3311f239` | Free S3 traffic from all subnets |
| EICE endpoint | `eice-0ecadad192e23f430` | Provides SSH to private instances |
| EICE security group | `sg-09d5ef7889a26f56a` | `hackathon-eice-instance` — use at launch |
| IAM instance profile | `hackathon-ec2-profile` | Grants S3 + Bedrock access without credentials |

Private instances have outbound internet via NAT. Never delete the NAT gateway between sessions — the idle cost (~$0.045/hr) is less than recreating it.

---

## Launch an instance

```bash
INSTANCE_TYPE=t3.medium    # light work / dev
# INSTANCE_TYPE=t3.xlarge  # ML / large data

INSTANCE_ID=$(aws ec2 run-instances \
  --image-id ami-00563078bca04e287 \
  --instance-type $INSTANCE_TYPE \
  --subnet-id subnet-0096ffc9c05bebab3 \
  --security-group-ids sg-09d5ef7889a26f56a \
  --iam-instance-profile Name=hackathon-ec2-profile \
  --metadata-options HttpTokens=required \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=smansfield-spinhance}]' \
  --profile hack-scripps --region us-west-2 \
  --query 'Instances[0].InstanceId' --output text)
echo "Instance: $INSTANCE_ID"

aws ec2 wait instance-running --instance-ids $INSTANCE_ID \
  --profile hack-scripps --region us-west-2
```

Add extra storage for large datasets (append to `run-instances`):
```bash
--block-device-mappings \
  '[{"DeviceName":"/dev/xvdf","Ebs":{"VolumeSize":100,"VolumeType":"gp3","DeleteOnTermination":true}}]'
```

Then on the instance after SSH:
```bash
sudo mkfs.xfs /dev/xvdf && sudo mkdir /data
sudo mount /dev/xvdf /data && sudo chown ec2-user:ec2-user /data
```

---

## SSH via EICE (no public IP needed)

```bash
# Generate a fresh key (60-second push window)
ssh-keygen -t ed25519 -N "" -f /tmp/eice-key -q <<< y

aws ec2-instance-connect send-ssh-public-key \
  --instance-id $INSTANCE_ID \
  --instance-os-user ec2-user \
  --ssh-public-key file:///tmp/eice-key.pub \
  --profile hack-scripps --region us-west-2

ssh -i /tmp/eice-key \
  -o StrictHostKeyChecking=no \
  -o ProxyCommand='aws ec2-instance-connect open-tunnel \
    --instance-id '"$INSTANCE_ID"' \
    --profile hack-scripps --region us-west-2' \
  ec2-user@$INSTANCE_ID
```

Copy files to/from the instance:
```bash
# Upload
scp -i /tmp/eice-key \
  -o ProxyCommand='aws ec2-instance-connect open-tunnel --instance-id '"$INSTANCE_ID"' --profile hack-scripps --region us-west-2' \
  ./script.py ec2-user@$INSTANCE_ID:/home/ec2-user/

# Download
scp -i /tmp/eice-key \
  -o ProxyCommand='aws ec2-instance-connect open-tunnel --instance-id '"$INSTANCE_ID"' --profile hack-scripps --region us-west-2' \
  ec2-user@$INSTANCE_ID:/data/output.npy ./
```

---

## Calling Bedrock from inside an EC2 instance

The `hackathon-ec2-profile` IAM instance profile includes Bedrock access. No profile or credentials needed — boto3 picks them up automatically from the instance metadata service:

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-west-2")

resp = bedrock.invoke_model(
    modelId="us.anthropic.claude-haiku-4-5-20251001-v1:0",
    contentType="application/json",
    accept="application/json",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Your prompt here."}],
    }),
)
print(json.loads(resp["body"].read())["content"][0]["text"])
```

---

## Terminate when done

```bash
aws ec2 terminate-instances --instance-ids $INSTANCE_ID \
  --profile hack-scripps --region us-west-2
```
