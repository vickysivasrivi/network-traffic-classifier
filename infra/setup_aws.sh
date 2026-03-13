#!/bin/bash
# =============================================================================
# setup_aws.sh — One-time AWS infrastructure provisioning for Week 4 MLOps.
#
# What this script creates:
#   1. ECR repository (private Docker registry, vulnerability scanning on push)
#   2. CloudWatch log group with 30-day retention
#   3. ECS Fargate cluster
#   4. ECS task definition (registers the infra/ecs_task_definition.json template)
#
# Prerequisites:
#   - AWS CLI v2 installed:  aws --version
#   - Credentials configured: aws configure  (or IAM role / env vars)
#   - IAM permissions: ecr:*, logs:*, ecs:*, iam:PassRole
#
# Usage (from project root):
#   chmod +x infra/setup_aws.sh
#   ./infra/setup_aws.sh
# =============================================================================
set -e

AWS_REGION="eu-west-1"
ECR_REPO_NAME="network-threat-classifier"
LOG_GROUP="/ecs/network-threat-api"
ECS_CLUSTER_NAME="network-threat-cluster"
TASK_FAMILY="network-threat-api"

echo "======================================================"
echo " Week 4 MLOps — AWS Infrastructure Setup"
echo " Region: $AWS_REGION"
echo "======================================================"
echo ""

# ------------------------------------------------------------------------------
# Step 0: Confirm AWS CLI is working and get the account ID.
# ------------------------------------------------------------------------------
echo "Verifying AWS credentials ..."
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
CALLER_ARN=$(aws sts get-caller-identity --query Arn --output text)
echo "  Account ID : $ACCOUNT_ID"
echo "  Caller ARN : $CALLER_ARN"
echo ""

# ------------------------------------------------------------------------------
# Step 1: Create ECR repository.
#
# ECR is a private Docker registry within your AWS account.
# --image-scanning-configuration scanOnPush=true automatically checks every
# pushed image against the Common Vulnerabilities and Exposures (CVE) database.
# || true prevents the script from stopping if the repo already exists.
# ------------------------------------------------------------------------------
echo "[1/4] Creating ECR repository: $ECR_REPO_NAME ..."
aws ecr create-repository \
    --repository-name "$ECR_REPO_NAME" \
    --region "$AWS_REGION" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE \
    --output table 2>/dev/null || echo "  Repository already exists — skipping."

ECR_URI="$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_NAME"
echo "  ECR URI: $ECR_URI"
echo ""

# ------------------------------------------------------------------------------
# Step 2: Create CloudWatch log group with 30-day retention.
#
# The awslogs log driver in the ECS task definition writes container stdout/stderr
# to this log group. Without a retention policy, logs accumulate indefinitely.
# 30 days is appropriate for a portfolio project.
# ------------------------------------------------------------------------------
echo "[2/4] Creating CloudWatch log group: $LOG_GROUP ..."
aws logs create-log-group \
    --log-group-name "$LOG_GROUP" \
    --region "$AWS_REGION" 2>/dev/null || echo "  Log group already exists — skipping creation."

aws logs put-retention-policy \
    --log-group-name "$LOG_GROUP" \
    --retention-in-days 30 \
    --region "$AWS_REGION"
echo "  Log group ready with 30-day retention."
echo ""

# ------------------------------------------------------------------------------
# Step 3: Create ECS Fargate cluster.
#
# An ECS cluster is a logical grouping of tasks and services.
# FARGATE capacity provider = AWS manages the underlying EC2 instances.
# FARGATE_SPOT = uses spare AWS capacity at up to 70% discount (may be interrupted).
# The default strategy (base=1) ensures at least one task always runs on regular
# FARGATE, with additional tasks eligible for SPOT pricing.
# ------------------------------------------------------------------------------
echo "[3/4] Creating ECS cluster: $ECS_CLUSTER_NAME ..."
aws ecs create-cluster \
    --cluster-name "$ECS_CLUSTER_NAME" \
    --capacity-providers FARGATE FARGATE_SPOT \
    --default-capacity-provider-strategy \
        capacityProvider=FARGATE,weight=1,base=1 \
    --region "$AWS_REGION" \
    --output table 2>/dev/null || echo "  Cluster already exists — skipping."
echo ""

# ------------------------------------------------------------------------------
# Step 4: Register the ECS task definition.
#
# The task definition template uses <YOUR_ACCOUNT_ID> as a placeholder.
# sed replaces it with the real account ID before sending to AWS.
# Each registration creates a new revision (network-threat-api:1, :2, etc.).
# The CD pipeline always registers a new revision before updating the service.
# ------------------------------------------------------------------------------
echo "[4/4] Registering ECS task definition: $TASK_FAMILY ..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_DEF_FILE="$SCRIPT_DIR/ecs_task_definition.json"

if [ ! -f "$TASK_DEF_FILE" ]; then
    echo "ERROR: Task definition not found at: $TASK_DEF_FILE"
    echo "       Run this script from the project root: ./infra/setup_aws.sh"
    exit 1
fi

RENDERED=$(sed "s/<YOUR_ACCOUNT_ID>/$ACCOUNT_ID/g" "$TASK_DEF_FILE")
aws ecs register-task-definition \
    --cli-input-json "$RENDERED" \
    --region "$AWS_REGION" \
    --output table
echo ""

# ------------------------------------------------------------------------------
# Summary + next steps
# ------------------------------------------------------------------------------
echo "======================================================"
echo " All resources created successfully!"
echo "======================================================"
echo ""
echo "  ECR URI : $ECR_URI"
echo "  Cluster : $ECS_CLUSTER_NAME"
echo "  Log grp : $LOG_GROUP"
echo ""
echo "======================================================="
echo " NEXT STEPS"
echo "======================================================="
echo ""
echo "--- A. Verify ecsTaskExecutionRole exists ---"
echo ""
echo "aws iam get-role --role-name ecsTaskExecutionRole"
echo ""
echo "If the command above fails with NoSuchEntity, create the role:"
echo ""
echo "aws iam create-role --role-name ecsTaskExecutionRole \\"
echo "  --assume-role-policy-document '{"
echo "    \"Version\":\"2012-10-17\","
echo "    \"Statement\":[{\"Effect\":\"Allow\","
echo "    \"Principal\":{\"Service\":\"ecs-tasks.amazonaws.com\"},"
echo "    \"Action\":\"sts:AssumeRole\"}]}'"
echo ""
echo "aws iam attach-role-policy --role-name ecsTaskExecutionRole \\"
echo "  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
echo ""
echo "--- B. Find your VPC subnet and security group IDs ---"
echo ""
echo "aws ec2 describe-subnets --region $AWS_REGION \\"
echo "  --query 'Subnets[*].[SubnetId,VpcId,CidrBlock,AvailabilityZone]' --output table"
echo ""
echo "aws ec2 describe-security-groups --region $AWS_REGION \\"
echo "  --query 'SecurityGroups[*].[GroupId,GroupName,Description]' --output table"
echo ""
echo "--- C. Create the ECS service (replace the placeholders below) ---"
echo ""
echo "aws ecs create-service \\"
echo "  --cluster $ECS_CLUSTER_NAME \\"
echo "  --service-name network-threat-service \\"
echo "  --task-definition $TASK_FAMILY \\"
echo "  --desired-count 1 \\"
echo "  --launch-type FARGATE \\"
echo "  --network-configuration 'awsvpcConfiguration={subnets=[subnet-xxxx],securityGroups=[sg-xxxx],assignPublicIp=ENABLED}' \\"
echo "  --region $AWS_REGION"
echo ""
echo "  Note: The security group must allow inbound TCP on port 8000."
echo "  Create one if needed:"
echo "  aws ec2 create-security-group --group-name network-threat-sg \\"
echo "    --description 'Allow inbound 8000 for threat API' --region $AWS_REGION"
echo "  aws ec2 authorize-security-group-ingress --group-name network-threat-sg \\"
echo "    --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region $AWS_REGION"
echo ""
echo "--- D. Add GitHub Actions Secrets ---"
echo ""
echo "  Go to: https://github.com/<your-username>/network-traffic-classifier/settings/secrets/actions"
echo "  Add two repository secrets:"
echo "    AWS_ACCESS_KEY_ID      (from your IAM user's access key)"
echo "    AWS_SECRET_ACCESS_KEY  (from your IAM user's access key)"
echo ""
echo "  The IAM user needs these minimum permissions:"
echo "    ecr:GetAuthorizationToken, ecr:BatchCheckLayerAvailability,"
echo "    ecr:GetDownloadUrlForLayer, ecr:BatchGetImage,"
echo "    ecr:PutImage, ecr:InitiateLayerUpload, ecr:UploadLayerPart, ecr:CompleteLayerUpload,"
echo "    ecs:RegisterTaskDefinition, ecs:DescribeServices, ecs:UpdateService,"
echo "    iam:PassRole"
echo ""
echo "--- E. Trigger the CD pipeline ---"
echo ""
echo "  git push origin main"
echo "  Then watch: https://github.com/<your-username>/network-traffic-classifier/actions"
echo ""
