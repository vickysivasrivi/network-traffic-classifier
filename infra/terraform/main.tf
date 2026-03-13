# =============================================================================
# main.tf — AWS provider configuration and optional remote state backend.
#
# Usage:
#   terraform init        # initialise providers
#   terraform plan        # preview changes
#   terraform apply       # create / update resources
#   terraform destroy     # tear down all resources
#
# Remote state (optional — uncomment and fill in bucket/key to enable):
# =============================================================================

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }

  # Uncomment to store state in S3 (recommended for team usage):
  # backend "s3" {
  #   bucket         = "my-terraform-state-bucket"
  #   key            = "network-threat/terraform.tfstate"
  #   region         = "eu-west-1"
  #   dynamodb_table = "terraform-state-lock"
  #   encrypt        = true
  # }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = var.project_name
      Environment = "production"
      ManagedBy   = "terraform"
    }
  }
}
