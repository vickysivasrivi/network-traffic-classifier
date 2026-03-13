# =============================================================================
# variables.tf — All configurable inputs for the Network Threat IaC stack.
# =============================================================================

variable "aws_region" {
  description = "AWS region where all resources will be created."
  type        = string
  default     = "eu-west-1"
}

variable "project_name" {
  description = "Short name used as a prefix / tag for every resource."
  type        = string
  default     = "network-threat"
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository that stores the API Docker image."
  type        = string
  default     = "network-threat-classifier"
}

variable "ecs_cluster_name" {
  description = "Name of the ECS cluster."
  type        = string
  default     = "network-threat-cluster"
}

variable "task_cpu" {
  description = "CPU units for the ECS Fargate task (1 vCPU = 1024 units)."
  type        = string
  default     = "512"
}

variable "task_memory" {
  description = "Memory (MiB) allocated to the ECS Fargate task."
  type        = string
  default     = "1024"
}

variable "container_port" {
  description = "Port the FastAPI container listens on."
  type        = number
  default     = 8000
}

variable "log_retention_days" {
  description = "Number of days to retain CloudWatch log events."
  type        = number
  default     = 30
}

variable "ecr_image_count" {
  description = "Maximum number of tagged images to keep in ECR (older ones are purged)."
  type        = number
  default     = 10
}
