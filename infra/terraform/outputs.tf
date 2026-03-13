# =============================================================================
# outputs.tf — Values surfaced after `terraform apply`.
# =============================================================================

output "ecr_repository_url" {
  description = "Full URL of the ECR repository (use as the Docker push target)."
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_arn" {
  description = "ARN of the ECS cluster."
  value       = aws_ecs_cluster.main.arn
}

output "cloudwatch_log_group_name" {
  description = "Name of the CloudWatch log group that receives container logs."
  value       = aws_cloudwatch_log_group.api.name
}
