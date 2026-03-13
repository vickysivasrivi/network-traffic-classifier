# =============================================================================
# ecr.tf — Elastic Container Registry repository for the API Docker image.
# =============================================================================

resource "aws_ecr_repository" "api" {
  name                 = var.ecr_repo_name
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Lifecycle policy: keep only the most recent N tagged images.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last ${var.ecr_image_count} images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = var.ecr_image_count
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}
