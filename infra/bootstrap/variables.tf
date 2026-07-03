variable "aws_region" {
  type        = string
  description = "AWS region for IAM resources"
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Short project prefix used in role and policy names"
  default     = "fashion-reco"
}

variable "environment" {
  type        = string
  description = "Environment label (dev, staging, prod)"
  default     = "dev"
}

variable "github_org" {
  type        = string
  description = "GitHub organization or username that owns the repository"
}

variable "github_repo" {
  type        = string
  description = "GitHub repository name (without org prefix)"
  default     = "fashion-recommendation-system"
}

variable "github_branch" {
  type        = string
  description = "Git branch allowed to assume the CI/CD role (use main for v1)"
  default     = "main"
}

variable "terraform_state_bucket" {
  type        = string
  description = "S3 bucket for Terraform remote state (created manually or in a prior step)"
  default     = ""
}

variable "terraform_state_lock_table" {
  type        = string
  description = "DynamoDB table for Terraform state locking (optional)"
  default     = ""
}

variable "tags" {
  type        = map(string)
  description = "Tags applied to IAM resources"
  default     = {}
}
