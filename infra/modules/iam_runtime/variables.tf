variable "project_name" {
  type        = string
  description = "Short project prefix used in role names"
}

variable "environment" {
  type        = string
  description = "Environment label (dev, staging, prod)"
}

variable "aws_region" {
  type        = string
  description = "AWS region"
}

variable "account_id" {
  type        = string
  description = "AWS account ID"
}

variable "s3_bucket" {
  type        = string
  description = "S3 data lake bucket name"
}

variable "tags" {
  type        = map(string)
  default     = {}
}
