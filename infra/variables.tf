variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Short project prefix for IAM role and resource names"
  default     = "fashion-reco"
}

variable "environment" {
  type        = string
  description = "Environment label (dev, staging, prod)"
  default     = "dev"
}

variable "enable_iam_runtime" {
  type        = bool
  description = "Create v1 runtime IAM roles (ECS, Lambda, Glue, SageMaker, Step Functions, EventBridge)"
  default     = true
}

variable "s3_bucket" {
  type        = string
  description = "S3 bucket for MLflow artifacts and experiments"
}

variable "mlflow_tracking_server_name" {
  type    = string
  default = "fashion-reco-mlflow-dev"
}

variable "sagemaker_mlflow_role_arn" {
  type        = string
  description = "IAM role for SageMaker MLflow tracking server. Leave empty to use iam_runtime module output."
  default     = null
  nullable    = true
}

variable "enable_optuna_rds" {
  type    = bool
  default = false
}

variable "enable_mlflow" {
  type        = bool
  description = "Create SageMaker MLflow tracking server (requires S3 bucket to exist)"
  default     = false
}

variable "optuna_rds_identifier" {
  type    = string
  default = "fashion-reco-optuna"
}

variable "optuna_db_password" {
  type      = string
  sensitive = true
  default   = null
  nullable  = true
}

variable "vpc_id" {
  type        = string
  description = "VPC for Optuna RDS"
  default     = null
  nullable    = true
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnets for Optuna RDS"
  default     = []
}

variable "optuna_allowed_cidr_blocks" {
  type        = list(string)
  description = "CIDRs allowed to reach Optuna RDS"
}

variable "tags" {
  type    = map(string)
  default = {}
}
