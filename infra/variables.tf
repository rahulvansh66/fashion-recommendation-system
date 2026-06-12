variable "aws_region" {
  type    = string
  default = "us-east-1"
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
  description = "IAM role for SageMaker MLflow tracking server"
}

variable "enable_optuna_rds" {
  type    = bool
  default = true
}

variable "optuna_rds_identifier" {
  type    = string
  default = "fashion-reco-optuna"
}

variable "optuna_db_password" {
  type      = string
  sensitive = true
}

variable "vpc_id" {
  type        = string
  description = "VPC for Optuna RDS"
}

variable "private_subnet_ids" {
  type        = list(string)
  description = "Private subnets for Optuna RDS"
}

variable "optuna_allowed_cidr_blocks" {
  type        = list(string)
  description = "CIDRs allowed to reach Optuna RDS"
}

variable "tags" {
  type    = map(string)
  default = {}
}
