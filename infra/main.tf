terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

# Runtime IAM roles — trusted entity: AWS service (ECS, Lambda, Glue, SageMaker, etc.)
# Apply after bootstrap stack (GitHub OIDC) on a fresh account.
module "iam_runtime" {
  source = "./modules/iam_runtime"
  count  = var.enable_iam_runtime ? 1 : 0

  project_name = var.project_name
  environment  = var.environment
  aws_region   = var.aws_region
  account_id   = data.aws_caller_identity.current.account_id
  s3_bucket    = var.s3_bucket
  tags         = var.tags
}

module "mlflow_tracking_server" {
  source = "./modules/mlflow_tracking_server"
  count  = var.enable_mlflow ? 1 : 0

  tracking_server_name = var.mlflow_tracking_server_name
  artifact_store_uri   = "s3://${var.s3_bucket}/mlflow/artifacts/"
  role_arn = coalesce(
    var.sagemaker_mlflow_role_arn,
    try(module.iam_runtime[0].sagemaker_execution_role_arn, null)
  )
  tags = var.tags
}

module "optuna_rds" {
  source = "./modules/optuna_rds"
  count  = var.enable_optuna_rds ? 1 : 0

  identifier          = var.optuna_rds_identifier
  password            = var.optuna_db_password
  vpc_id              = var.vpc_id
  subnet_ids          = var.private_subnet_ids
  allowed_cidr_blocks = var.optuna_allowed_cidr_blocks
  tags                = var.tags
}
