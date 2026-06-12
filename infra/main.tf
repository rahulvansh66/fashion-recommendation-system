terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

module "mlflow_tracking_server" {
  source = "./modules/mlflow_tracking_server"

  tracking_server_name = var.mlflow_tracking_server_name
  artifact_store_uri   = "s3://${var.s3_bucket}/mlflow/artifacts/"
  role_arn             = var.sagemaker_mlflow_role_arn
  tags                 = var.tags
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
