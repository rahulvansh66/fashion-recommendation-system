output "iam_runtime_role_arns" {
  value       = var.enable_iam_runtime ? module.iam_runtime[0].role_arns : null
  description = "Runtime IAM role ARNs for ECS, Lambda, Glue, SageMaker, Step Functions, EventBridge"
}

output "sagemaker_execution_role_arn" {
  value       = var.enable_iam_runtime ? module.iam_runtime[0].sagemaker_execution_role_arn : var.sagemaker_mlflow_role_arn
  description = "Set as SAGEMAKER_ROLE_ARN in .env.local and training scripts"
}

output "ecs_task_execution_role_arn" {
  value = var.enable_iam_runtime ? module.iam_runtime[0].ecs_task_execution_role_arn : null
}

output "ecs_app_task_role_arn" {
  value = var.enable_iam_runtime ? module.iam_runtime[0].ecs_app_task_role_arn : null
}

output "mlflow_tracking_server_arn" {
  value       = var.enable_mlflow ? module.mlflow_tracking_server[0].tracking_server_arn : null
  description = "Set as MLFLOW_TRACKING_URI in .env.local"
}

output "mlflow_tracking_server_name" {
  value = var.enable_mlflow ? module.mlflow_tracking_server[0].tracking_server_name : null
}

output "optuna_storage_uri" {
  value       = var.enable_optuna_rds ? module.optuna_rds[0].optuna_storage_uri : null
  sensitive   = true
  description = "Set as OPTUNA_STORAGE_URI in .env.local"
}

output "optuna_rds_endpoint" {
  value = var.enable_optuna_rds ? module.optuna_rds[0].endpoint : null
}
