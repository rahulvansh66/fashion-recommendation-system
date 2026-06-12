output "mlflow_tracking_server_arn" {
  value       = module.mlflow_tracking_server.tracking_server_arn
  description = "Set as MLFLOW_TRACKING_URI in .env.local"
}

output "mlflow_tracking_server_name" {
  value = module.mlflow_tracking_server.tracking_server_name
}

output "optuna_storage_uri" {
  value       = var.enable_optuna_rds ? module.optuna_rds[0].optuna_storage_uri : null
  sensitive   = true
  description = "Set as OPTUNA_STORAGE_URI in .env.local"
}

output "optuna_rds_endpoint" {
  value = var.enable_optuna_rds ? module.optuna_rds[0].endpoint : null
}
