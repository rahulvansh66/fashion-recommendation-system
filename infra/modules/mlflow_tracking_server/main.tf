variable "tracking_server_name" {
  type        = string
  description = "SageMaker MLflow tracking server name"
}

variable "artifact_store_uri" {
  type        = string
  description = "S3 URI for MLflow artifacts"
}

variable "role_arn" {
  type        = string
  description = "IAM role ARN for the MLflow tracking server"
}

variable "mlflow_version" {
  type        = string
  default     = "3.0.0"
  description = "MLflow version for the tracking server"
}

variable "tracking_server_size" {
  type        = string
  default     = "Small"
  description = "Tracking server size (Small or Medium)"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Resource tags"
}

resource "aws_sagemaker_mlflow_tracking_server" "this" {
  tracking_server_name = var.tracking_server_name
  artifact_store_uri   = var.artifact_store_uri
  role_arn             = var.role_arn
  mlflow_version       = var.mlflow_version
  tracking_server_size = var.tracking_server_size

  tags = var.tags
}

output "tracking_server_arn" {
  value       = aws_sagemaker_mlflow_tracking_server.this.tracking_server_arn
  description = "MLflow tracking server ARN for MLFLOW_TRACKING_URI"
}

output "tracking_server_name" {
  value = aws_sagemaker_mlflow_tracking_server.this.tracking_server_name
}
