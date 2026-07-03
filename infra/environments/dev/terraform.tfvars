aws_region         = "us-east-1"
project_name       = "fashion-reco"
environment        = "dev"
enable_iam_runtime = true
enable_mlflow      = false
enable_optuna_rds  = false
s3_bucket          = "fashion-reco-dev"

tags = {
  Owner = "rahul.vansh"
}
