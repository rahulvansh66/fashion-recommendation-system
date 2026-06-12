aws_region                      = "us-east-1"
s3_bucket                       = "fashion-reco-dev"
sagemaker_mlflow_role_arn       = "arn:aws:iam::ACCOUNT_ID:role/SageMakerMLflowRole"
optuna_db_password              = "CHANGE_ME"
vpc_id                          = "vpc-xxxxxxxx"
private_subnet_ids              = ["subnet-aaa", "subnet-bbb"]
optuna_allowed_cidr_blocks      = ["10.0.0.0/16"]
