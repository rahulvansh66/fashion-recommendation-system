output "ecs_task_execution_role_arn" {
  value       = aws_iam_role.ecs_task_execution.arn
  description = "ECS task execution role — ECR pull + CloudWatch logs"
}

output "ecs_app_task_role_arn" {
  value       = aws_iam_role.ecs_app_task.arn
  description = "ECS task role — SageMaker, Lambda, S3 at runtime"
}

output "lambda_faiss_role_arn" {
  value = aws_iam_role.lambda_faiss.arn
}

output "lambda_prewarm_producer_role_arn" {
  value = aws_iam_role.lambda_prewarm_producer.arn
}

output "lambda_prewarm_consumer_role_arn" {
  value = aws_iam_role.lambda_prewarm_consumer.arn
}

output "lambda_faiss_index_build_role_arn" {
  value = aws_iam_role.lambda_faiss_index_build.arn
}

output "glue_etl_role_arn" {
  value = aws_iam_role.glue_etl.arn
}

output "sagemaker_execution_role_arn" {
  value       = aws_iam_role.sagemaker_execution.arn
  description = "Use as SAGEMAKER_ROLE_ARN for training and MLflow"
}

output "step_functions_role_arn" {
  value = aws_iam_role.step_functions.arn
}

output "eventbridge_role_arn" {
  value = aws_iam_role.eventbridge.arn
}

output "role_arns" {
  value = {
    ecs_task_execution        = aws_iam_role.ecs_task_execution.arn
    ecs_app_task              = aws_iam_role.ecs_app_task.arn
    lambda_faiss              = aws_iam_role.lambda_faiss.arn
    lambda_prewarm_producer   = aws_iam_role.lambda_prewarm_producer.arn
    lambda_prewarm_consumer   = aws_iam_role.lambda_prewarm_consumer.arn
    lambda_faiss_index_build  = aws_iam_role.lambda_faiss_index_build.arn
    glue_etl                  = aws_iam_role.glue_etl.arn
    sagemaker_execution       = aws_iam_role.sagemaker_execution.arn
    step_functions            = aws_iam_role.step_functions.arn
    eventbridge               = aws_iam_role.eventbridge.arn
  }
  description = "All runtime role ARNs — reference in service configs"
}
