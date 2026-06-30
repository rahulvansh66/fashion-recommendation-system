locals {
  name_prefix = "${var.project_name}-${var.environment}"
  common_tags = merge(var.tags, {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
    Module      = "iam_runtime"
  })
  s3_bucket_arn = "arn:aws:s3:::${var.s3_bucket}"
  s3_objects_arn = "${local.s3_bucket_arn}/*"
}

# ---------------------------------------------------------------------------
# ECS Fargate — execution role (pull ECR images, write CloudWatch logs)
# Trusted entity: AWS service → Elastic Container Service → ECS Task
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ecs_task_execution" {
  name = "${local.name_prefix}-ecs-task-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution_managed" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# Optional: read secrets from SSM Parameter Store at task startup.
resource "aws_iam_role_policy" "ecs_task_execution_ssm" {
  name = "${local.name_prefix}-ecs-task-execution-ssm"
  role = aws_iam_role.ecs_task_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "ssm:GetParameters",
        "ssm:GetParameter",
      ]
      Resource = "arn:aws:ssm:${var.aws_region}:${var.account_id}:parameter/${local.name_prefix}/*"
    }]
  })
}

# ---------------------------------------------------------------------------
# ECS Fargate — task role (app runtime: SageMaker, Lambda, S3, Redis via VPC)
# Trusted entity: AWS service → Elastic Container Service → ECS Task
# ---------------------------------------------------------------------------
resource "aws_iam_role" "ecs_app_task" {
  name = "${local.name_prefix}-ecs-app-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "ecs_app_task" {
  name = "${local.name_prefix}-ecs-app-task"
  role = aws_iam_role.ecs_app_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeSageMakerEndpoints"
        Effect = "Allow"
        Action = [
          "sagemaker:InvokeEndpoint",
        ]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:endpoint/${local.name_prefix}-*"
      },
      {
        Sid    = "InvokeFaissLambda"
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction",
        ]
        Resource = "arn:aws:lambda:${var.aws_region}:${var.account_id}:function:${local.name_prefix}-faiss-*"
      },
      {
        Sid    = "ReadFeaturesFromS3"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Resource = [local.s3_bucket_arn, local.s3_objects_arn]
      },
      {
        Sid    = "CloudWatchMetrics"
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData",
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "Recommendation"
          }
        }
      },
      {
        Sid    = "XRayTracing"
        Effect = "Allow"
        Action = [
          "xray:PutTraceSegments",
          "xray:PutTelemetryRecords",
        ]
        Resource = "*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda — FAISS vector search
# Trusted entity: AWS service → Lambda
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_faiss" {
  name = "${local.name_prefix}-lambda-faiss"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_faiss_basic" {
  role       = aws_iam_role.lambda_faiss.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_faiss" {
  name = "${local.name_prefix}-lambda-faiss"
  role = aws_iam_role.lambda_faiss.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "ReadFaissIndexFromS3"
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:ListBucket",
      ]
      Resource = [local.s3_bucket_arn, local.s3_objects_arn]
    }]
  })
}

# ---------------------------------------------------------------------------
# Lambda — cache pre-warm producer (EventBridge → Redis read → SQS send)
# Trusted entity: AWS service → Lambda
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_prewarm_producer" {
  name = "${local.name_prefix}-lambda-prewarm-producer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_prewarm_producer_basic" {
  role       = aws_iam_role.lambda_prewarm_producer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_prewarm_producer" {
  name = "${local.name_prefix}-lambda-prewarm-producer"
  role = aws_iam_role.lambda_prewarm_producer.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "SendToPrewarmQueue"
      Effect = "Allow"
      Action = [
        "sqs:SendMessage",
        "sqs:SendMessageBatch",
        "sqs:GetQueueUrl",
      ]
      Resource = "arn:aws:sqs:${var.aws_region}:${var.account_id}:${local.name_prefix}-cache-prewarm-*"
    }]
  })
}

# ---------------------------------------------------------------------------
# Lambda — cache pre-warm consumer (SQS → full 5-stage pipeline → Redis write)
# Trusted entity: AWS service → Lambda
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_prewarm_consumer" {
  name = "${local.name_prefix}-lambda-prewarm-consumer"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_prewarm_consumer_basic" {
  role       = aws_iam_role.lambda_prewarm_consumer.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_prewarm_consumer" {
  name = "${local.name_prefix}-lambda-prewarm-consumer"
  role = aws_iam_role.lambda_prewarm_consumer.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ConsumePrewarmQueue"
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes",
          "sqs:ChangeMessageVisibility",
        ]
        Resource = "arn:aws:sqs:${var.aws_region}:${var.account_id}:${local.name_prefix}-cache-prewarm-*"
      },
      {
        Sid    = "InvokeSageMakerEndpoints"
        Effect = "Allow"
        Action = ["sagemaker:InvokeEndpoint"]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:endpoint/${local.name_prefix}-*"
      },
      {
        Sid    = "InvokeFaissLambda"
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = "arn:aws:lambda:${var.aws_region}:${var.account_id}:function:${local.name_prefix}-faiss-*"
      },
      {
        Sid    = "ReadFeaturesFromS3"
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [local.s3_bucket_arn, local.s3_objects_arn]
      },
      {
        Sid    = "PrewarmMetrics"
        Effect = "Allow"
        Action = ["cloudwatch:PutMetricData"]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "Recommendation"
          }
        }
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda — FAISS index build (ML pipeline step)
# Trusted entity: AWS service → Lambda
# ---------------------------------------------------------------------------
resource "aws_iam_role" "lambda_faiss_index_build" {
  name = "${local.name_prefix}-lambda-faiss-index-build"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "lambda_faiss_index_build_basic" {
  role       = aws_iam_role.lambda_faiss_index_build.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "lambda_faiss_index_build" {
  name = "${local.name_prefix}-lambda-faiss-index-build"
  role = aws_iam_role.lambda_faiss_index_build.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "ReadWriteEmbeddingsAndIndices"
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
      ]
      Resource = [local.s3_bucket_arn, local.s3_objects_arn]
    }]
  })
}

# ---------------------------------------------------------------------------
# AWS Glue — ETL jobs (raw → clean → features → Redis warm-up)
# Trusted entity: AWS service → Glue
# ---------------------------------------------------------------------------
resource "aws_iam_role" "glue_etl" {
  name = "${local.name_prefix}-glue-etl"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "glue.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy_attachment" "glue_service" {
  role       = aws_iam_role.glue_etl.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole"
}

resource "aws_iam_role_policy" "glue_etl_s3" {
  name = "${local.name_prefix}-glue-etl-s3"
  role = aws_iam_role.glue_etl.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid    = "DataLakeReadWrite"
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
      ]
      Resource = [local.s3_bucket_arn, local.s3_objects_arn]
    }]
  })
}

# ---------------------------------------------------------------------------
# SageMaker — training, processing, batch transform, endpoints, MLflow
# Trusted entity: AWS service → SageMaker
# ---------------------------------------------------------------------------
resource "aws_iam_role" "sagemaker_execution" {
  name = "${local.name_prefix}-sagemaker-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "sagemaker_execution" {
  name = "${local.name_prefix}-sagemaker-execution"
  role = aws_iam_role.sagemaker_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECRImagePull"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams",
        ]
        Resource = "arn:aws:logs:${var.aws_region}:${var.account_id}:log-group:/aws/sagemaker/*"
      },
      {
        Sid    = "DataLakeAndArtifacts"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket",
        ]
        Resource = [local.s3_bucket_arn, local.s3_objects_arn]
      },
      {
        Sid    = "ModelRegistry"
        Effect = "Allow"
        Action = [
          "sagemaker:CreateModel",
          "sagemaker:CreateModelPackage",
          "sagemaker:CreateModelPackageGroup",
          "sagemaker:DescribeModel",
          "sagemaker:DescribeModelPackage",
          "sagemaker:DescribeModelPackageGroup",
          "sagemaker:ListModelPackages",
          "sagemaker:UpdateModelPackage",
          "sagemaker:AddTags",
        ]
        Resource = [
          "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:model/${local.name_prefix}-*",
          "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:model-package/${local.name_prefix}-*",
          "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:model-package-group/${local.name_prefix}-*",
        ]
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Step Functions — orchestrates Glue + SageMaker Pipeline trigger
# Trusted entity: AWS service → Step Functions
# ---------------------------------------------------------------------------
resource "aws_iam_role" "step_functions" {
  name = "${local.name_prefix}-step-functions"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "states.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "step_functions" {
  name = "${local.name_prefix}-step-functions"
  role = aws_iam_role.step_functions.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "InvokeGlueJobs"
        Effect = "Allow"
        Action = [
          "glue:StartJobRun",
          "glue:GetJobRun",
          "glue:GetJobRuns",
          "glue:BatchStopJobRun",
        ]
        Resource = "arn:aws:glue:${var.aws_region}:${var.account_id}:job/${local.name_prefix}-*"
      },
      {
        Sid    = "InvokeLambda"
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = "arn:aws:lambda:${var.aws_region}:${var.account_id}:function:${local.name_prefix}-*"
      },
      {
        Sid    = "StartSageMakerPipeline"
        Effect = "Allow"
        Action = [
          "sagemaker:StartPipelineExecution",
          "sagemaker:DescribePipelineExecution",
          "sagemaker:ListPipelineExecutionSteps",
        ]
        Resource = "arn:aws:sagemaker:${var.aws_region}:${var.account_id}:pipeline/${local.name_prefix}-*"
      },
      {
        Sid    = "StepFunctionsLogging"
        Effect = "Allow"
        Action = [
          "logs:CreateLogDelivery",
          "logs:GetLogDelivery",
          "logs:UpdateLogDelivery",
          "logs:DeleteLogDelivery",
          "logs:ListLogDeliveries",
          "logs:PutResourcePolicy",
          "logs:DescribeResourcePolicies",
          "logs:DescribeLogGroups",
        ]
        Resource = "*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# EventBridge — cron triggers (Step Functions, Lambda pre-warm producer)
# Trusted entity: AWS service → EventBridge
# ---------------------------------------------------------------------------
resource "aws_iam_role" "eventbridge" {
  name = "${local.name_prefix}-eventbridge"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Principal = {
        Service = "events.amazonaws.com"
      }
    }]
  })

  tags = local.common_tags
}

resource "aws_iam_role_policy" "eventbridge" {
  name = "${local.name_prefix}-eventbridge"
  role = aws_iam_role.eventbridge.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "StartStepFunctions"
        Effect = "Allow"
        Action = ["states:StartExecution"]
        Resource = "arn:aws:states:${var.aws_region}:${var.account_id}:stateMachine:${local.name_prefix}-*"
      },
      {
        Sid    = "InvokePrewarmProducer"
        Effect = "Allow"
        Action = ["lambda:InvokeFunction"]
        Resource = "arn:aws:lambda:${var.aws_region}:${var.account_id}:function:${local.name_prefix}-lambda-prewarm-producer"
      },
    ]
  })
}
