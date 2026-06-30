# Bootstrap IAM for v1 — Phase 1 (GitHub OIDC) + Phase 2 (runtime roles)
# Idempotent: skips resources that already exist.
# Uses temp JSON files (reliable on Windows PowerShell).

param(
    [string]$ProjectName = "fashion-reco",
    [string]$Environment = "dev",
    [string]$Region = "us-east-1",
    [string]$GitHubOrg = "rahulvansh66",
    [string]$GitHubRepo = "fashion-recommendation-system",
    [string]$GitHubBranch = "main",
    [string]$S3Bucket = "fashion-reco-dev"
)

$ErrorActionPreference = "Stop"
$Prefix = "$ProjectName-$Environment"
$AccountId = (aws sts get-caller-identity --query Account --output text).Trim()
$TempDir = Join-Path $env:TEMP "fashion-reco-iam-bootstrap"
New-Item -ItemType Directory -Force -Path $TempDir | Out-Null

function Write-PolicyFile {
    param([string]$Name, [string]$Json)
    $path = Join-Path $TempDir "$Name.json"
    [System.IO.File]::WriteAllText($path, $Json)
    return ($path -replace "\\", "/")
}

function Test-IamRole {
    param([string]$Name)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    aws iam get-role --role-name $Name *> $null
    $exists = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = $prev
    return $exists
}

function Ensure-OidcProvider {
    $providers = aws iam list-open-id-connect-providers --output json | ConvertFrom-Json
    $githubArn = "arn:aws:iam::${AccountId}:oidc-provider/token.actions.githubusercontent.com"
    if ($providers.OpenIDConnectProviderList.Arn -contains $githubArn) {
        Write-Host "OK  GitHub OIDC provider already exists"
        return $githubArn
    }
    aws iam create-open-id-connect-provider `
        --url "https://token.actions.githubusercontent.com" `
        --client-id-list "sts.amazonaws.com" `
        --thumbprint-list "6938fd4d98bab03fa75378ce8443c16eb053b7b9" | Out-Null
    Write-Host "CREATED GitHub OIDC provider"
    return $githubArn
}

function Ensure-Role {
    param(
        [string]$Name,
        [string]$TrustPolicyJson,
        [string]$InlinePolicyName = $null,
        [string]$InlinePolicyJson = $null,
        [string[]]$ManagedPolicies = @()
    )
    $trustPath = Write-PolicyFile "trust-$Name" $TrustPolicyJson
    if (-not (Test-IamRole $Name)) {
        aws iam create-role --role-name $Name --assume-role-policy-document "file://$trustPath" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to create role $Name" }
        Write-Host "CREATED role $Name"
    } else {
        Write-Host "OK  role $Name exists"
    }
    foreach ($mp in $ManagedPolicies) {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "SilentlyContinue"
        aws iam attach-role-policy --role-name $Name --policy-arn $mp *> $null
        $ErrorActionPreference = $prev
    }
    if ($InlinePolicyName -and $InlinePolicyJson) {
        $policyPath = Write-PolicyFile "inline-$Name" $InlinePolicyJson
        aws iam put-role-policy --role-name $Name --policy-name $InlinePolicyName --policy-document "file://$policyPath" | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Failed to attach inline policy to $Name" }
    }
    return "arn:aws:iam::${AccountId}:role/${Name}"
}

Write-Host "Account: $AccountId  Prefix: $Prefix  Region: $Region"
Write-Host "--- Phase 1: GitHub OIDC CI/CD role ---"

$oidcArn = Ensure-OidcProvider
$githubSub = "repo:${GitHubOrg}/${GitHubRepo}:ref:refs/heads/${GitHubBranch}"

$githubTrust = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Principal": { "Federated": "$oidcArn" },
      "Condition": {
        "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
        "StringLike": { "token.actions.githubusercontent.com:sub": "$githubSub" }
      }
    }
  ]
}
"@

$githubPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECRAuthAndPush",
      "Effect": "Allow",
      "Action": ["ecr:GetAuthorizationToken"],
      "Resource": "*"
    },
    {
      "Sid": "ECRRepository",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories",
        "ecr:CreateRepository"
      ],
      "Resource": "arn:aws:ecr:${Region}:${AccountId}:repository/${Prefix}-*"
    },
    {
      "Sid": "PassRuntimeRoles",
      "Effect": "Allow",
      "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::${AccountId}:role/${Prefix}-*",
      "Condition": {
        "StringEquals": {
          "iam:PassedToService": [
            "ecs-tasks.amazonaws.com",
            "lambda.amazonaws.com",
            "sagemaker.amazonaws.com",
            "glue.amazonaws.com"
          ]
        }
      }
    },
    {
      "Sid": "TerraformApply",
      "Effect": "Allow",
      "Action": [
        "iam:*", "s3:*", "ec2:*", "elasticache:*", "apigateway:*", "sqs:*", "events:*",
        "states:*", "logs:*", "cloudwatch:*", "sagemaker:*", "glue:*", "lambda:*", "ecs:*",
        "servicediscovery:*", "ecr:*", "cloudfront:*", "acm:*",
        "ssm:GetParameter", "ssm:GetParameters", "ssm:PutParameter", "ssm:DeleteParameter",
        "kms:Decrypt", "kms:Encrypt", "kms:GenerateDataKey", "kms:DescribeKey", "kms:CreateGrant"
      ],
      "Resource": "*"
    }
  ]
}
"@

$githubRoleArn = Ensure-Role `
    -Name "${Prefix}-github-actions-deploy" `
    -TrustPolicyJson $githubTrust `
    -InlinePolicyName "${Prefix}-github-actions-deploy" `
    -InlinePolicyJson $githubPolicy

Write-Host "--- Phase 2: Runtime service roles ---"

$ecsTrust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Principal":{"Service":"ecs-tasks.amazonaws.com"}}]}'
$lambdaTrust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Principal":{"Service":"lambda.amazonaws.com"}}]}'
$glueTrust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Principal":{"Service":"glue.amazonaws.com"}}]}'
$sagemakerTrust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Principal":{"Service":"sagemaker.amazonaws.com"}}]}'
$statesTrust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Principal":{"Service":"states.amazonaws.com"}}]}'
$eventsTrust = '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Action":"sts:AssumeRole","Principal":{"Service":"events.amazonaws.com"}}]}'

$s3BucketArn = "arn:aws:s3:::$S3Bucket"
$s3ObjectsArn = "${s3BucketArn}/*"

$ecsExecArn = Ensure-Role -Name "${Prefix}-ecs-task-execution" -TrustPolicyJson $ecsTrust -ManagedPolicies @(
    "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
)

$ecsAppPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "InvokeSageMaker",
      "Effect": "Allow",
      "Action": ["sagemaker:InvokeEndpoint"],
      "Resource": "arn:aws:sagemaker:${Region}:${AccountId}:endpoint/${Prefix}-*"
    },
    {
      "Sid": "InvokeFaiss",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:${Region}:${AccountId}:function:${Prefix}-faiss-*"
    },
    {
      "Sid": "ReadS3",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["$s3BucketArn", "$s3ObjectsArn"]
    },
    {
      "Sid": "Metrics",
      "Effect": "Allow",
      "Action": ["cloudwatch:PutMetricData"],
      "Resource": "*",
      "Condition": { "StringEquals": { "cloudwatch:namespace": "Recommendation" } }
    }
  ]
}
"@
$ecsAppArn = Ensure-Role -Name "${Prefix}-ecs-app-task" -TrustPolicyJson $ecsTrust -InlinePolicyName "${Prefix}-ecs-app-task" -InlinePolicyJson $ecsAppPolicy

$faissPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadIndex",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["$s3BucketArn", "$s3ObjectsArn"]
    }
  ]
}
"@
$faissArn = Ensure-Role -Name "${Prefix}-lambda-faiss" -TrustPolicyJson $lambdaTrust `
    -ManagedPolicies @("arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole") `
    -InlinePolicyName "${Prefix}-lambda-faiss" -InlinePolicyJson $faissPolicy

$producerPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SQS",
      "Effect": "Allow",
      "Action": ["sqs:SendMessage", "sqs:SendMessageBatch", "sqs:GetQueueUrl"],
      "Resource": "arn:aws:sqs:${Region}:${AccountId}:${Prefix}-cache-prewarm-*"
    }
  ]
}
"@
$producerArn = Ensure-Role -Name "${Prefix}-lambda-prewarm-producer" -TrustPolicyJson $lambdaTrust `
    -ManagedPolicies @("arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole") `
    -InlinePolicyName "${Prefix}-lambda-prewarm-producer" -InlinePolicyJson $producerPolicy

$consumerPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SQS",
      "Effect": "Allow",
      "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes", "sqs:ChangeMessageVisibility"],
      "Resource": "arn:aws:sqs:${Region}:${AccountId}:${Prefix}-cache-prewarm-*"
    },
    {
      "Sid": "SageMaker",
      "Effect": "Allow",
      "Action": ["sagemaker:InvokeEndpoint"],
      "Resource": "arn:aws:sagemaker:${Region}:${AccountId}:endpoint/${Prefix}-*"
    },
    {
      "Sid": "Faiss",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:${Region}:${AccountId}:function:${Prefix}-faiss-*"
    },
    {
      "Sid": "S3",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": ["$s3BucketArn", "$s3ObjectsArn"]
    }
  ]
}
"@
$consumerArn = Ensure-Role -Name "${Prefix}-lambda-prewarm-consumer" -TrustPolicyJson $lambdaTrust `
    -ManagedPolicies @("arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole") `
    -InlinePolicyName "${Prefix}-lambda-prewarm-consumer" -InlinePolicyJson $consumerPolicy

$indexPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "S3",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": ["$s3BucketArn", "$s3ObjectsArn"]
    }
  ]
}
"@
$indexArn = Ensure-Role -Name "${Prefix}-lambda-faiss-index-build" -TrustPolicyJson $lambdaTrust `
    -ManagedPolicies @("arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole") `
    -InlinePolicyName "${Prefix}-lambda-faiss-index-build" -InlinePolicyJson $indexPolicy

$glueArn = Ensure-Role -Name "${Prefix}-glue-etl" -TrustPolicyJson $glueTrust `
    -ManagedPolicies @("arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole") `
    -InlinePolicyName "${Prefix}-glue-etl-s3" -InlinePolicyJson $faissPolicy

$smPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ECR",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    },
    {
      "Sid": "Logs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:${Region}:${AccountId}:log-group:/aws/sagemaker/*"
    },
    {
      "Sid": "S3",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": ["$s3BucketArn", "$s3ObjectsArn"]
    },
    {
      "Sid": "Registry",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateModel",
        "sagemaker:CreateModelPackage",
        "sagemaker:CreateModelPackageGroup",
        "sagemaker:DescribeModel",
        "sagemaker:DescribeModelPackage",
        "sagemaker:DescribeModelPackageGroup",
        "sagemaker:ListModelPackages",
        "sagemaker:UpdateModelPackage",
        "sagemaker:AddTags"
      ],
      "Resource": [
        "arn:aws:sagemaker:${Region}:${AccountId}:model/${Prefix}-*",
        "arn:aws:sagemaker:${Region}:${AccountId}:model-package/${Prefix}-*",
        "arn:aws:sagemaker:${Region}:${AccountId}:model-package-group/${Prefix}-*"
      ]
    }
  ]
}
"@
$smArn = Ensure-Role -Name "${Prefix}-sagemaker-execution" -TrustPolicyJson $sagemakerTrust `
    -InlinePolicyName "${Prefix}-sagemaker-execution" -InlinePolicyJson $smPolicy

$sfPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "Glue",
      "Effect": "Allow",
      "Action": ["glue:StartJobRun", "glue:GetJobRun", "glue:GetJobRuns", "glue:BatchStopJobRun"],
      "Resource": "arn:aws:glue:${Region}:${AccountId}:job/${Prefix}-*"
    },
    {
      "Sid": "Lambda",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:${Region}:${AccountId}:function:${Prefix}-*"
    },
    {
      "Sid": "SMPipeline",
      "Effect": "Allow",
      "Action": [
        "sagemaker:StartPipelineExecution",
        "sagemaker:DescribePipelineExecution",
        "sagemaker:ListPipelineExecutionSteps"
      ],
      "Resource": "arn:aws:sagemaker:${Region}:${AccountId}:pipeline/${Prefix}-*"
    }
  ]
}
"@
$sfArn = Ensure-Role -Name "${Prefix}-step-functions" -TrustPolicyJson $statesTrust `
    -InlinePolicyName "${Prefix}-step-functions" -InlinePolicyJson $sfPolicy

$ebPolicy = @"
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "SFN",
      "Effect": "Allow",
      "Action": ["states:StartExecution"],
      "Resource": "arn:aws:states:${Region}:${AccountId}:stateMachine:${Prefix}-*"
    },
    {
      "Sid": "Lambda",
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:${Region}:${AccountId}:function:${Prefix}-lambda-prewarm-producer"
    }
  ]
}
"@
$ebArn = Ensure-Role -Name "${Prefix}-eventbridge" -TrustPolicyJson $eventsTrust `
    -InlinePolicyName "${Prefix}-eventbridge" -InlinePolicyJson $ebPolicy

Write-Host ""
Write-Host "=== Bootstrap complete ==="
Write-Host "GitHub Actions role (AWS_ROLE_ARN): $githubRoleArn"
Write-Host "SageMaker execution role:           $smArn"

$outputs = [ordered]@{
    github_actions_role_arn           = $githubRoleArn
    sagemaker_execution_role_arn      = $smArn
    ecs_task_execution_role_arn       = $ecsExecArn
    ecs_app_task_role_arn             = $ecsAppArn
    lambda_faiss_role_arn             = $faissArn
    lambda_prewarm_producer_role_arn  = $producerArn
    lambda_prewarm_consumer_role_arn  = $consumerArn
    lambda_faiss_index_build_role_arn = $indexArn
    glue_etl_role_arn                 = $glueArn
    step_functions_role_arn           = $sfArn
    eventbridge_role_arn              = $ebArn
}
$outPath = Join-Path (Split-Path $PSScriptRoot -Parent) "bootstrap\iam-outputs.json"
($outputs | ConvertTo-Json) | Set-Content -Path $outPath -Encoding utf8
Write-Host "Role ARNs saved to $outPath"
