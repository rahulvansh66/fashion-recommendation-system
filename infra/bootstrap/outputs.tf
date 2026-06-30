output "github_oidc_provider_arn" {
  value       = aws_iam_openid_connect_provider.github.arn
  description = "GitHub OIDC provider ARN — reference in GitHub Actions workflow"
}

output "github_actions_role_arn" {
  value       = aws_iam_role.github_actions_deploy.arn
  description = "Set as AWS_ROLE_ARN in GitHub Actions secrets/vars"
}

output "github_actions_role_name" {
  value = aws_iam_role.github_actions_deploy.name
}

output "next_steps" {
  value = <<-EOT
    Bootstrap complete. Next steps:
    1. In GitHub repo Settings → Secrets and variables → Actions, add:
         AWS_ROLE_ARN = ${aws_iam_role.github_actions_deploy.arn}
         AWS_REGION   = ${var.aws_region}
    2. cd ../ && terraform init && terraform apply  (creates runtime service roles)
    3. Update infra/environments/dev/terraform.tfvars with role ARNs from outputs
  EOT
}
