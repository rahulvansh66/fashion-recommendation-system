variable "identifier" {
  type        = string
  default     = "fashion-reco-optuna"
  description = "RDS instance identifier"
}

variable "db_name" {
  type        = string
  default     = "optuna"
  description = "Database name for Optuna studies"
}

variable "username" {
  type        = string
  default     = "optuna_admin"
  description = "Master username"
}

variable "password" {
  type        = string
  sensitive   = true
  description = "Master password"
}

variable "vpc_id" {
  type        = string
  description = "VPC for RDS"
}

variable "subnet_ids" {
  type        = list(string)
  description = "Private subnet IDs for RDS"
}

variable "allowed_cidr_blocks" {
  type        = list(string)
  description = "CIDR blocks allowed to connect to Optuna RDS"
}

variable "tags" {
  type        = map(string)
  default     = {}
  description = "Resource tags"
}

resource "aws_security_group" "optuna_rds" {
  name        = "${var.identifier}-sg"
  description = "Optuna PostgreSQL access"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidr_blocks
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = var.tags
}

resource "aws_db_subnet_group" "optuna" {
  name       = "${var.identifier}-subnet-group"
  subnet_ids = var.subnet_ids
  tags       = var.tags
}

resource "aws_db_instance" "optuna" {
  identifier             = var.identifier
  engine                 = "postgres"
  engine_version         = "15"
  instance_class         = "db.t4g.micro"
  allocated_storage      = 20
  db_name                = var.db_name
  username               = var.username
  password               = var.password
  db_subnet_group_name   = aws_db_subnet_group.optuna.name
  vpc_security_group_ids = [aws_security_group.optuna_rds.id]
  skip_final_snapshot    = true
  publicly_accessible    = false

  tags = var.tags
}

output "optuna_storage_uri" {
  value       = "postgresql+psycopg2://${var.username}:${var.password}@${aws_db_instance.optuna.address}:5432/${var.db_name}"
  sensitive   = true
  description = "Optuna storage URI for OPTUNA_STORAGE_URI"
}

output "endpoint" {
  value = aws_db_instance.optuna.address
}
