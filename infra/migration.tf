data "aws_ecr_repository" "migration" {
  name = "${var.project_name}-migration"
}

resource "aws_secretsmanager_secret" "dvc_credentials" {
  name = "${var.project_name}-dvc-credentials"
}

resource "aws_secretsmanager_secret_version" "dvc_credentials" {
  secret_id = aws_secretsmanager_secret.dvc_credentials.id

  secret_string = jsonencode({
    user     = var.dvc_user
    password = var.dvc_password
  })
}

resource "aws_iam_policy" "ecs_execution_dvc_secret" {
  name = "${var.project_name}-ecs-execution-dvc-secret-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = "secretsmanager:GetSecretValue"
        Resource = aws_secretsmanager_secret.dvc_credentials.arn
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution_dvc_secret" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = aws_iam_policy.ecs_execution_dvc_secret.arn
}

resource "aws_cloudwatch_log_group" "migration" {
  name              = "/ecs/${var.project_name}-migration"
  retention_in_days = 7
}

resource "aws_ecs_task_definition" "migration" {
  family                   = "${var.project_name}-migration"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = "1024"
  memory                   = "2048"
  execution_role_arn       = aws_iam_role.ecs_execution.arn

  container_definitions = jsonencode([
    {
      name      = "migration"
      image     = "${data.aws_ecr_repository.migration.repository_url}:latest"
      essential = true

      environment = [
          { name = "POSTGRES_HOST", value = aws_db_instance.main.address },
          { name = "POSTGRES_PORT", value = tostring(aws_db_instance.main.port) },
          { name = "POSTGRES_DB", value = aws_db_instance.main.db_name },
          { name = "POSTGRES_USER", value = aws_db_instance.main.username },
          # Required by Settings() (config.py validates every field, not just
          # the ones this container actually uses). GROQ_API_KEY is never read
          # by anything this task runs — alembic/dvc/dbt never touch Groq.
          # LOGFIRE_TOKEN uses the real secret below instead: restore.py spans
          # each chunk, so a valid token gives per-chunk visibility on Logfire.
          { name = "GROQ_API_KEY", value = "unused-by-migration-task" }
        ]

      secrets = [
        {
          name      = "POSTGRES_PASSWORD"
          valueFrom = "${aws_db_instance.main.master_user_secret[0].secret_arn}:password::"
        },
        {
          name      = "LOGFIRE_TOKEN"
          valueFrom = aws_secretsmanager_secret.logfire_token.arn
        },
        {
          name      = "DVC_USER"
          valueFrom = "${aws_secretsmanager_secret.dvc_credentials.arn}:user::"
        },
        {
          name      = "DVC_PASSWORD"
          valueFrom = "${aws_secretsmanager_secret.dvc_credentials.arn}:password::"
        }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.migration.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "migration"
        }
      }
    }
  ])
}
