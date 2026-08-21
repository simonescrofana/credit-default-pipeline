data "archive_file" "wake_lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/aws_lambda"
  output_path = "${path.module}/lambda_wake.zip"
  excludes    = ["sleep_handler.py", "__pycache__"]
}

resource "aws_lambda_function" "wake" {
  function_name    = "${var.project_name}-wake"
  filename         = data.archive_file.wake_lambda_zip.output_path
  source_code_hash = data.archive_file.wake_lambda_zip.output_base64sha256
  handler          = "handler.handler"
  runtime          = "python3.13"
  role             = aws_iam_role.lambda_wake_sleep.arn
  timeout          = 30

  environment {
    variables = {
      DYNAMODB_TABLE          = aws_dynamodb_table.wake_state.name
      DB_INSTANCE_IDENTIFIER  = aws_db_instance.main.identifier
      ECS_CLUSTER_NAME        = aws_ecs_cluster.main.name
      ECS_API_SERVICE_NAME    = aws_ecs_service.api.name
      ECS_UI_SERVICE_NAME     = aws_ecs_service.ui.name
      SNS_TOPIC_ARN           = aws_sns_topic.wake_sleep_notifications.arn
    }
  }
}
