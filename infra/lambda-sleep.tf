data "archive_file" "sleep_lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/aws_lambda"
  output_path = "${path.module}/lambda_sleep.zip"
  excludes    = ["wake_handler.py", "__pycache__"]
}

resource "aws_lambda_function" "sleep" {
  function_name    = "${var.project_name}-sleep"
  filename         = data.archive_file.sleep_lambda_zip.output_path
  source_code_hash = data.archive_file.sleep_lambda_zip.output_base64sha256
  handler          = "sleep_handler.handler"
  runtime          = "python3.13"
  role             = aws_iam_role.lambda_wake_sleep.arn
  timeout          = 30

  environment {
    variables = {
      DYNAMODB_TABLE               = aws_dynamodb_table.wake_state.name
      DB_INSTANCE_IDENTIFIER       = aws_db_instance.main.identifier
      ECS_CLUSTER_NAME             = aws_ecs_cluster.main.name
      ECS_API_SERVICE_NAME         = aws_ecs_service.api.name
      ECS_UI_SERVICE_NAME          = aws_ecs_service.ui.name
      SNS_TOPIC_ARN                = aws_sns_topic.wake_sleep_notifications.arn
      INACTIVITY_THRESHOLD_SECONDS = "1800"
    }
  }
}

resource "aws_cloudwatch_event_rule" "sleep_schedule" {
  name                = "${var.project_name}-sleep-check"
  schedule_expression = "rate(15 minutes)"
}

resource "aws_cloudwatch_event_target" "sleep_lambda_target" {
  rule = aws_cloudwatch_event_rule.sleep_schedule.name
  arn  = aws_lambda_function.sleep.arn
}

resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.sleep.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.sleep_schedule.arn
}
