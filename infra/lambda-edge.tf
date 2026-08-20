data "archive_file" "edge_wake_lambda_zip" {
  type        = "zip"
  output_path = "${path.module}/lambda_edge_wake.zip"

  source {
    content = templatefile("${path.module}/aws_lambda/edge_wake_handler.py.tftpl", {
      dynamodb_table         = aws_dynamodb_table.wake_state.name
      db_instance_identifier = aws_db_instance.main.identifier
      ecs_cluster_name       = aws_ecs_cluster.main.name
      ecs_api_service_name   = aws_ecs_service.api.name
      ecs_ui_service_name    = aws_ecs_service.ui.name
      sns_topic_arn          = aws_sns_topic.wake_sleep_notifications.arn
    })
    filename = "wake_handler.py"
  }
}

resource "aws_iam_role" "lambda_edge" {
  provider = aws.us_east_1
  name     = "${var.project_name}-lambda-edge-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = ["lambda.amazonaws.com", "edgelambda.amazonaws.com"]
        }
      }
    ]
  })
}

# Lambda@Edge always runs against resources in us-east-1 regardless of
# where it is invoked from, so the same permissions as the regular wake
# Lambda are needed here, attached to this dedicated edge role instead.
resource "aws_iam_role_policy_attachment" "lambda_edge" {
  role       = aws_iam_role.lambda_edge.name
  policy_arn = aws_iam_policy.lambda_wake_sleep.arn
}

resource "aws_lambda_function" "edge_wake" {
  provider         = aws.us_east_1
  function_name    = "${var.project_name}-edge-wake"
  filename         = data.archive_file.edge_wake_lambda_zip.output_path
  source_code_hash = data.archive_file.edge_wake_lambda_zip.output_base64sha256
  handler          = "wake_handler.handler"
  runtime          = "python3.13"
  role             = aws_iam_role.lambda_edge.arn
  timeout          = 5
  publish          = true # Required: Lambda@Edge can only target a numbered version, never $LATEST.
}
