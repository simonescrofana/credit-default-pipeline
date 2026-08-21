resource "aws_sns_topic" "wake_sleep_notifications" {
  name = "${var.project_name}-wake-sleep-notifications"
}

resource "aws_sns_topic_subscription" "wake_sleep_email" {
  topic_arn = aws_sns_topic.wake_sleep_notifications.arn
  protocol  = "email"
  endpoint  = var.notification_email
}
