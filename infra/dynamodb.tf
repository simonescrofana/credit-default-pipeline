resource "aws_dynamodb_table" "wake_state" {
  name         = "${var.project_name}-wake-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "id"

  attribute {
    name = "id"
    type = "S"
  }
}

# Seeds the table with an initial "awake" state, matching the fact that
# RDS/ECS come up already running right after apply. The system is put
# to sleep manually the first time (see the runbook), after which the
# wake/sleep Lambdas take over automatically.
resource "aws_dynamodb_table_item" "initial_state" {
  table_name = aws_dynamodb_table.wake_state.name
  hash_key   = aws_dynamodb_table.wake_state.hash_key

  item = jsonencode({
    id               = { S = "state" }
    status           = { S = "awake" }
    last_request_at  = { N = "0" }
  })

  lifecycle {
    ignore_changes = [item] # Only used to seed the initial value; the Lambdas own it after that.
  }
}
