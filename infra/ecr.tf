data "aws_ecr_repository" "api" {
  name = "${var.project_name}-api"
}

data "aws_ecr_repository" "ui" {
  name = "${var.project_name}-ui"
}
