variable "project_name" {
  description = "Prefix used to name every resource created for this project"
  type        = string
  default     = "credit-default-project"
}

variable "aws_region" {
  description = "AWS region hosting all the infrastructure for this project"
  type        = string
  default     = "us-east-1"
}

variable "db_snapshot_identifier" {
  description = "Identifier of a DB snapshot to restore the database from. Leave empty to create a fresh, empty database instead."
  type        = string
  default     = ""
}
