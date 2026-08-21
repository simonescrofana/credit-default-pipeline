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

variable "dvc_user" {
  description = "Username for basic auth against the DVC remote (DagsHub), used only by the one-off data migration task."
  type        = string
  sensitive   = true
}

variable "dvc_password" {
  description = "Password for basic auth against the DVC remote (DagsHub), used only by the one-off data migration task."
  type        = string
  sensitive   = true
}

variable "groq_api_key" {
  description = "API key for the Groq-hosted LLM used by the agent, read at runtime by the api container."
  type        = string
  sensitive   = true
}

variable "logfire_token" {
  description = "Write token for Logfire observability, read at runtime by the api container."
  type        = string
  sensitive   = true
}

variable "notification_email" {
  description = "Email address to receive wake/sleep notifications for the project."
  type        = string
}
