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
