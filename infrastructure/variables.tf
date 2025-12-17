variable "salt" {
  description = "Optional salt for use in the name"
  default     = ""
  type        = string
}

variable "location" {
  description = "Default location to use if not specified"
  default     = "switzerlandnorth"
  type        = string
}

variable "app_name" {
  description = "Name of the application"
  default     = "mcpchat"
  type        = string
}

variable "environment" {
  description = "Environment (dev, staging, prod)"
  default     = "dev"
  type        = string
}

variable "acr_branches" {
  description = "branches to use for acr tasks"
  type        = list(string)
  default     = ["main"]
}

variable "azure_cli_client_id" {
  description = "Azure CLI client ID for local development pre-authorization"
  type        = string
  default     = "04b07795-8ddb-461a-bbee-02f9e1bf7b46"
}