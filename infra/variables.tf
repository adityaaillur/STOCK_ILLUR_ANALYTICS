variable "project_name" {
  description = "Project name"
  type        = "string"
}

variable "location" {
  description = "Azure region"
  type        = string
}

variable "node_count" {
  description = "Number of AKS nodes"
  type        = number
  default     = 3
}

variable "vm_size" {
  description = "VM size for AKS nodes"
  type        = string
  default     = "Standard_B2s"
}

variable "min_node_count" {
  description = "Minimum number of nodes for autoscaling"
  type        = number
  default     = 1
}

variable "max_node_count" {
  description = "Maximum number of nodes for autoscaling"
  type        = number
  default     = 5
}

variable "db_username" {
  description = "Database administrator username"
  type        = string
}

variable "db_password" {
  description = "Database administrator password"
  type        = string
  sensitive   = true
}

variable "db_sku" {
  description = "Database SKU"
  type        = string
  default     = "GP_Standard_D4s_v3"
}

# Azure Configuration
variable "subscription_id" {
  description = "Azure subscription ID"
  type        = string
}

# Redis Configuration
variable "redis_capacity" {
  description = "Redis cache capacity"
  type        = number
  default     = 1
}

variable "redis_family" {
  description = "Redis cache family"
  type        = string
  default     = "P"
}

variable "redis_sku" {
  description = "Redis cache SKU"
  type        = string
  default     = "Premium"
}

# Network Configuration
variable "vnet_address_space" {
  description = "Virtual network address space"
  type        = list
  default     = ["10.1.0.0/16"]
}

variable "aks_subnet_prefix" {
  description = "AKS subnet address prefix"
  type        = list
  default     = ["10.1.1.0/24"]
}

variable "appgw_subnet_prefix" {
  description = "Application Gateway subnet address prefix"
  type        = list
  default     = ["10.1.2.0/24"]
}

variable "tenant_id" {
  description = "Azure tenant ID"
  type        = string
}

variable "object_id" {
  description = "Azure AD Object ID"
  type        = string
} 