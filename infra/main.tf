terraform {
  required_version = ">= 0.12"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

# Resource Group
resource "azurerm_resource_group" "trading" {
  name     = "${var.project_name}-rg"
  location = "East US"  # Use hardcoded value temporarily
}

# AKS Cluster
resource "azurerm_kubernetes_cluster" "trading" {
  name                = "${var.project_name}-aks"
  location            = "East US"
  resource_group_name = "${var.project_name}-rg"
  dns_prefix          = "${var.project_name}-aks"

  default_node_pool {
    name       = "default"
    node_count = 3
    vm_size    = "Standard_B2s"
    enable_auto_scaling = true
    min_count  = 1
    max_count  = 5
  }

  identity {
    type = "SystemAssigned"
  }

  auto_scaler_profile {
    balance_similar_node_groups = true
    max_graceful_termination_sec = 600
    scale_down_delay_after_add = "10m"
  }
}

# PostgreSQL Server
resource "azurerm_postgresql_flexible_server" "trading" {
  name                   = "${var.project_name}-db"
  resource_group_name    = "${var.project_name}-rg"
  location              = "East US"
  version               = "15"
  administrator_login    = "${var.db_username}"
  administrator_password = "${var.db_password}"
  storage_mb            = 32768
  sku_name              = "${var.db_sku}"
  zone                  = "1"
}

# Redis Cache
resource "azurerm_redis_cache" "trading" {
  name                = "${var.project_name}-redis"
  location            = "East US"
  resource_group_name = "${var.project_name}-rg"
  capacity            = 1
  family              = "P"
  sku_name            = "Premium"
  non_ssl_port_enabled = false
  minimum_tls_version = "1.2"

  redis_configuration {
    maxmemory_reserved = 2
    maxmemory_delta    = 2
    maxmemory_policy   = "allkeys-lru"
  }
}

# Container Registry
resource "azurerm_container_registry" "trading" {
  name                = "tradingacr"
  resource_group_name = "${var.project_name}-rg"
  location            = "East US"
  sku                 = "Premium"
  admin_enabled       = true
}

# Key Vault
resource "azurerm_key_vault" "trading" {
  name                = "${var.project_name}-kv"
  location            = "East US"
  resource_group_name = "${var.project_name}-rg"
  tenant_id          = "${var.tenant_id}"
  sku_name           = "premium"

  access_policy {
    tenant_id = "${var.tenant_id}"
    object_id = "${var.object_id}"

    secret_permissions = [
      "Get", "List", "Set", "Delete"
    ]
  }
}

# Network Security Group
resource "azurerm_network_security_group" "trading" {
  name                = "trading-nsg"
  location            = "East US"
  resource_group_name = "${var.project_name}-rg"

  security_rule {
    name                       = "https"
    priority                   = 100
    direction                  = "Inbound"
    access                     = "Allow"
    protocol                   = "Tcp"
    source_port_range          = "*"
    destination_port_range      = "443"
    source_address_prefix      = "*"
    destination_address_prefix = "*"
  }
}

# Virtual Network
resource "azurerm_virtual_network" "trading" {
  name                = "trading-vnet"
  address_space       = ["10.1.0.0/16"]  # Hardcode temporarily
  location            = "East US"
  resource_group_name = "${var.project_name}-rg"
}

# Subnets
resource "azurerm_subnet" "aks" {
  name                 = "aks-subnet"
  resource_group_name  = "${var.project_name}-rg"
  virtual_network_name = "trading-vnet"
  address_prefixes     = ["10.1.1.0/24"]
}

resource "azurerm_subnet" "appgw" {
  name                 = "appgw-subnet"
  resource_group_name  = "${var.project_name}-rg"
  virtual_network_name = "trading-vnet"
  address_prefixes     = ["10.1.2.0/24"]
}

# Public IP for Application Gateway
resource "azurerm_public_ip" "appgw" {
  name                = "appgw-pip"
  resource_group_name = "${var.project_name}-rg"
  location            = "East US"
  allocation_method   = "Static"
  sku                = "Standard"
}

# Application Gateway
resource "azurerm_application_gateway" "trading" {
  name                = "trading-appgw"
  resource_group_name = "${var.project_name}-rg"
  location            = "East US"

  sku {
    name     = "WAF_v2"
    tier     = "WAF_v2"
    capacity = 2
  }

  gateway_ip_configuration {
    name      = "appgw-ip-config"
    subnet_id = "/subscriptions/${var.subscription_id}/resourceGroups/${var.project_name}-rg/providers/Microsoft.Network/virtualNetworks/trading-vnet/subnets/appgw-subnet"
  }

  frontend_port {
    name = "https"
    port = 443
  }

  frontend_ip_configuration {
    name                 = "appgw-feip"
    public_ip_address_id = "/subscriptions/${var.subscription_id}/resourceGroups/${var.project_name}-rg/providers/Microsoft.Network/publicIPAddresses/appgw-pip"
  }

  # Backend address pool
  backend_address_pool {
    name = "trading-backend-pool"
  }

  # Backend HTTP settings
  backend_http_settings {
    name                  = "trading-http-settings"
    cookie_based_affinity = "Disabled"
    port                 = 80
    protocol             = "Http"
    request_timeout      = 60
  }

  # HTTP Listener
  http_listener {
    name                           = "trading-listener"
    frontend_ip_configuration_name = "appgw-feip"
    frontend_port_name            = "https"
    protocol                      = "Https"
  }

  # Request routing rule
  request_routing_rule {
    name                       = "trading-rule"
    rule_type                 = "Basic"
    http_listener_name        = "trading-listener"
    backend_address_pool_name  = "trading-backend-pool"
    backend_http_settings_name = "trading-http-settings"
    priority                  = 100
  }
}

data "azurerm_client_config" "current" {}

output "aks_cluster_name" {
  value = "${var.project_name}-aks"
}

output "postgres_host" {
  value = "${var.project_name}-db.postgres.database.azure.com"
}

output "redis_host" {
  value = "${var.project_name}-redis.redis.cache.windows.net"
}

# Backup Configuration
resource "azurerm_recovery_services_vault" "trading" {
  name                = "trading-backup-vault"
  location            = "East US"
  resource_group_name = "${var.project_name}-rg"
  sku                 = "Standard"
}

resource "azurerm_backup_policy_vm" "db" {
  name                = "db-backup-policy"
  resource_group_name = "${var.project_name}-rg"
  recovery_vault_name = "trading-backup-vault"

  backup {
    frequency = "Daily"
    time      = "23:00"
  }

  retention_daily {
    count = 30
  }
}

# Debug outputs
output "debug_db_username" {
  value = "${var.db_username}"
}

output "debug_project_name" {
  value = "${var.project_name}"
} 