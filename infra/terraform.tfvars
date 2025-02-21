# Project Settings
project_name = "trading"
location     = "East US"

# AKS Configuration
node_count     = 3
vm_size        = "Standard_B2s"
min_node_count = 1
max_node_count = 5

# Database Configuration
db_username = "dbadmin"
db_password = "YourSecurePassword123!"  # Should be in a secure vault in production
db_sku      = "GP_Standard_D4s_v3"

# Redis Configuration
redis_capacity = 1
redis_family   = "P"
redis_sku      = "Premium"

# Network Configuration
vnet_address_space = ["10.1.0.0/16"]
aks_subnet_prefix  = ["10.1.1.0/24"]
appgw_subnet_prefix = ["10.1.2.0/24"]

tenant_id = "a8eec281-aaa3-4dae-ac9b-9a398b9215e7"  # Get this from Azure portal or `az account show`

# Azure AD Configuration
object_id = "c7ad7f7f-a927-4e03-b507-7874cfae6537"
subscription_id = "1b036bc3-d33a-4a61-a8b4-edb33458e352"