resource "azurerm_log_analytics_workspace" "trading" {
  name                = "${var.project_name}-logs"
  location            = "East US"
  resource_group_name = "trading-rg"
  sku                 = "PerGB2018"
  retention_in_days   = 30
}

resource "azurerm_monitor_diagnostic_setting" "aks" {
  name               = "${var.project_name}-aks-diag"
  target_resource_id = "/subscriptions/${var.subscription_id}/resourceGroups/trading-rg/providers/Microsoft.ContainerService/managedClusters/trading-aks"
  log_analytics_workspace_id = "/subscriptions/${var.subscription_id}/resourceGroups/trading-rg/providers/Microsoft.OperationalInsights/workspaces/${var.project_name}-logs"

  log {
    category = "kube-apiserver"
    enabled  = true
  }

  metric {
    category = "AllMetrics"
    enabled  = true
  }
}