resource "azurerm_virtual_network_peering" "aks_to_database" {
  name                      = "aks-db-peering"
  resource_group_name       = azurerm_resource_group.trading.name
  virtual_network_name      = azurerm_virtual_network.trading.name
  remote_virtual_network_id = azurerm_virtual_network.database.id
  allow_forwarded_traffic  = true
}

resource "azurerm_virtual_network" "database" {
  name                = "database-vnet"
  address_space       = ["10.2.0.0/16"]
  location            = azurerm_resource_group.trading.location
  resource_group_name = azurerm_resource_group.trading.name
}