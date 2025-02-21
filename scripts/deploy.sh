#!/bin/bash

# Login to Azure
az login

# Set subscription
az account set --subscription $AZURE_SUBSCRIPTION_ID

# Initialize Terraform
terraform init

# Validate Terraform configuration
terraform validate

# Plan the deployment
terraform plan -out=tfplan

# Apply the configuration
terraform apply tfplan 