<#
.SYNOPSIS
    Deploy Patient Matching Service to Azure

.DESCRIPTION
    This script deploys all Azure infrastructure and the Patient Matching Service application.
    
    Resources deployed:
    - Azure Container Registry
    - Azure Container Apps Environment
    - Azure Container App (API)
    - Azure Cosmos DB (Gremlin API)
    - Azure OpenAI Service with GPT-4o and text-embedding-ada-002

.PARAMETER Environment
    Target environment: dev, staging, or prod

.PARAMETER Location
    Azure region for deployment (default: westus2)

.PARAMETER SkipInfrastructure
    Skip infrastructure deployment, only build and push container

.PARAMETER SkipBuild
    Skip Docker build, only deploy infrastructure

.EXAMPLE
    .\deploy.ps1 -Environment dev
    
.EXAMPLE
    .\deploy.ps1 -Environment prod -Location eastus2
#>

param(
    [Parameter(Mandatory = $false)]
    [ValidateSet("dev", "staging", "prod")]
    [string]$Environment = "dev",

    [Parameter(Mandatory = $false)]
    [string]$Location = "westus2",

    [Parameter(Mandatory = $false)]
    [switch]$SkipInfrastructure,

    [Parameter(Mandatory = $false)]
    [switch]$SkipBuild
)

# Configuration
$SubscriptionId = "cde003d3-960c-474d-a3a8-aa57d803282f"
$TenantId = "REDACTED"
$ResourceGroup = "rg-patient-matching"
$BaseName = "patientmatch"

# Colors for output
function Write-Info { param($Message) Write-Host "[INFO] $Message" -ForegroundColor Cyan }
function Write-Success { param($Message) Write-Host "[SUCCESS] $Message" -ForegroundColor Green }
function Write-Warning { param($Message) Write-Host "[WARNING] $Message" -ForegroundColor Yellow }
function Write-Error { param($Message) Write-Host "[ERROR] $Message" -ForegroundColor Red }

# Banner
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║           Patient Matching Service - Azure Deployment         ║" -ForegroundColor Blue
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

Write-Info "Environment: $Environment"
Write-Info "Location: $Location"
Write-Info "Subscription: $SubscriptionId"
Write-Info "Tenant: $TenantId"
Write-Info "Resource Group: $ResourceGroup"
Write-Host ""

# Step 1: Login to Azure
Write-Info "Checking Azure CLI login status..."
$loginStatus = az account show 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Info "Logging in to Azure..."
    az login --tenant $TenantId
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to login to Azure"
        exit 1
    }
}

# Set subscription
Write-Info "Setting subscription to $SubscriptionId..."
az account set --subscription $SubscriptionId
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to set subscription"
    exit 1
}
Write-Success "Subscription set successfully"

# Step 2: Create Resource Group if not exists
Write-Info "Checking resource group $ResourceGroup..."
$rgExists = az group exists --name $ResourceGroup
if ($rgExists -eq "false") {
    Write-Info "Creating resource group $ResourceGroup in $Location..."
    az group create --name $ResourceGroup --location $Location --tags "environment=$Environment" "application=patient-matching"
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to create resource group"
        exit 1
    }
    Write-Success "Resource group created"
} else {
    Write-Success "Resource group already exists"
}

# Step 3: Deploy Infrastructure
if (-not $SkipInfrastructure) {
    Write-Info "Deploying Azure infrastructure with Bicep..."
    
    $deploymentName = "patient-matching-$Environment-$(Get-Date -Format 'yyyyMMddHHmmss')"
    $bicepFile = Join-Path $PSScriptRoot "main.bicep"
    
    $deploymentOutput = az deployment group create `
        --name $deploymentName `
        --resource-group $ResourceGroup `
        --template-file $bicepFile `
        --parameters environment=$Environment baseName=$BaseName `
        --output json
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Infrastructure deployment failed"
        exit 1
    }
    
    Write-Success "Infrastructure deployed successfully"
    
    # Parse outputs
    $outputs = $deploymentOutput | ConvertFrom-Json
    $acrLoginServer = $outputs.properties.outputs.containerRegistryLoginServer.value
    $acrName = $outputs.properties.outputs.containerRegistryName.value
    $containerAppUrl = $outputs.properties.outputs.containerAppUrl.value
    $cosmosEndpoint = $outputs.properties.outputs.cosmosDbEndpoint.value
    $openAiEndpoint = $outputs.properties.outputs.openAiEndpoint.value
    
    Write-Host ""
    Write-Info "Deployment Outputs:"
    Write-Host "  Container Registry: $acrLoginServer" -ForegroundColor White
    Write-Host "  Container App URL: $containerAppUrl" -ForegroundColor White
    Write-Host "  Cosmos DB Endpoint: $cosmosEndpoint" -ForegroundColor White
    Write-Host "  OpenAI Endpoint: $openAiEndpoint" -ForegroundColor White
} else {
    Write-Warning "Skipping infrastructure deployment"
    
    # Get existing ACR name
    $acrList = az acr list --resource-group $ResourceGroup --query "[0].name" -o tsv
    if ($acrList) {
        $acrName = $acrList
        $acrLoginServer = az acr show --name $acrName --query "loginServer" -o tsv
    } else {
        Write-Error "No Container Registry found. Run without -SkipInfrastructure first."
        exit 1
    }
}

# Step 4: Build and Push Docker Image
if (-not $SkipBuild) {
    Write-Host ""
    Write-Info "Building and pushing Docker image..."
    
    # Login to ACR
    Write-Info "Logging in to Azure Container Registry..."
    az acr login --name $acrName
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to login to ACR"
        exit 1
    }
    
    # Build image
    $imageTag = "$(Get-Date -Format 'yyyyMMddHHmmss')"
    $imageName = "$acrLoginServer/patient-matching"
    
    Write-Info "Building Docker image..."
    $projectRoot = Split-Path -Parent $PSScriptRoot
    Push-Location $projectRoot
    
    docker build -t "${imageName}:${imageTag}" -t "${imageName}:latest" .
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "Docker build failed"
        exit 1
    }
    
    Write-Info "Pushing Docker image..."
    docker push "${imageName}:${imageTag}"
    docker push "${imageName}:latest"
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "Docker push failed"
        exit 1
    }
    
    Pop-Location
    Write-Success "Docker image pushed: ${imageName}:${imageTag}"
    
    # Update Container App with new image
    Write-Info "Updating Container App with new image..."
    $containerAppName = "ca-${BaseName}-${Environment}-api"
    az containerapp update `
        --name $containerAppName `
        --resource-group $ResourceGroup `
        --image "${imageName}:${imageTag}"
    
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Container App update failed - app may need manual update"
    } else {
        Write-Success "Container App updated with new image"
    }
} else {
    Write-Warning "Skipping Docker build"
}

# Step 5: Display Summary
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║                    Deployment Complete!                       ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""

# Get Container App URL
$containerAppName = "ca-${BaseName}-${Environment}-api"
$appUrl = az containerapp show --name $containerAppName --resource-group $ResourceGroup --query "properties.configuration.ingress.fqdn" -o tsv 2>$null

if ($appUrl) {
    Write-Success "Application URL: https://$appUrl"
    Write-Host ""
    Write-Info "Test endpoints:"
    Write-Host "  Health: https://$appUrl/health" -ForegroundColor White
    Write-Host "  API Docs: https://$appUrl/docs" -ForegroundColor White
    Write-Host "  OpenAPI: https://$appUrl/openapi.json" -ForegroundColor White
}

Write-Host ""
Write-Info "Next steps:"
Write-Host "  1. Verify the application is running: az containerapp logs show -n $containerAppName -g $ResourceGroup" -ForegroundColor White
Write-Host "  2. Check application health: curl https://$appUrl/health" -ForegroundColor White
Write-Host "  3. View API documentation: https://$appUrl/docs" -ForegroundColor White
Write-Host ""
