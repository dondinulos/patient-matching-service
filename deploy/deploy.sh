#!/bin/bash
#
# Deploy Patient Matching Service to Azure
#
# Usage:
#   ./deploy.sh                          # Deploy with defaults (dev environment)
#   ./deploy.sh -e prod -l eastus2       # Deploy to prod in East US 2
#   ./deploy.sh --skip-infra             # Skip infrastructure, only build container
#   ./deploy.sh --skip-build             # Skip build, only deploy infrastructure
#

set -e

# Configuration - set via environment variables
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:?Error: AZURE_SUBSCRIPTION_ID environment variable is required}"
TENANT_ID="${AZURE_TENANT_ID:?Error: AZURE_TENANT_ID environment variable is required}"
RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-rg-patient-matching}"
BASE_NAME="patientmatch"

# Defaults
ENVIRONMENT="dev"
LOCATION="westus2"
SKIP_INFRASTRUCTURE=false
SKIP_BUILD=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
info() { echo -e "${CYAN}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        --skip-infra)
            SKIP_INFRASTRUCTURE=true
            shift
            ;;
        --skip-build)
            SKIP_BUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -e, --environment  Environment (dev, staging, prod) [default: dev]"
            echo "  -l, --location     Azure region [default: westus2]"
            echo "  --skip-infra       Skip infrastructure deployment"
            echo "  --skip-build       Skip Docker build and push"
            echo "  -h, --help         Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be dev, staging, or prod."
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Banner
echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           Patient Matching Service - Azure Deployment         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

info "Environment: $ENVIRONMENT"
info "Location: $LOCATION"
info "Subscription: $SUBSCRIPTION_ID"
info "Tenant: $TENANT_ID"
info "Resource Group: $RESOURCE_GROUP"
echo ""

# Step 1: Login to Azure
info "Checking Azure CLI login status..."
if ! az account show &> /dev/null; then
    info "Logging in to Azure..."
    az login --tenant "$TENANT_ID" || error "Failed to login to Azure"
fi

# Set subscription
info "Setting subscription to $SUBSCRIPTION_ID..."
az account set --subscription "$SUBSCRIPTION_ID" || error "Failed to set subscription"
success "Subscription set successfully"

# Step 2: Create Resource Group if not exists
info "Checking resource group $RESOURCE_GROUP..."
if [ "$(az group exists --name "$RESOURCE_GROUP")" == "false" ]; then
    info "Creating resource group $RESOURCE_GROUP in $LOCATION..."
    az group create \
        --name "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --tags "environment=$ENVIRONMENT" "application=patient-matching" \
        || error "Failed to create resource group"
    success "Resource group created"
else
    success "Resource group already exists"
fi

# Step 3: Deploy Infrastructure
if [ "$SKIP_INFRASTRUCTURE" == "false" ]; then
    info "Deploying Azure infrastructure with Bicep..."
    
    DEPLOYMENT_NAME="patient-matching-$ENVIRONMENT-$(date +%Y%m%d%H%M%S)"
    BICEP_FILE="$SCRIPT_DIR/main.bicep"
    
    DEPLOYMENT_OUTPUT=$(az deployment group create \
        --name "$DEPLOYMENT_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --template-file "$BICEP_FILE" \
        --parameters environment="$ENVIRONMENT" baseName="$BASE_NAME" \
        --output json) || error "Infrastructure deployment failed"
    
    success "Infrastructure deployed successfully"
    
    # Parse outputs
    ACR_LOGIN_SERVER=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerRegistryLoginServer.value')
    ACR_NAME=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerRegistryName.value')
    CONTAINER_APP_URL=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.containerAppUrl.value')
    COSMOS_ENDPOINT=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.cosmosDbEndpoint.value')
    OPENAI_ENDPOINT=$(echo "$DEPLOYMENT_OUTPUT" | jq -r '.properties.outputs.openAiEndpoint.value')
    
    echo ""
    info "Deployment Outputs:"
    echo "  Container Registry: $ACR_LOGIN_SERVER"
    echo "  Container App URL: $CONTAINER_APP_URL"
    echo "  Cosmos DB Endpoint: $COSMOS_ENDPOINT"
    echo "  OpenAI Endpoint: $OPENAI_ENDPOINT"
else
    warning "Skipping infrastructure deployment"
    
    # Get existing ACR name
    ACR_NAME=$(az acr list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv)
    if [ -z "$ACR_NAME" ]; then
        error "No Container Registry found. Run without --skip-infra first."
    fi
    ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query "loginServer" -o tsv)
fi

# Step 4: Build and Push Docker Image
if [ "$SKIP_BUILD" == "false" ]; then
    echo ""
    info "Building and pushing Docker image..."
    
    # Login to ACR
    info "Logging in to Azure Container Registry..."
    az acr login --name "$ACR_NAME" || error "Failed to login to ACR"
    
    # Build image
    IMAGE_TAG=$(date +%Y%m%d%H%M%S)
    IMAGE_NAME="$ACR_LOGIN_SERVER/patient-matching"
    
    info "Building Docker image..."
    cd "$PROJECT_ROOT"
    
    docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" -t "${IMAGE_NAME}:latest" . \
        || error "Docker build failed"
    
    info "Pushing Docker image..."
    docker push "${IMAGE_NAME}:${IMAGE_TAG}" || error "Docker push failed"
    docker push "${IMAGE_NAME}:latest" || error "Docker push failed"
    
    success "Docker image pushed: ${IMAGE_NAME}:${IMAGE_TAG}"
    
    # Update Container App with new image
    info "Updating Container App with new image..."
    CONTAINER_APP_NAME="ca-${BASE_NAME}-${ENVIRONMENT}-api"
    az containerapp update \
        --name "$CONTAINER_APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --image "${IMAGE_NAME}:${IMAGE_TAG}" \
        || warning "Container App update failed - app may need manual update"
    
    success "Container App updated with new image"
else
    warning "Skipping Docker build"
fi

# Step 5: Display Summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Deployment Complete!                       ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get Container App URL
CONTAINER_APP_NAME="ca-${BASE_NAME}-${ENVIRONMENT}-api"
APP_URL=$(az containerapp show \
    --name "$CONTAINER_APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query "properties.configuration.ingress.fqdn" -o tsv 2>/dev/null)

if [ -n "$APP_URL" ]; then
    success "Application URL: https://$APP_URL"
    echo ""
    info "Test endpoints:"
    echo "  Health: https://$APP_URL/health"
    echo "  API Docs: https://$APP_URL/docs"
    echo "  OpenAPI: https://$APP_URL/openapi.json"
fi

echo ""
info "Next steps:"
echo "  1. Verify the application is running:"
echo "     az containerapp logs show -n $CONTAINER_APP_NAME -g $RESOURCE_GROUP"
echo "  2. Check application health:"
echo "     curl https://$APP_URL/health"
echo "  3. View API documentation:"
echo "     https://$APP_URL/docs"
echo ""
