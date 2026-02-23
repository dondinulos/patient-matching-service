"""
Deploy Patient Matching Agent to Azure AI Foundry Agent Service.

This script registers the agent with its tools in the Foundry project.
Once deployed, the agent is accessible via the Foundry portal, API, or
any Agent Framework client.

Usage:
    python scripts/deploy_agent.py

Environment variables required:
    AZURE_AI_FOUNDRY_PROJECT_ENDPOINT  - Foundry project endpoint
    AZURE_OPENAI_DEPLOYMENT            - Model deployment name (default: gpt-4o)
    PM_DB_TYPE                         - Database type (default: cosmos)
    COSMOS_GREMLIN_ENDPOINT            - Cosmos DB Gremlin endpoint
    COSMOS_DATABASE                    - Cosmos DB database name
    COSMOS_CONTAINER                   - Cosmos DB container name
    COSMOS_KEY                         - Cosmos DB key
"""

import asyncio
import os
import sys
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.patient_matching.agent import create_foundry_agent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


async def deploy():
    """Deploy the Patient Matching Agent to Foundry."""
    project_endpoint = os.getenv("AZURE_AI_FOUNDRY_PROJECT_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    api_base_url = os.getenv("PM_API_BASE_URL")

    if not project_endpoint:
        logger.error(
            "AZURE_AI_FOUNDRY_PROJECT_ENDPOINT is required.\n"
            "Format: https://<resource>.services.ai.azure.com/api/projects/<project-name>"
        )
        sys.exit(1)

    logger.info("Deploying Patient Matching Agent to Foundry Agent Service")
    logger.info("  Project endpoint: %s", project_endpoint)
    logger.info("  Model deployment: %s", deployment)
    logger.info("  API base URL: %s", api_base_url or "(local function tools)")

    # create_foundry_agent returns an async context manager.
    # Entering the context registers the agent server-side if it doesn't exist.
    async with create_foundry_agent(
        project_endpoint=project_endpoint,
        deployment_name=deployment,
        agent_name="PatientMatchingAgent",
        api_base_url=api_base_url,
    ) as agent:
        logger.info("Agent registered successfully!")
        logger.info("  Agent name: PatientMatchingAgent")
        logger.info("  Agent ID:   %s", getattr(agent, 'id', 'N/A'))

        # Verify the agent works with a simple query
        logger.info("Running verification query...")
        result = await agent.run("What are the current service statistics?")
        logger.info("Agent responded: %s", result.text[:200] if result.text else "(no response)")

    logger.info("Deployment complete!")
    logger.info("")
    logger.info("To interact with the agent:")
    logger.info("  CLI:    python -m src.patient_matching.agent --foundry")
    logger.info("  Python:")
    logger.info("    async with create_foundry_agent() as agent:")
    logger.info('        result = await agent.run("Find matches for patient P123")')


if __name__ == "__main__":
    asyncio.run(deploy())
