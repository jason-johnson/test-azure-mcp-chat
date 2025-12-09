import os
import logging
from uuid import uuid4
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form, Header
from fastapi.responses import HTMLResponse
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents import AuthorRole
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel import Kernel
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.filters import FunctionInvocationContext
from fastapi import Depends
from typing import AsyncGenerator
import asyncio

# Load environment variables
load_dotenv()

# Configure logging to reduce Azure telemetry noise
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose Azure telemetry logs
logging.getLogger("azure.monitor.opentelemetry.exporter").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
#logging.getLogger("azure.identity").setLevel(logging.WARNING)
#logging.getLogger("azure.core").setLevel(logging.WARNING)

# Keep our application logger at INFO level
logger = logging.getLogger(__name__)

# Global variables for Azure resources
azure_creds = None
azure_mcp = None
agent = None
azure_plugin = None

# Cache agents per user to avoid recreating MCP connections
user_agents: dict[str, ChatCompletionAgent] = {}
user_plugins: dict[str, MCPStreamableHttpPlugin] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global azure_creds
    logger.info("Initializing Azure credentials and client...")
    try:
        logger.info("Creating Azure credentials...")
        azure_creds = DefaultAzureCredential()
        
        # Note: init_chat will be called lazily per user on first request

    except Exception as e:
        logger.error(f"Failed to initialize Azure credentials: {e}")

    yield
    
    # Shutdown
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Maintain persistent agent threads per context
user_threads: dict[str, ChatHistoryAgentThread] = {}


async def get_agent_and_thread_dependency(
    context_id: str = Form("default"),
    x_ms_client_principal_id: str = Header(..., alias="x-ms-client-principal-id"),
    x_ms_token_aad_access_token: str = Header(..., alias="x-ms-token-aad-access-token")
) -> AsyncGenerator[tuple[ChatCompletionAgent, ChatHistoryAgentThread], None]:
    """FastAPI dependency to get or create agent and thread with automatic cleanup"""
    
    # Get or create agent for this user
    agent, azure_plugin = await init_chat(x_ms_token_aad_access_token, x_ms_client_principal_id)
    
    # Ensure MCP connection is active with retry logic
    await ensure_mcp_connection(azure_plugin, x_ms_client_principal_id)
    
    # Create a unique thread key combining user and context
    thread_key = f"{x_ms_client_principal_id}:{context_id}"
    
    # Get or create persistent thread for this user/context combination
    thread = user_threads.get(thread_key)
    if thread is None:
        # Create new thread with proper system message from agent
        chat_history = ChatHistory(
            messages=[],
            system_message=agent.instructions
        )
        thread = ChatHistoryAgentThread(
            chat_history=chat_history, 
            thread_id=thread_key  # Use stable thread ID for persistence
        )
        user_threads[thread_key] = thread
        logger.info(f"Created new persistent thread for {thread_key}")
    
    try:
        yield agent, thread
    finally:
        # FastAPI will automatically handle cleanup here if needed
        logger.debug(f"Request completed for thread {thread_key}")

# Function invocation filter to log function calls and responses


async def function_invocation_filter(context: FunctionInvocationContext, next):
    """A filter that will be called for each function call in the response."""
    if "messages" not in context.arguments:
        await next(context)
        return
    logger.info(
        f"    Agent [{context.function.name}] called with messages: {context.arguments['messages']}")
    await next(context)
    logger.info(
        f"    Response from agent [{context.function.name}]: {context.result.value}")


async def ensure_mcp_connection(plugin: MCPStreamableHttpPlugin, user_id: str):
    """Ensure MCP connection is active with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await plugin.connect()
            return True
        except Exception as e:
            logger.warning(f"MCP connection attempt {attempt + 1} failed for user {user_id}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to establish MCP connection for user {user_id} after {max_retries} attempts")
                raise
            await asyncio.sleep(1)  # Brief delay before retry
    return False


async def init_chat(user_token: str, user_id: str) -> tuple[ChatCompletionAgent, MCPStreamableHttpPlugin]:
    global azure_creds, user_agents, user_plugins
    
    # Use Azure-provided user ID as the cache key (stable across token renewals)
    user_key = user_id
    
    # Return cached instances if they exist
    if user_key in user_agents and user_key in user_plugins:
        logger.info(f"Returning cached agent for user {user_key}")
        return user_agents[user_key], user_plugins[user_key]
    
    logger.info(f"Creating new agent for user {user_key}")
    
    # Create MCP plugin with user token for OBO authentication
    headers = {"Authorization": f"Bearer {user_token}"}
    azure_plugin = MCPStreamableHttpPlugin(
        name="AzurePlugin",
        description="Azure Resources Plugin",
        load_prompts=False,
        url=os.getenv('MCP_URL', 'http://localhost:5008'),
        headers=headers
    )

    kernel = Kernel()
    kernel.add_filter("function_invocation", function_invocation_filter)
    
    # Add the Azure MCP plugin to the kernel so the agent can access its functions
    kernel.add_plugin(azure_plugin)

    # Add Azure OpenAI chat completion with better error handling
    chat_completion = AzureChatCompletion(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        api_version="2024-12-01-preview",
        azure_credential=azure_creds
    )
    
    # Define comprehensive SRE instructions
    sre_instructions = """
Role: Azure Service Reliability Engineer (SRE)

You are an expert Azure SRE responsible for:
- Investigating and resolving incidents and outages
- Responding to monitoring alerts with actionable insights
- Proactively identifying potential issues
- Providing technical guidance to the SRE team
- Executing Azure operations directly when requested

Behavior Guidelines:
1. Always use available tools to gather data before providing answers
2. When users ask about Azure tasks, execute them directly rather than providing instructions
3. Provide clear, actionable recommendations based on data
4. Include relevant metrics, logs, or configuration details in your responses
5. Prioritize system reliability and security in all recommendations

Use your Azure tools to investigate, analyze, and take action as appropriate.
"""

    agent = ChatCompletionAgent(
        kernel=kernel,
        service=chat_completion,
        name="SREAgent",
        instructions=sre_instructions
    )
    
    # Cache the agent and plugin for this user
    user_agents[user_key] = agent
    user_plugins[user_key] = azure_plugin
    
    return agent, azure_plugin


@app.post("/chat")
async def chat(
    user_input: str = Form(...),
    agent_thread: tuple[ChatCompletionAgent, ChatHistoryAgentThread] = Depends(get_agent_and_thread_dependency)
):
    agent, thread = agent_thread
    logger.info(f"Received chat request: {user_input} for thread: {thread.thread_id}")
    
    try:
        # Get response from the agent - this automatically manages chat history
        response = await agent.get_response(message=user_input, thread=thread)
        
        # Log the response content
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"SRE Agent response: {response_content[:100]}...")  # Truncate long responses in logs
        
        return {"response": response_content}
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}", exc_info=True)
        return {"response": f"I encountered an error while processing your request. Please try again. Error: {str(e)}", "error": True}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error(
            "index.html file not found. Please ensure it exists in the current directory.")
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
