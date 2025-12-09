import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form, Header
from fastapi.responses import HTMLResponse
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory

from azure.identity.aio import DefaultAzureCredential

from semantic_kernel import Kernel
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.filters import FunctionInvocationContext
from fastapi import Depends
from typing import AsyncGenerator
import asyncio
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging to provide maximum visibility for debugging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for maximum visibility
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    force=True  # Force reconfiguration even if logging was already configured
)

# Suppress verbose Azure telemetry logs but keep them at INFO level for important errors
logging.getLogger("azure.monitor.opentelemetry.exporter").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)

# Enable more detailed logging for key components to help debug hangs
logging.getLogger("azure.identity").setLevel(logging.INFO)  # Keep auth info visible
logging.getLogger("azure.core").setLevel(logging.INFO)  # Keep core Azure operations visible
logging.getLogger("semantic_kernel").setLevel(logging.DEBUG)  # Enable SK debug logs
logging.getLogger("fastapi").setLevel(logging.DEBUG)  # Enable FastAPI debug logs
logging.getLogger("uvicorn").setLevel(logging.DEBUG)  # Enable uvicorn debug logs
logging.getLogger("gunicorn").setLevel(logging.DEBUG)  # Enable gunicorn debug logs

# Disable urllib3 debug logs as they're too noisy
logging.getLogger("urllib3").setLevel(logging.INFO)

# Keep our application logger at DEBUG level
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
    logger.info("=== APPLICATION STARTUP BEGINNING ===")
    try:
        logger.debug("Initializing Azure credentials...")
        azure_creds = DefaultAzureCredential()
        logger.info("Azure credentials created successfully")
        
        # Note: init_chat will be called lazily per user on first request
        logger.info("=== APPLICATION STARTUP COMPLETED ===")

    except Exception as e:
        logger.error(f"Failed to initialize Azure credentials: {e}", exc_info=True)
        raise

    logger.info("FastAPI app is now ready to serve requests")
    yield
    
    # Shutdown
    logger.info("=== APPLICATION SHUTDOWN BEGINNING ===")
    logger.info("=== APPLICATION SHUTDOWN COMPLETED ===")


app = FastAPI(lifespan=lifespan)

# Add middleware to log all requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    logger.debug(f"Request headers: {dict(request.headers)}")
    
    response = await call_next(request)
    
    logger.info(f"Response: {response.status_code}")
    return response

@app.get("/")
async def root():
    """Root endpoint - simplest possible"""
    logger.info("Root endpoint accessed")
    return "Azure MCP Chat Agent is running!"

@app.get("/simple")
def simple():
    """Simplest sync endpoint"""
    logger.info("Simple endpoint accessed")
    return "OK"

@app.get("/ping")
async def ping():
    """Simplest possible endpoint to test if the app is responding"""
    logger.info("Ping endpoint accessed")
    return {"status": "pong", "timestamp": datetime.utcnow().isoformat()}

@app.get("/alive")
def alive():
    """Synchronous endpoint to test basic functionality"""
    logger.info("Alive endpoint accessed")
    return {"status": "alive", "pid": os.getpid()}



# Maintain persistent agent threads per context
user_threads: dict[str, ChatHistoryAgentThread] = {}


async def get_agent_and_thread_dependency(
    context_id: str = Form("default"),
    x_ms_client_principal_id: str = Header(..., alias="x-ms-client-principal-id"),
    x_ms_token_aad_access_token: str = Header(..., alias="x-ms-token-aad-access-token")
) -> AsyncGenerator[tuple[ChatCompletionAgent, ChatHistoryAgentThread], None]:
    """FastAPI dependency to get or create agent and thread with automatic cleanup"""
    
    logger.debug(f"Dependency called for user {x_ms_client_principal_id}, context {context_id}")
    
    try:
        logger.debug(f"Getting agent for user {x_ms_client_principal_id}")
        # Get or create agent for this user
        agent, azure_plugin = await init_chat(x_ms_token_aad_access_token, x_ms_client_principal_id)
        logger.debug(f"Agent retrieved successfully for user {x_ms_client_principal_id}")
        
        logger.debug(f"Ensuring MCP connection for user {x_ms_client_principal_id}")
        # Ensure MCP connection is active with retry logic
        await ensure_mcp_connection(azure_plugin, x_ms_client_principal_id)
        logger.debug(f"MCP connection established for user {x_ms_client_principal_id}")
        
        # Create a unique thread key combining user and context
        thread_key = f"{x_ms_client_principal_id}:{context_id}"
        logger.debug(f"Thread key: {thread_key}")
        
        # Get or create persistent thread for this user/context combination
        thread = user_threads.get(thread_key)
        if thread is None:
            logger.debug(f"Creating new thread for {thread_key}")
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
        else:
            logger.debug(f"Using existing thread for {thread_key}")
        
        logger.debug(f"Yielding agent and thread for {thread_key}")
        yield agent, thread
        
    except Exception as e:
        logger.error(f"Error in dependency for user {x_ms_client_principal_id}: {str(e)}", exc_info=True)
        raise
    finally:
        # FastAPI will automatically handle cleanup here if needed
        logger.debug(f"Dependency cleanup completed for thread {thread_key}")

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
    logger.debug(f"Ensuring MCP connection for user {user_id}")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.debug(f"MCP connection attempt {attempt + 1} for user {user_id}")
            await plugin.connect()
            logger.debug(f"MCP connection successful for user {user_id}")
            return True
        except Exception as e:
            logger.warning(f"MCP connection attempt {attempt + 1} failed for user {user_id}: {e}")
            if attempt == max_retries - 1:
                logger.error(f"Failed to establish MCP connection for user {user_id} after {max_retries} attempts")
                raise
            logger.debug(f"Waiting before retry for user {user_id}")
            await asyncio.sleep(1)  # Brief delay before retry
    return False


async def init_chat(user_token: str, user_id: str) -> tuple[ChatCompletionAgent, MCPStreamableHttpPlugin]:
    global azure_creds, user_agents, user_plugins
    
    # Use Azure-provided user ID as the cache key (stable across token renewals)
    user_key = user_id
    
    logger.debug(f"init_chat called for user: {user_key}")
    
    # Return cached instances if they exist
    if user_key in user_agents and user_key in user_plugins:
        logger.debug(f"Returning cached agent for user {user_key}")
        return user_agents[user_key], user_plugins[user_key]
    
    logger.info(f"Creating new agent for user {user_key}")
    
    try:
        logger.debug(f"Creating MCP plugin for user {user_key}")
        # Create MCP plugin with user token for OBO authentication
        headers = {"Authorization": f"Bearer {user_token}"}
        azure_plugin = MCPStreamableHttpPlugin(
            name="AzurePlugin",
            description="Azure Resources Plugin",
            load_prompts=False,
            url=os.getenv('MCP_URL', 'http://localhost:5008'),
            headers=headers
        )
        logger.debug(f"MCP plugin created successfully for user {user_key}")

        logger.debug(f"Creating kernel for user {user_key}")
        kernel = Kernel()
        kernel.add_filter("function_invocation", function_invocation_filter)
        
        # Add the Azure MCP plugin to the kernel so the agent can access its functions
        kernel.add_plugin(azure_plugin)
        logger.debug(f"Kernel and plugins configured for user {user_key}")

        logger.debug(f"Creating Azure OpenAI chat completion service for user {user_key}")
        # Add Azure OpenAI chat completion with better error handling
        chat_completion = AzureChatCompletion(
            api_key=os.getenv('AZURE_OPENAI_API_KEY'),
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            api_version="2024-12-01-preview",
            azure_credential=azure_creds
        )
        logger.debug(f"Azure OpenAI service created for user {user_key}")
        
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

        logger.debug(f"Creating ChatCompletionAgent for user {user_key}")
        agent = ChatCompletionAgent(
            kernel=kernel,
            service=chat_completion,
            name="SREAgent",
            instructions=sre_instructions
        )
        logger.debug(f"ChatCompletionAgent created successfully for user {user_key}")
        
        # Cache the agent and plugin for this user
        user_agents[user_key] = agent
        user_plugins[user_key] = azure_plugin
        logger.info(f"Agent and plugin cached successfully for user {user_key}")
        
        return agent, azure_plugin
        
    except Exception as e:
        logger.error(f"Error creating agent for user {user_key}: {str(e)}", exc_info=True)
        raise


@app.post("/chat")
async def chat(
    user_input: str = Form(...),
    agent_thread: tuple[ChatCompletionAgent, ChatHistoryAgentThread] = Depends(get_agent_and_thread_dependency)
):
    agent, thread = agent_thread
    logger.info(f"=== CHAT REQUEST START === Input: '{user_input}' for thread: {thread.thread_id}")
    
    try:
        logger.debug(f"Calling agent.get_response for thread {thread.thread_id}")
        # Get response from the agent - this automatically manages chat history
        response = await agent.get_response(message=user_input, thread=thread)
        logger.debug(f"Agent response received for thread {thread.thread_id}")
        
        # Log the response content
        response_content = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"SRE Agent response length: {len(response_content)} chars for thread {thread.thread_id}")
        logger.debug(f"SRE Agent full response: {response_content}")
        
        logger.info(f"=== CHAT REQUEST SUCCESS === for thread: {thread.thread_id}")
        return {"response": response_content}
        
    except Exception as e:
        logger.error(f"=== CHAT REQUEST ERROR === for thread {thread.thread_id}: {str(e)}", exc_info=True)
        return {"response": f"I encountered an error while processing your request. Please try again. Error: {str(e)}", "error": True}


@app.get("/health")
async def health_check():
    """Health check endpoint for debugging"""
    logger.info("=== HEALTH CHECK REQUEST ===")
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "azure_creds": "initialized" if azure_creds else "not_initialized",
            "user_agents_count": len(user_agents),
            "user_threads_count": len(user_threads),
            "mcp_url": os.getenv('MCP_URL', 'not_set'),
            "openai_endpoint": os.getenv('AZURE_OPENAI_ENDPOINT', 'not_set')[:50] + "..." if os.getenv('AZURE_OPENAI_ENDPOINT') else 'not_set'
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "error": str(e)}


@app.get("/debug/test-auth")
async def test_auth_headers(
    x_ms_client_principal_id: str = Header(None, alias="x-ms-client-principal-id"),
    x_ms_token_aad_access_token: str = Header(None, alias="x-ms-token-aad-access-token")
):
    """Test endpoint to check Azure App Service authentication headers"""
    logger.info(f"=== AUTH TEST REQUEST === User ID: {x_ms_client_principal_id}")
    
    return {
        "user_id": x_ms_client_principal_id,
        "has_token": bool(x_ms_token_aad_access_token),
        "token_length": len(x_ms_token_aad_access_token) if x_ms_token_aad_access_token else 0
    }


@app.get("/debug/test-agent")
async def test_agent_creation():
    """Test endpoint to check if agent can be created without auth"""
    logger.info("=== AGENT TEST REQUEST ===")
    
    # Use dummy values for testing
    dummy_token = "dummy_token"
    dummy_user_id = "test_user"
    
    try:
        logger.info("Testing agent creation with dummy values...")
        agent, plugin = await init_chat(dummy_token, dummy_user_id)
        logger.info("Agent created successfully!")
        
        return {
            "status": "success",
            "agent_name": agent.name,
            "plugin_name": plugin.name,
            "cached_agents": len(user_agents)
        }
    except Exception as e:
        logger.error(f"Agent creation failed: {str(e)}", exc_info=True)
        return {
            "status": "error", 
            "error": str(e),
            "error_type": type(e).__name__
        }


@app.get("/index", response_class=HTMLResponse)
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
    print("=== STARTING UVICORN DIRECTLY ===")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
