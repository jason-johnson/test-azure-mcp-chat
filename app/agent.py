import os
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from typing import Tuple

from fastapi import FastAPI, Request, Form, Header
from fastapi.responses import HTMLResponse
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.connectors.ai import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory

from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential
from azure.identity import DefaultAzureCredential

from semantic_kernel import Kernel
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.mcp import MCPStreamableHttpPlugin
from semantic_kernel.filters import FunctionInvocationContext
from fastapi import Depends
from typing import AsyncGenerator
import asyncio
import hashlib
import time
from datetime import datetime, timezone

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
logging.getLogger("semantic_kernel").setLevel(logging.INFO)  # Reduce SK verbosity - was DEBUG
logging.getLogger("fastapi").setLevel(logging.DEBUG)  # Enable FastAPI debug logs
logging.getLogger("uvicorn").setLevel(logging.DEBUG)  # Enable uvicorn debug logs
logging.getLogger("gunicorn").setLevel(logging.DEBUG)  # Enable gunicorn debug logs

# Disable urllib3 debug logs as they're too noisy
logging.getLogger("urllib3").setLevel(logging.INFO)

# Reduce MCP client verbosity - tool definitions are very verbose at DEBUG level
logging.getLogger("mcp.client.streamable_http").setLevel(logging.INFO)

# Keep our application logger at DEBUG level
logger = logging.getLogger(__name__)

# Global variables for Azure resources
azure_mcp = None
agent = None
azure_plugin = None

# Cache agents per user to avoid recreating MCP connections
user_agents: dict[str, ChatCompletionAgent] = {}
user_plugins: dict[str, MCPStreamableHttpPlugin] = {}
user_cache_timestamps: dict[str, float] = {}  # Track when each cache entry was created
CACHE_TTL_MINUTES = 45  # Cache TTL in minutes (less than typical 60-min token lifetime)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("=== APPLICATION STARTUP BEGINNING ===")
    logger.info("Azure credentials will be initialized on first user request")
    logger.info("=== APPLICATION STARTUP COMPLETED ===")
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

@app.get("/ping")
async def ping():
    """Simplest possible endpoint to test if the app is responding"""
    logger.info("Ping endpoint accessed")
    return {"status": "pong", "timestamp": datetime.now(timezone.utc).isoformat()}

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
    x_ms_token_aad_access_token: str = Header(..., alias="x-ms-token-aad-access-token"),
    x_ms_token_aad_refresh_token: str = Header(None, alias="x-ms-token-aad-refresh-token")
) -> AsyncGenerator[tuple[ChatCompletionAgent, ChatHistoryAgentThread], None]:
    """FastAPI dependency to get or create agent and thread with automatic cleanup"""
    
    logger.debug(f"Dependency called for user {x_ms_client_principal_id}, context {context_id}")
    
    # Initialize thread_key early to avoid UnboundLocalError in finally block
    thread_key = f"{x_ms_client_principal_id}:{context_id}"
    
    try:
        logger.debug(f"Getting agent for user {x_ms_client_principal_id}")
        # Get or create agent for this user
        agent, azure_plugin = await init_chat(x_ms_token_aad_access_token, x_ms_client_principal_id, x_ms_token_aad_refresh_token)
        logger.debug(f"Agent retrieved successfully for user {x_ms_client_principal_id}")
        
        logger.debug(f"Ensuring MCP connection for user {x_ms_client_principal_id}")
        # Ensure MCP connection is active with retry logic
        await ensure_mcp_connection(azure_plugin, x_ms_client_principal_id)
        logger.debug(f"MCP connection established for user {x_ms_client_principal_id}")
        
        # Use the thread key for thread management
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
        yield agent, thread, thread_key
        
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
            logger.info(f"MCP connection successful for user {user_id}")
            
            # Log details about the plugin state after connection
            logger.info(f"Plugin session after connect: {plugin.session}")
            logger.info(f"Plugin load_tools_flag: {plugin.load_tools_flag}")
            
            # Check if functions were loaded
            if hasattr(plugin, 'functions'):
                func_count = len(plugin.functions) if plugin.functions else 0
                logger.info(f"Functions loaded after connect: {func_count}")
                if func_count > 0:
                    logger.info(f"Function names: {list(plugin.functions.keys())[:10]}")
            
            # Try to manually trigger load_tools if no functions were loaded
            if hasattr(plugin, 'functions') and (not plugin.functions or len(plugin.functions) == 0):
                logger.info(f"No functions loaded automatically, attempting manual load_tools()")
                try:
                    await plugin.load_tools()
                    func_count_after = len(plugin.functions) if plugin.functions else 0
                    logger.info(f"Functions after manual load_tools: {func_count_after}")
                except Exception as load_err:
                    logger.error(f"Error during manual load_tools: {load_err}", exc_info=True)
            
            return True
        except Exception as e:
            logger.warning(f"MCP connection attempt {attempt + 1} failed for user {user_id}: {e}", exc_info=True)
            if attempt == max_retries - 1:
                logger.error(f"Failed to establish MCP connection for user {user_id} after {max_retries} attempts")
                raise
            logger.debug(f"Waiting before retry for user {user_id}")
            await asyncio.sleep(1)  # Brief delay before retry
    return False


async def refresh_access_token(refresh_token: str, user_id: str) -> str:
    """Refresh access token using Microsoft identity platform token endpoint"""
    try:
        import httpx
        
        # Use Microsoft identity platform token endpoint directly
        # Get tenant ID from environment or decode it from the existing token
        tenant_id = os.getenv('TENANT_ID') or '9412b47a-813f-4d21-85a5-7772f28bf719'  # Fallback from logs
        client_id = os.getenv('AZURE_CLIENT_ID') or '4835f9b9-7b7f-433a-acd1-0545bd15b7cb'  # Frontend app client ID
        client_secret = os.getenv('MICROSOFT_PROVIDER_AUTHENTICATION_SECRET')  # Client secret from Key Vault
        
        if not client_secret:
            logger.warning(f"No client secret available for token refresh for user {user_id} - MICROSOFT_PROVIDER_AUTHENTICATION_SECRET not found")
            return None
            
        token_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        
        # Standard OAuth2 refresh token request
        data = {
            'client_id': client_id,
            'client_secret': client_secret,
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
            'scope': 'openid profile email offline_access api://1a9ee35a-715e-40a6-b14c-087f3edf7589/Mcp.Tools.ReadWrite'
        }
        
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        logger.debug(f"Attempting token refresh for user {user_id} via Microsoft identity platform")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data, headers=headers)
            response.raise_for_status()
            
            token_response = response.json()
            new_access_token = token_response.get('access_token')
            
            if new_access_token:
                logger.info(f"Successfully refreshed token for user {user_id}")
                return new_access_token
            else:
                logger.warning(f"No access token in refresh response for user {user_id}")
                return None
                
    except Exception as e:
        logger.warning(f"Token refresh failed for user {user_id}: {e}")
        return None


async def init_chat(user_token: str, user_id: str, refresh_token: str = None) -> Tuple[ChatCompletionAgent, MCPStreamableHttpPlugin]:
    global user_agents, user_plugins, user_cache_timestamps
    
    # Create cache key that includes token signature to handle token changes
    token_hash = hashlib.md5(user_token.encode()).hexdigest()[:8]  # First 8 chars of MD5
    user_key = f"{user_id}:{token_hash}"
    
    logger.debug(f"init_chat called for user: {user_id}, cache_key: {user_key}")
    
    # Check token expiration first
    token_expired = False
    try:
        import base64
        import json
        token_parts = user_token.split('.')
        if len(token_parts) >= 2:
            payload = base64.urlsafe_b64decode(token_parts[1] + '==')
            token_data = json.loads(payload)
            exp_timestamp = token_data.get('exp', 0)
            current_timestamp = time.time()
            if current_timestamp >= exp_timestamp:
                token_expired = True
                logger.warning(f"Token for user {user_id} is expired (exp: {exp_timestamp}, now: {current_timestamp})")
                
                # Attempt token refresh if refresh token is available
                if refresh_token:
                    logger.info(f"Attempting to refresh expired token for user {user_id}")
                    new_access_token = await refresh_access_token(refresh_token, user_id)
                    if new_access_token:
                        user_token = new_access_token
                        token_expired = False
                        logger.info(f"Token successfully refreshed for user {user_id}")
                        # Recalculate cache key with new token
                        token_hash = hashlib.md5(user_token.encode()).hexdigest()[:8]
                        user_key = f"{user_id}:{token_hash}"
                        logger.debug(f"Updated cache key after refresh: {user_key}")
                    else:
                        logger.warning(f"Token refresh failed for user {user_id}, will recreate with expired token")
                else:
                    logger.warning(f"No refresh token available for user {user_id}")
    except Exception as e:
        logger.warning(f"Could not check token expiration for user {user_id}: {e}")
    
    # Check if cached instances exist and are not expired (both TTL and token expiration)
    current_time = time.time()
    cache_valid = (user_key in user_agents and 
                   user_key in user_plugins and 
                   user_key in user_cache_timestamps and
                   (current_time - user_cache_timestamps[user_key]) < (CACHE_TTL_MINUTES * 60) and
                   not token_expired)
    
    if user_key in user_cache_timestamps:
        cache_age_minutes = (current_time - user_cache_timestamps[user_key]) / 60
        logger.debug(f"Cache entry for {user_key} is {cache_age_minutes:.1f} minutes old (TTL: {CACHE_TTL_MINUTES} min), token_expired: {token_expired}")
    
    if cache_valid:
        logger.debug(f"Returning cached agent for user {user_key}")
        return user_agents[user_key], user_plugins[user_key]
    
    # Clear expired cache entries
    if user_key in user_cache_timestamps:
        reason = "token expired" if token_expired else "TTL expired"
        logger.info(f"Cache invalid for user {user_key} ({reason}), recreating agent and plugin")
        user_agents.pop(user_key, None)
        user_plugins.pop(user_key, None)
        user_cache_timestamps.pop(user_key, None)
    
    logger.info(f"Creating new agent for user {user_key}")
    
    try:
        # Create fresh Azure credentials for this user
        logger.debug(f"Creating fresh Azure credentials for user {user_key}")
        # We'll use DefaultAzureCredential directly in the services - no need to extract tokens manually
        logger.debug(f"Azure credentials will be created automatically for user {user_key}")
        logger.debug(f"Creating MCP plugin for user {user_key}")
        # Create MCP plugin with user token for OBO authentication and improved settings
        headers = {"Authorization": f"Bearer {user_token}"}
        azure_plugin = MCPStreamableHttpPlugin(
            name="AzurePlugin",
            description="Azure Resources Plugin",
            load_tools=True,  # Explicitly load tools
            load_prompts=False,  # Skip prompts to avoid hanging issues
            request_timeout=120,  # Increased timeout for MCP server tool execution
            url=os.getenv('MCP_URL', 'http://localhost:5008'),
            headers=headers
        )
        logger.debug(f"MCP plugin created successfully for user {user_key}")

        # IMPORTANT: Connect and load tools BEFORE adding to kernel
        # The kernel.add_plugin() scans for @kernel_function decorated methods, which are only
        # added by load_tools() after connect(). If we add the plugin before connecting,
        # no functions will be found.
        logger.debug(f"Establishing MCP connection before adding plugin to kernel for user {user_key}")
        await ensure_mcp_connection(azure_plugin, user_key)
        
        # Wait for functions to be loaded after connection
        logger.debug(f"Waiting for MCP functions to be loaded for user {user_key}")
        max_attempts = 10
        functions_loaded = False
        for attempt in range(max_attempts):
            await asyncio.sleep(0.5)  # Wait half a second
            
            # Check for @kernel_function decorated methods (these are added by load_tools)
            tool_methods = [attr for attr in dir(azure_plugin) 
                          if not attr.startswith('_') and 
                          hasattr(getattr(azure_plugin, attr, None), '__kernel_function__')]
            
            if tool_methods:
                logger.info(f"Attempt {attempt + 1}: MCP plugin loaded {len(tool_methods)} tool methods: {tool_methods[:10]}...")
                functions_loaded = True
                break
            else:
                logger.debug(f"Attempt {attempt + 1}: No @kernel_function methods yet. Checking plugin state...")
                # The attributes exist but may not be kernel_function decorated yet
                callable_attrs = [attr for attr in dir(azure_plugin) 
                                 if not attr.startswith('_') and 
                                 callable(getattr(azure_plugin, attr, None)) and
                                 attr not in ('connect', 'close', 'load_tools', 'load_prompts', 'call_tool', 'get_mcp_client', 'get_prompt', 'message_handler', 'sampling_callback', 'logging_callback')]
                logger.debug(f"Callable attributes (potential tools): {len(callable_attrs)}")
        
        if not functions_loaded:
            logger.warning(f"MCP tools may not have loaded properly for user {user_key}")

        logger.debug(f"Creating kernel for user {user_key}")
        kernel = Kernel()
        kernel.add_filter("function_invocation", function_invocation_filter)
        
        # OpenAI has a maximum of 128 tools per request
        # Filter MCP tools to stay within this limit by removing excess tools
        MAX_TOOLS = 128
        
        # Priority tools that should always be kept (essential Azure operations)
        PRIORITY_TOOLS = {
            'subscription_list',  # Essential for listing subscriptions
            'group_list',  # Essential for listing resource groups
            'monitor_metrics_query',  # Essential for monitoring
            'monitor_workspace_log_query',  # Essential for log queries
            'storage_account_get',  # Common storage operation
            'storage_blob_get',  # Common storage operation
        }
        
        tool_methods = [attr for attr in dir(azure_plugin) 
                       if not attr.startswith('_') and 
                       hasattr(getattr(azure_plugin, attr, None), '__kernel_function__')]
        
        if len(tool_methods) > MAX_TOOLS:
            logger.warning(f"MCP plugin has {len(tool_methods)} tools, exceeds OpenAI limit of {MAX_TOOLS}")
            # Remove excess tool methods from the plugin to stay within limit
            # Keep priority tools first, then fill remaining slots alphabetically
            priority_available = PRIORITY_TOOLS & set(tool_methods)
            non_priority = sorted(set(tool_methods) - priority_available)
            remaining_slots = MAX_TOOLS - len(priority_available)
            tools_to_keep = priority_available | set(non_priority[:remaining_slots])
            tools_to_remove = set(tool_methods) - tools_to_keep
            logger.info(f"Keeping {len(priority_available)} priority tools: {sorted(priority_available)}")
            logger.info(f"Removing {len(tools_to_remove)} excess tools: {sorted(tools_to_remove)}")
            
            for tool_name in tools_to_remove:
                try:
                    delattr(azure_plugin, tool_name)
                except Exception as e:
                    logger.warning(f"Failed to remove tool {tool_name}: {e}")
            
            # Verify remaining tools
            remaining_tools = [attr for attr in dir(azure_plugin) 
                             if not attr.startswith('_') and 
                             hasattr(getattr(azure_plugin, attr, None), '__kernel_function__')]
            logger.info(f"Filtered to {len(remaining_tools)} tools (limit: {MAX_TOOLS})")
        
        # NOW add the MCP plugin to the kernel AFTER tools are loaded
        # This ensures kernel.add_plugin() can find the @kernel_function decorated methods
        logger.debug(f"Adding MCP plugin to kernel for user {user_key}")
        kernel.add_plugin(azure_plugin)
        
        # Verify functions were added to kernel
        if "AzurePlugin" in kernel.plugins:
            plugin_funcs = list(kernel.plugins["AzurePlugin"].functions.keys()) if hasattr(kernel.plugins["AzurePlugin"], 'functions') else []
            logger.info(f"Kernel plugin 'AzurePlugin' has {len(plugin_funcs)} functions: {plugin_funcs[:10]}...")
        else:
            logger.warning(f"AzurePlugin not found in kernel plugins after add_plugin()")
        
        logger.debug(f"Kernel and plugins configured for user {user_key}")

        logger.debug(f"Creating Azure OpenAI chat completion service for user {user_key}")
        # Use DefaultAzureCredential directly - it handles token management automatically
        chat_completion = AzureChatCompletion(
            deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
            endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
            credential=DefaultAzureCredential(),  # This handles token refreshing automatically
            api_version="2024-12-01-preview"
        )
        logger.debug(f"Azure OpenAI service created for user {user_key}")
        
        # Add the chat completion service to the kernel
        kernel.add_service(chat_completion)
        
        # Define comprehensive SRE instructions
        sre_instructions = """
Role: Azure Service Reliability Engineer (SRE)

You are an expert Azure SRE with direct access to Azure operations via tools.

CRITICAL: When a user asks about Azure resources, subscriptions, or any Azure data:
- IMMEDIATELY call the appropriate tool to get the data
- DO NOT ask clarifying questions
- DO NOT say you need more information
- Just call the tool and return the results

For example:
- "list my subscriptions" → call subscription_list tool immediately
- "show my resource groups" → call group_list tool immediately  
- "check storage accounts" → call storage_account_get tool immediately

You have 128 Azure tools available. Use them proactively to investigate, analyze, and take action.

Behavior Guidelines:
1. Always use available tools FIRST before responding
2. Execute Azure operations directly - never just provide instructions
3. Provide clear, actionable recommendations based on data
4. Include relevant metrics, logs, or configuration details in your responses
5. Prioritize system reliability and security in all recommendations
"""

        logger.debug(f"Creating ChatCompletionAgent for user {user_key}")
        
        agent = ChatCompletionAgent(
            service=chat_completion,
            kernel=kernel,
            name="SREAgent", 
            instructions=sre_instructions,
            function_choice_behavior=FunctionChoiceBehavior.Auto()  # Pass function choice behavior directly
        )
        logger.debug(f"ChatCompletionAgent created successfully for user {user_key}")
        
        # Verify the agent has access to functions
        if hasattr(agent, 'kernel') and hasattr(agent.kernel, 'plugins'):
            plugin_functions = []
            for plugin_name, plugin in agent.kernel.plugins.items():
                if hasattr(plugin, 'functions'):
                    plugin_functions.extend(list(plugin.functions.keys()))
            logger.info(f"Agent kernel has access to {len(plugin_functions)} total functions")
        else:
            logger.warning("Agent kernel or plugins not accessible for verification")
        
        # Cache both agent and plugin for this user with timestamp
        user_agents[user_key] = agent
        user_plugins[user_key] = azure_plugin
        user_cache_timestamps[user_key] = time.time()
        logger.info(f"Agent and plugin cached successfully for user {user_key}")
        
        return agent, azure_plugin
        
    except Exception as e:
        logger.error(f"Error creating agent for user {user_key}: {str(e)}", exc_info=True)
        raise


@app.post("/chat")
async def chat(
    user_input: str = Form(...),
    agent_thread: tuple[ChatCompletionAgent, ChatHistoryAgentThread, str] = Depends(get_agent_and_thread_dependency)
):
    agent, thread, thread_key = agent_thread
    logger.info(f"=== CHAT REQUEST START === Input: '{user_input}' for thread: {thread_key}")
    
    # Debug: Check if agent has functions available
    try:
        if hasattr(agent, 'kernel') and hasattr(agent.kernel, 'plugins'):
            total_functions = 0
            for plugin_name, plugin in agent.kernel.plugins.items():
                if hasattr(plugin, 'functions'):
                    func_count = len(plugin.functions)
                    total_functions += func_count
                    logger.info(f"Chat endpoint - Plugin '{plugin_name}' has {func_count} functions")
            logger.info(f"Chat endpoint - Agent has access to {total_functions} total functions")
            
            # Check execution settings
            if hasattr(agent, 'execution_settings'):
                logger.info(f"Agent execution settings: {agent.execution_settings}")
                if hasattr(agent.execution_settings, 'function_choice_behavior'):
                    logger.info(f"Function choice behavior: {agent.execution_settings.function_choice_behavior}")
                else:
                    logger.warning("No function_choice_behavior in agent execution_settings")
            else:
                logger.warning("Agent has no execution_settings attribute")
        else:
            logger.warning("Chat endpoint - Agent kernel or plugins not accessible")
    except Exception as debug_error:
        logger.warning(f"Debug check failed: {debug_error}")
    
    try:
        logger.debug(f"Calling agent.get_response for thread {thread_key}")
        # Get response from the agent - this automatically manages chat history
        response = await agent.get_response(message=user_input, thread=thread)
        logger.debug(f"Agent response received for thread {thread_key}")
        
        # Log the response content
        if hasattr(response, 'content'):
            # Handle ChatMessageContent object
            if hasattr(response.content, '__str__'):
                response_content = str(response.content)
            elif hasattr(response.content, 'value'):
                response_content = response.content.value
            else:
                response_content = str(response.content)
        else:
            response_content = str(response)
        
        logger.info(f"SRE Agent response length: {len(response_content)} chars for thread {thread_key}")
        logger.debug(f"SRE Agent full response: {response_content}")
        
        logger.info(f"=== CHAT REQUEST SUCCESS === for thread: {thread_key}")
        return {"response": response_content}
        
    except Exception as e:
        logger.error(f"=== CHAT REQUEST ERROR === for thread {thread_key}: {str(e)}", exc_info=True)
        return {"response": f"I encountered an error while processing your request. Please try again. Error: {str(e)}", "error": True}


@app.get("/clear-cache")
async def clear_cache():
    """Debug endpoint to clear user caches"""
    global user_agents, user_plugins, user_cache_timestamps, user_threads
    
    count_agents = len(user_agents)
    count_plugins = len(user_plugins)
    count_threads = len(user_threads)
    
    user_agents.clear()
    user_plugins.clear()
    user_cache_timestamps.clear()
    user_threads.clear()
    
    logger.info(f"Cleared caches: {count_agents} agents, {count_plugins} plugins, {count_threads} threads")
    
    return {
        "cleared": {
            "agents": count_agents,
            "plugins": count_plugins, 
            "threads": count_threads
        }
    }


@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint for debugging"""
    logger.info("=== HEALTH CHECK REQUEST ===")
    
    # Optionally validate x-ms-auth-internal-token for security
    internal_token = request.headers.get("x-ms-auth-internal-token")
    if internal_token:
        try:
            import hashlib
            import base64
            env_key = os.getenv("WEBSITE_AUTH_ENCRYPTION_KEY")
            if env_key:
                expected_hash = base64.b64encode(
                    hashlib.sha256(env_key.encode('utf-8')).digest()
                ).decode('utf-8')
                if expected_hash != internal_token:
                    logger.warning("Health check request with invalid internal token")
                    return {"status": "unauthorized"}, 401
                else:
                    logger.info("Health check request validated with internal token")
            else:
                logger.info("Health check request has internal token but no validation key available")
        except Exception as e:
            logger.warning(f"Error validating internal token: {e}")
    
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat() + "Z",
            "azure_creds": "created_per_request",
            "user_agents_count": len(user_agents),
            "user_threads_count": len(user_threads),
            "mcp_url": os.getenv('MCP_URL', 'not_set'),
            "openai_endpoint": os.getenv('AZURE_OPENAI_ENDPOINT', 'not_set')[:50] + "..." if os.getenv('AZURE_OPENAI_ENDPOINT') else 'not_set',
            "authenticated_request": internal_token is not None
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
async def test_agent_creation(
    x_ms_client_principal_id: str = Header(None, alias="x-ms-client-principal-id"),
    x_ms_token_aad_access_token: str = Header(None, alias="x-ms-token-aad-access-token")
):
    """Test endpoint to check if agent can be created"""
    logger.info("=== AGENT TEST REQUEST ===")
    
    # Use real authentication if available, otherwise fallback to dummy values
    user_token = x_ms_token_aad_access_token if x_ms_token_aad_access_token else "dummy_token"
    user_id = x_ms_client_principal_id if x_ms_client_principal_id else "test_user"
    
    logger.info(f"Testing agent creation for user: {user_id} (real_auth: {bool(x_ms_client_principal_id)})")
    
    try:
        agent, plugin = await init_chat(user_token, user_id)
        logger.info("Agent created successfully!")
        
        return {
            "status": "success",
            "agent_name": agent.name,
            "plugin_name": plugin.name,
            "user_id": user_id,
            "using_real_auth": bool(x_ms_client_principal_id),
            "cached_agents": len(user_agents)
        }
    except Exception as e:
        logger.error(f"Agent creation failed: {str(e)}", exc_info=True)
        return {
            "status": "error", 
            "error": str(e),
            "error_type": type(e).__name__,
            "user_id": user_id,
            "using_real_auth": bool(x_ms_client_principal_id)
        }


@app.get("/debug/test-mcp")
async def test_mcp_connection(
    x_ms_client_principal_id: str = Header(None, alias="x-ms-client-principal-id"),
    x_ms_token_aad_access_token: str = Header(None, alias="x-ms-token-aad-access-token"),
    x_ms_token_aad_refresh_token: str = Header(None, alias="x-ms-token-aad-refresh-token")
):
    """Test endpoint to check MCP server connection and functionality"""
    logger.info("=== MCP TEST REQUEST ===")
    
    # Use real authentication if available, otherwise fallback to dummy values
    user_token = x_ms_token_aad_access_token if x_ms_token_aad_access_token else "dummy_token"
    user_id = x_ms_client_principal_id if x_ms_client_principal_id else "test_mcp_user"
    
    logger.info(f"Testing MCP connection for user: {user_id} (real_auth: {bool(x_ms_client_principal_id)})")
    
    # Debug: Let's decode the token to see what audience we're getting
    if x_ms_token_aad_access_token:
        import base64
        import json
        try:
            # Decode JWT payload (middle part)
            token_parts = user_token.split('.')
            if len(token_parts) >= 2:
                payload = base64.urlsafe_b64decode(token_parts[1] + '==')  # Add padding
                token_data = json.loads(payload)
                logger.info(f"Token issuer: '{token_data.get('iss', 'NOT_FOUND')}'")
                logger.info(f"Token audience: '{token_data.get('aud', 'NOT_FOUND')}'")
                logger.info(f"Token scope: '{token_data.get('scp', 'NOT_FOUND')}'")
        except Exception as e:
            logger.warning(f"Could not decode token: {e}")
    
    try:
        # Create agent and plugin
        agent, plugin = await init_chat(user_token, user_id, x_ms_token_aad_refresh_token)
        logger.info(f"Agent and plugin created: {agent.name}, {plugin.name}")
        
        # Test MCP connection
        logger.info("Testing MCP connection...")
        await ensure_mcp_connection(plugin, user_id)
        logger.info("MCP connection established successfully")
        
        # Try to get available tools from MCP server
        try:
            logger.info("Fetching available MCP tools...")
            # Get the kernel from the agent
            kernel = agent.kernel if hasattr(agent, 'kernel') else None
            if kernel:
                plugins = kernel.plugins
                logger.info(f"Available plugins: {list(plugins.keys())}")
                mcp_functions = []
                
                # Wait a bit for MCP tools to be loaded if needed
                max_wait_attempts = 5
                wait_delay = 0.5
                
                for plugin_name, plugin_obj in plugins.items():
                    logger.info(f"Checking plugin: {plugin_name}")
                    if plugin_name.lower() == "azureplugin":
                        # Try multiple times to get functions as they may load asynchronously
                        for attempt in range(max_wait_attempts):
                            functions = list(plugin_obj.functions.keys()) if hasattr(plugin_obj, 'functions') else []
                            if functions:
                                break
                            logger.debug(f"Attempt {attempt + 1}: No functions found yet, waiting {wait_delay}s...")
                            await asyncio.sleep(wait_delay)
                        
                        mcp_functions.extend(functions)
                        logger.info(f"Found {len(functions)} functions in MCP plugin: {functions}")
                        
                        # Also log some details about the plugin
                        logger.debug(f"Plugin {plugin_name} type: {type(plugin_obj)}")
                        logger.debug(f"Plugin {plugin_name} hasattr functions: {hasattr(plugin_obj, 'functions')}")
                        if hasattr(plugin_obj, 'functions'):
                            logger.debug(f"Plugin {plugin_name} functions type: {type(plugin_obj.functions)}")
                
                return {
                    "status": "success",
                    "agent_name": agent.name,
                    "plugin_name": plugin.name,
                    "user_id": user_id,
                    "using_real_auth": bool(x_ms_client_principal_id),
                    "mcp_url": os.getenv('MCP_URL', 'not_set'),
                    "mcp_connected": True,
                    "available_functions": mcp_functions[:10],  # Limit to first 10 for readability
                    "total_function_count": len(mcp_functions),
                    "cached_agents": len(user_agents),
                    "plugin_session_active": plugin.session is not None,
                    "plugin_load_tools_flag": plugin.load_tools_flag if hasattr(plugin, 'load_tools_flag') else "unknown",
                    "plugin_attributes": [attr for attr in dir(plugin) if not attr.startswith('_')][:20]
                }
            else:
                logger.warning("Could not access kernel from agent")
                return {
                    "status": "partial_success",
                    "agent_name": agent.name,
                    "plugin_name": plugin.name,
                    "user_id": user_id,
                    "using_real_auth": bool(x_ms_client_principal_id),
                    "mcp_url": os.getenv('MCP_URL', 'not_set'),
                    "mcp_connected": True,
                    "warning": "Could not enumerate functions",
                    "cached_agents": len(user_agents)
                }
                
        except Exception as func_error:
            logger.warning(f"Could not enumerate MCP functions: {func_error}")
            return {
                "status": "partial_success",
                "agent_name": agent.name,
                "plugin_name": plugin.name,
                "user_id": user_id,
                "using_real_auth": bool(x_ms_client_principal_id),
                "mcp_url": os.getenv('MCP_URL', 'not_set'),
                "mcp_connected": True,
                "warning": f"Function enumeration failed: {str(func_error)}",
                "cached_agents": len(user_agents)
            }
            
    except Exception as e:
        logger.error(f"MCP test failed: {str(e)}", exc_info=True)
        return {
            "status": "error", 
            "error": str(e),
            "error_type": type(e).__name__,
            "user_id": user_id,
            "using_real_auth": bool(x_ms_client_principal_id),
            "mcp_url": os.getenv('MCP_URL', 'not_set'),
            "mcp_connected": False
        }


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
    print("=== STARTING UVICORN DIRECTLY ===")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")
