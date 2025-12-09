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

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
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
    global azure_creds, azure_ai_client, bing_grounding, search_agent, azure_mcp
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

# Maintain chat history per context
chat_history_store: dict[str, ChatHistory] = {}

# Function invocation filter to log function calls and responses


async def function_invocation_filter(context: FunctionInvocationContext, next):
    """A filter that will be called for each function call in the response."""
    if "messages" not in context.arguments:
        await next(context)
        return
    print(
        f"    Agent [{context.function.name}] called with messages: {context.arguments['messages']}")
    await next(context)
    print(
        f"    Response from agent [{context.function.name}]: {context.result.value}")


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

    # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        api_version="2024-12-01-preview",
        azure_credential=azure_creds
    )

    agent = ChatCompletionAgent(
        kernel=kernel,
        service=chat_completion,
        name="SREAgent",
        plugins=[azure_plugin],
        instructions="""
    Role Definition:

    You are acting as a Azure Service Reliability Engineer (SRE).
    You provide help to solve and prevent incidents and outages.
    
    You react into monitoring alerts, or questions from the SRE team.
    Use tools to investigate incidents, gather data, and provide answers.

    If use asks about azure related tasks, intention is that you execute them and not give users instructions to do so (unless specially asked)

    """)
    
    # Cache the agent and plugin for this user
    user_agents[user_key] = agent
    user_plugins[user_key] = azure_plugin
    
    return agent, azure_plugin


@app.post("/chat")
async def chat(
    user_input: str = Form(...), 
    context_id: str = Form("default"),
    x_ms_client_principal_id: str = Header(..., alias="x-ms-client-principal-id"),
    x_ms_token_aad_access_token: str = Header(..., alias="x-ms-token-aad-access-token")
):
    logger.info(
        f"Received chat request: {user_input} with context ID: {context_id} for user: {x_ms_client_principal_id}")
    
    # Get or create agent for this user (singleton pattern using Azure user ID)
    agent, azure_plugin = await init_chat(x_ms_token_aad_access_token, x_ms_client_principal_id)
    
    # Ensure connection is active
    await azure_plugin.connect()
    # Get or create ChatHistory for the context
    chat_history = chat_history_store.get(context_id)
    if chat_history is None:
        chat_history = ChatHistory(
            messages=[],
            system_message="You are a helpful assistant.",
        )
        chat_history_store[context_id] = chat_history
        # Add user input to chat history
        logger.info(f"Created new ChatHistory for context ID: {context_id}")
    chat_history.messages.append(ChatMessageContent(
        role=AuthorRole.USER, content=user_input))

    # Create a new thread from the chat history
    thread = ChatHistoryAgentThread(
        chat_history=chat_history, thread_id=str(uuid4()))

    # Get response from the agent
    # Add assistant response to chat history
    response = await agent.get_response(message=user_input, thread=thread)
    chat_history.messages.append(ChatMessageContent(
        role=AuthorRole.ASSISTANT, content=response.content.content))

    logger.info(f"response: {response.content.content}")

    return {"response": response.content.content}


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
