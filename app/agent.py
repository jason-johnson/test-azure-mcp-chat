import os
import logging
from uuid import uuid4
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents import AuthorRole
from semantic_kernel.contents.annotation_content import AnnotationContent
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from azure.identity.aio import DefaultAzureCredential

from semantic_kernel.agents import AzureAIAgent, AzureAIAgentThread
from azure.ai.projects.aio import AIProjectClient
from azure.ai.agents.models import BingGroundingTool
from semantic_kernel import Kernel
from semantic_kernel.connectors.mcp import MCPSsePlugin, MCPStdioPlugin

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for Azure resources
azure_creds = None
azure_mcp = None
agent = None
azure_plugin = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global azure_creds, azure_ai_client, bing_grounding, search_agent, azure_mcp
    logger.info("Initializing Azure credentials and client...")
    try:
        
        logger.info("Creating Azure credentials...")
        azure_creds = DefaultAzureCredential()
                        
        await init_chat()

    except Exception as e:
        logger.error(f"Failed to initialize Azure resources: {e}")
        logger.error(f"Search functionality will be unavailable. Error details: {type(e).__name__}: {str(e)}")
        # Set search_agent to None explicitly to make it clear it failed
        search_agent = None

    
    yield
      # Shutdown       
    logger.info("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Maintain chat history per context
chat_history_store: dict[str, ChatHistory] = {}



async def init_chat():
    # 1. Create the agent
    #async with MCPStdioPlugin(
    #    name="AzurePlugin",
    #    description="Azure Resources Plugin",
    #    command="npx",
    #    load_tools=True,
    #    args=["-y", "@azure/mcp@latest", "server", "start"]
    #) as azure_plugin:

    global agent, azure_plugin
    azure_plugin =  MCPSsePlugin(name="AzurePlugin",
                        description="Azure Resources Plugin",
                        load_prompts=False,
                        url=os.getenv('MCP_URL', 'http://localhost:5008/sse'))
    #azure_plugin = MCPStdioPlugin(
    #   name="AzurePlugin",
    #    description="Azure Resources Plugin",
    #    command="npx",
    #    load_tools=True,
    #    args=["-y", "@azure/mcp@latest", "server", "start"])
    await azure_plugin.connect()
    kernel = Kernel()
    kernel.add_plugin(azure_plugin)
        # Add Azure OpenAI chat completion
    chat_completion = AzureChatCompletion(
        api_key=os.getenv('AZURE_OPENAI_API_KEY'),
        endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
        deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
        api_version="2024-12-01-preview",
    )
    kernel.add_service(chat_completion)
    

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


@app.post("/chat")
async def chat(user_input: str = Form(...), context_id: str = Form("default")):
    logger.info(f"Received chat request: {user_input} with context ID: {context_id}")
    await azure_plugin.connect()
    # Get or create ChatHistory for the context
    chat_history = chat_history_store.get(context_id)
    if chat_history is None:
        chat_history = ChatHistory(
            messages=[],
            system_message="You are a helpful assistant.",
        )
        chat_history_store[context_id] = chat_history
        logger.info(f"Created new ChatHistory for context ID: {context_id}")    # Add user input to chat history
    chat_history.messages.append(ChatMessageContent(role=AuthorRole.USER, content=user_input))

    # Create a new thread from the chat history
    thread = ChatHistoryAgentThread(chat_history=chat_history, thread_id=str(uuid4()))

    # Get response from the agent
    response = await agent.get_response(message=user_input, thread=thread)    # Add assistant response to chat history
    chat_history.messages.append(ChatMessageContent(role=AuthorRole.ASSISTANT, content=response.content.content))

    logger.info(f"response: {response.content.content}")

    return {"response": response.content.content}   



@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        logger.error("index.html file not found. Please ensure it exists in the current directory.")
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)