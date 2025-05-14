from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.models.openai import OpenAIChat
#from agno.playground import Playground, serve_playground_app
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.vectordb.pgvector import PgVector, SearchType
# Reranker
from agno.reranker.cohere import CohereReranker

# Chunking strategy
from agno.document.chunking.agentic import AgenticChunking
from agno.document.chunking.document import DocumentChunking
from agno.document.chunking.recursive import RecursiveChunking
#from agno.document.chunking.semantic import SemanticChunking
from agno.document.chunking.fixed import FixedSizeChunking


# Transform this into a RAG agent that uses a knowledge base of PDFs
# Use Gemini as the model
# Use PgVector as the vector database
# Use Ollama for embeddings
from agno.models.google import Gemini
from agno.embedder.google import GeminiEmbedder
from agno.embedder.ollama import OllamaEmbedder
from agno.models.ollama import Ollama

# CombinedKnowledge Base::
from agno.knowledge.combined import CombinedKnowledgeBase
from agno.knowledge.csv import CSVKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.website import WebsiteKnowledgeBase

# for user memory
from agno.agent import AgentMemory
from agno.memory.db.sqlite import SqliteMemoryDb
from agno.storage.agent.sqlite import SqliteAgentStorage
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.prompt import Prompt
from agno.memory.db.postgres import PgMemoryDb
from agno.tools.duckduckgo import DuckDuckGoTools

#from agno.models.deepseek import DeepSeek

from textwrap import dedent
from pathlib import Path

### For ui
from typing import List
from typing import Optional
import re
import debugpy
import streamlit as st
from PIL import Image


db_url = "postgresql+psycopg://ai:ai@rag_database:5432/ai"


def get_rag_assistant(
    model_id: str = None,
    llm_model: str = "gemini-2.0-flash",
    embeddings_model: str = "text-embedding-004",
    instructions: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
    user: str = "user",
) -> Agent:
    """Get a Local RAG Assistant."""

    # # Initialize storage for both agent sessions and memories
    # agent_storage = SqliteAgentStorage(
    #     table_name="agent_memories", db_file="tmp/agents.db"
    # )

    # existing_sessions = agent_storage.get_all_session_ids(user)
    # if len(existing_sessions) > 0:
    #     session_id = existing_sessions[0]

    pdf_kb=PDFKnowledgeBase( # Define the knowledge base using PDFs
        path = Path("pdf_src"),
        # Use PgVector as the vector database and store embeddings in the `ai.recipes` table
        vector_db=PgVector(
            table_name="pdf_documents",
            db_url=db_url,
        ),
    )

    pdf_url_kb = PDFUrlKnowledgeBase( # Create PDF URL knowledge base
        urls=["https://agno-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
        vector_db=PgVector(
            table_name="pdf_documents",
            db_url=db_url,
        ),
    )

    csv_kb = CSVKnowledgeBase(# Create CSV knowledge base
        path=Path("csv_src"),
        vector_db=PgVector(
            table_name="csv_documents",
            db_url=db_url,
        ),
    )

    website_kb = WebsiteKnowledgeBase(# Create Website knowledge base
        urls=["https://www.liverpool.ac.uk/courses/digital-media-data-and-society-ma",
                "https://www.liverpool.ac.uk/study/postgraduate-research/degrees/computer-science/",
                "https://online.liverpool.ac.uk/programmes/msc-artificial-intelligence/",
                "https://online.liverpool.ac.uk/programmes/msc-big-data-analytics/",
                "https://online.liverpool.ac.uk/programmes/msc-information-systems-management/",
                "https://www.liverpool.ac.uk/courses/cyber-security-msc-online",
                "https://online.liverpool.ac.uk/programmes/msc-computer-science/",
                "https://www.liverpool.ac.uk/courses/computer-science-bsc-hons-algorithms-and-optimisation-pathway-2",
                "https://www.liverpool.ac.uk/courses/computer-science-bsc-hons-artificial-intelligence-pathway-2",
                "https://www.liverpool.ac.uk/courses/computer-science-bsc-hons-cyber-security-pathway-2",
                "https://www.liverpool.ac.uk/courses/computer-science-with-software-development-bsc-hons",
            ],
        max_links=10,
        vector_db=PgVector(
            table_name="website_documents",
            db_url=db_url,
        ),
    )

    knowledge_base = CombinedKnowledgeBase(
        sources=[
            csv_kb,
            #pdf_url_kb,
            #website_kb,
            pdf_kb,
        ],
        vector_db=PgVector(
            table_name="combined_documents",
            db_url=db_url,
            search_type=SearchType.hybrid,
            #embedder=GeminiEmbedder(id="text-embedding-004", dimensions=768),
            #embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
            embedder=OpenAIEmbedder(id="text-embedding-3-small"),
            #reranker=CohereReranker(model="rerank-multilingual-v3.0"),
        ),
        #chunking_strategy=AgenticChunking(),
        chunking_strategy=DocumentChunking(overlap=30),
        num_documents=40,
    )


    return Agent(
        name="RAG Agent",
        agent_id="rag-agent",
        model=OpenAIChat(id="gpt-4o"),
        #model=OpenAIChat(id="o3-mini"),
        session_id=session_id,
        reasoning=True,
        #model=Gemini(id="gemini-2.0-flash"),
        # model=Ollama(id="llama3.2"),
        knowledge=knowledge_base,
        # Add a tool to read chat history.
        memory=AgentMemory(
            db=PgMemoryDb(table_name="agent_memory", db_url=db_url), create_user_memories=True, create_session_summary=True
        ),
        # Store the agent sessions in the `ai.rag_agent_sessions` table
        storage=PostgresAgentStorage(table_name="agentic_rag_agent_sessions", db_url=db_url),
        description=dedent("""\
            You are a helpful and knowledgeable University Major Selection Agent called 'Agentic RAG' and your goal is to assist the user in the best way possible.
            Think of yourself as a combination of an academic advisor, a career counselor, a major advisor, and a student success advocate,and a university information specialist.
            \
            """),
        instructions=[
            "   - Normalize vague or short queries like 'I want to study AI' into specific, well-defined search prompts.",
            "   - Always assume that vague educational queries refer to Liverpool University unless stated otherwise.",
            "   - Examples of query expansion:",
            "       - 'I want to learn AI' → 'Show me all AI-related programs at University of Liverpool'",
            "       - 'I want to learn cyberSecurity' → 'Show me all cyberSecurity-related programs at Liverpool University'",
            "1. Intent Analysis & Contextual Understanding:",
            "   - **ALWAYS** begin by analyzing the user's current query within the context of the entire conversation history.",
            "   - Use the `get_chat_history` tool to understand the user's previous questions, preferences, and any prior clarifications",
            "   - Identify the core intent of the user's query and any implicit or explicit references to previous topics",
            "   - **Pay special attention to follow-up questions that refer to previously mentioned programs or majors.**",
            "   - **Ensure that any information provided is directly related to the specific programs or majors the user has expressed interest in.**",
            "   - Based on this analysis, formulate a clear understanding of what information the user is truly seeking",
            "2. Knowledge Base Search:",
            "   - **After intent analysis**, search the knowledge base using the `search_knowledge_base` tool",
            "   - Analyze ALL returned documents thoroughly before responding",
            "   - If multiple documents are returned, synthesize the information coherently",
            "3. Context Management:",
            "   - Use get_chat_history tool and refer previous interactions to maintain conversation continuity",
            "   - Keep track of user preferences and prior clarifications",
            "4. Response Quality:",
            "   - Provide specific citations and sources for claims",
            "   - Structure responses with clear sections and bullet points when appropriate",
            "   - Include relevant quotes from source materials",
            "   - Avoid hedging phrases like 'based on my knowledge' or 'depending on the information'",
            "   - **When providing tuition fee information, ensure it is specific to the programs the user has inquired about.**"
            "   - **When providing tuition fee information, ALWAYS search the knowledge base using the `search_knowledge_base` tool to make sure**"
            "   - **If possible, provide a breakdown of tuition fees for each program separately.**",
            "   For major recommendations, include:",
            "   - Major name, number, and description, and modules",
            "   - Available sections and scheduling information.",
            "   - Tips for success and common challenges.",
            "   - List of prerequisites and recommended background.",
            "5. User Interaction:",
            "   - Ask for clarification if the query is ambiguous",
            "   - Break down complex questions into manageable parts",
            "   - Proactively suggest related topics or follow-up questions",
            "6. Error Handling:",
            "   - If no relevant information is found, clearly state this",
            "   - Suggest alternative approaches or questions",
            "   - Be transparent about limitations in available information",
            "7. Endings:",
            "   - End each response with an uplifting sign-off like:",
            "   - 'Happy studying! May your academic journey be successful!'",
            "   - 'May your major selections lead to academic excellence!'",
            "   - 'Enjoy your university learning experience!'",
        ],
        # Add a tool to search the knowledge base which enables agentic RAG.
        # This is enabled by default when `knowledge` is provided to the Agent.
        search_knowledge=True,# This setting gives the model a tool to search the knowledge base for information
        read_chat_history=True,# This setting gives the model a tool to get chat history
        markdown=True, # This setting tells the model to format messages in markdown
        show_tool_calls=True,
        read_tool_call_history=True,
        add_history_to_messages=True,
        debug_mode=True,
        num_history_responses=10,
        #add_references=True,
    )

if __name__ == "__main__":
    test_agent = get_rag_assistant()
    test_agent.knowledge.load(recreate=False)
    test_agent.print_response("推荐一些Liverpool大学的计算机科学专科课程", markdown=True)
    test_agent.print_response(
    "What courses did you learn in each year of Computer Science and Electronic Engineering with a Year in Industry?", stream=True)
