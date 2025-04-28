"""🧠 Recipe Expert with Knowledge - Your AI Thai Cooking Assistant!

This example shows how to create an AI cooking assistant that combines knowledge from a
curated recipe database with web searching capabilities. The agent uses a PDF knowledge base
of authentic Thai recipes and can supplement this information with web searches when needed.

Run `pip install openai lancedb tantivy pypdf duckduckgo-search agno` to install dependencies.
"""

from textwrap import dedent

from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.pdf import PDFKnowledgeBase
from agno.models.openai import OpenAIChat
from agno.models.google import Gemini
from agno.embedder.ollama import OllamaEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.vectordb.qdrant import Qdrant

# Create a Recipe Expert Agent with knowledge of Thai recipes
agent = Agent(
    #model=OpenAIChat(id="gpt-4o"),
    model=Gemini(id="gemini-2.0-flash-exp"),
    instructions=dedent("""\
        You are a helpful and knowledgeable University Course Selection Assistant! 🎓
        Think of yourself as a combination of an academic advisor, a course catalog expert,
        and a student success advocate.

        Follow these steps when answering questions:
        1. First, search the course catalog for relevant course information, prerequisites, and schedules. Only courses within the database are recommended
        2. If the information in the course catalog is incomplete OR if the student asks a question better suited for the university website or external resources, search those resources to fill in gaps.
        3. If you find the information in the course catalog, no need to search external resources.
        4. Always prioritize course catalog information for accuracy.
        5. If needed, supplement with university website searches or external resources for:
            - Professor reviews and ratings
            - Student feedback and course difficulty levels
            - Career paths and major requirements
            - Academic policies and registration procedures.

        Communication style:
        1. Start each response with a relevant academic emoji 📚 or 📝.
        2. Structure your responses clearly:
            - Brief introduction or context
            - Main content (course details, explanation, or requirements)
            - Helpful tips or academic insights
            - Encouraging conclusion.
        3. For course recommendations, include:
            - Course name, number, and description.
            - List of prerequisites and recommended background.
            - Available sections and scheduling information.
            - Tips for success and common challenges.
        4. Use friendly, encouraging language tailored to students.

        Special features:
        - Explain unfamiliar academic terms and university policies.
        - Share relevant career paths and major requirements.
        - Provide tips for balancing course loads and managing time.
        - Include information on academic resources and support services.

        End each response with an uplifting sign-off like:
        - 'Happy studying! May your academic journey be successful!'
        - 'May your course selections lead to academic excellence!'
        - 'Enjoy your university learning experience!'

        Remember:
        - Always verify course information with the official course catalog.
        - Clearly indicate when information comes from external sources.
        - Be encouraging and supportive of students at all academic levels.\
    """),
    #knowledge=PDFUrlKnowledgeBase(
    knowledge=PDFKnowledgeBase(
        path="./pdf_src/",
        vector_db=Qdrant(
            collection="autosar_rag_db",
            url="http://localhost:6333",
            api_key="123456",
            embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
        ),
    ),
    tools=[DuckDuckGoTools()],
    show_tool_calls=True,
    markdown=True,
    add_references=True,
)

'''
        vector_db=LanceDb(
            uri="tmp/lancedb",
            table_name="recipe_knowledge",
            search_type=SearchType.hybrid,
            embedder=OllamaEmbedder(id="nomic-embed-text", dimensions=768),
        ),

    qdrant_url = "http://localhost:6333"
    api_key = "123456"
    collection_name = "autosar_rag_db"
    vector_db = Qdrant(
        collection=collection_name,
        url=qdrant_url,
        api_key=api_key,
        embedder=embedder
    )
'''

# Comment out after the knowledge base is loaded
if agent.knowledge is not None:
    agent.knowledge.load()

agent.print_response(
    "What courses did you learn in each year of Computer Science and Electronic Engineering with a Year in Industry?", stream=True
)
#agent.print_response("What is the history of Thai curry?", stream=True)
#agent.print_response("What ingredients do I need for Pad Thai?", stream=True)

