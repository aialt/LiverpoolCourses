from agno.playground import Playground, serve_playground_app
from rag_agent import get_rag_assistant

test_agent = get_rag_assistant()
app = Playground(agents=[test_agent]).get_app()


if __name__ == "__main__":
    # Load the knowledge base: Comment after first run as the knowledge base is already loaded
    #test_agent.knowledge.load(upsert=True)
    test_agent.knowledge.load(skip_existing=True)

    serve_playground_app("app_ui:app", reload=True)