import os
import tempfile
from typing import List

import nest_asyncio
import requests
import streamlit as st
from rag_agent import get_rag_assistant
from agno.agent import Agent
from agno.document import Document
from agno.document.reader.csv_reader import CSVReader
from agno.document.reader.pdf_reader import PDFReader
from agno.document.reader.text_reader import TextReader
from agno.document.reader.website_reader import WebsiteReader
from agno.utils.log import logger
from utils import (
    CUSTOM_CSS,
    add_message,
    display_tool_calls,
    export_chat_history,
)

nest_asyncio.apply()
st.set_page_config(
    page_title="Agentic RAG",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom CSS

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["agentic_rag_agent"] = None
    st.session_state["agentic_rag_agent_session_id"] = None
    st.session_state["messages"] = []
    st.rerun()


def get_reader(file_type: str):
    """Return appropriate reader based on file type."""
    readers = {
        "pdf": PDFReader(),
        "csv": CSVReader(),
        "txt": TextReader(),
    }
    return readers.get(file_type.lower(), None)


def initialize_agent(model_id: str):
    """Initialize or retrieve the Agentic RAG."""
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
    ):
        logger.info(f"---*--- Creating {model_id} Agent ---*---")
        agent: Agent = get_rag_assistant(
            model_id=model_id,
            session_id=st.session_state.get("agentic_rag_agent_session_id"),
        )
        st.session_state["agentic_rag_agent"] = agent
        st.session_state["agentic_rag_agent_session_id"] = agent.session_id
    return st.session_state["agentic_rag_agent"]

def set_page_config():
    st.image("asset/liverpool_back.jpg", width=1000)  # Adjust the width as needed

import base64

@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file, background_height_px=150):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-position: top center;
        background-size: 75vw 20vh;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return



def main():
    ####################################################################
    # App header
    ####################################################################
    #set_page_config()
    #set_png_as_page_bg('asset/liverpool_logo.jpg')
    st.title("University of Liverpool Assistant ")
    st.subheader("Your AI Guide for Majors & Academic Success. How can I assist you today?")
    #st.sidebar.image('asset/logo.jpg')

    ####################################################################
    # Model selector
    ####################################################################
    model_options = {
        "o3-mini": "openai:o3-mini",
        "gpt-4o": "openai:gpt-4o",
        "gemini-2.0-flash-exp": "google:gemini-2.0-flash-exp",
        #"claude-3-5-sonnet": "anthropic:claude-3-5-sonnet-20241022",
        "llama-3.3-70b": "groq:llama-3.3-70b-versatile",
    }
    selected_model = st.sidebar.selectbox(
        "Select a model",
        options=list(model_options.keys()),
        index=0,
        key="model_selector",
    )
    model_id = model_options[selected_model]

    ####################################################################
    # Initialize Agent
    ####################################################################
    agentic_rag_agent: Agent
    if (
        "agentic_rag_agent" not in st.session_state
        or st.session_state["agentic_rag_agent"] is None
        or st.session_state.get("current_model") != model_id
    ):
        logger.info("---*--- Creating new Agentic RAG  ---*---")
        agentic_rag_agent = get_rag_assistant(model_id=model_id)
        agentic_rag_agent.knowledge.load(recreate=True)#re create db
        st.session_state["agentic_rag_agent"] = agentic_rag_agent
        st.session_state["current_model"] = model_id
    else:
        agentic_rag_agent = st.session_state["agentic_rag_agent"]

    ####################################################################
    # Load Agent Session from the database
    ####################################################################
    try:
        st.session_state["agentic_rag_agent_session_id"] = (
            agentic_rag_agent.load_session()
        )
    except Exception:
        st.warning("Could not create Agent session, is the database running?")
        return

    ####################################################################
    # Load runs from memory
    ####################################################################
    agent_runs = agentic_rag_agent.memory.runs
    if len(agent_runs) > 0:
        logger.debug("Loading run history")
        st.session_state["messages"] = []
        for _run in agent_runs:
            if _run.message is not None:
                add_message(_run.message.role, _run.message.content)
            if _run.response is not None:
                add_message("assistant", _run.response.content, _run.response.tools)
    else:
        logger.debug("No run history found")
        st.session_state["messages"] = []

    if prompt := st.chat_input("👋 Ask me anything!"):
        add_message("user", prompt)

    ###############################################################
    # Sample Question
    ###############################################################
    st.sidebar.markdown("#### ❓ Sample Questions")
    if st.sidebar.button("📝 Summarize"):
        add_message(
            "user",
            "Can you summarize what is currently in the knowledge base (use `search_knowledge_base` tool)?",
        )

    ###############################################################
    # Utility buttons
    ###############################################################
    st.sidebar.markdown("#### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])  # Equal width columns
    with col1:
        if st.sidebar.button(
            "🔄 New Chat", use_container_width=True
        ):  # Added use_container_width
            restart_agent()
    with col2:
        if st.sidebar.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="rag_chat_history.md",
            mime="text/markdown",
            use_container_width=True,  # Added use_container_width
        ):
            st.sidebar.success("Chat history exported!")

    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
                    # Display tool calls if they exist in the message
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)


    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🤔 Thinking..."):
                response = ""
                try:
                    # Run the agent and stream the response
                    run_response = agentic_rag_agent.run(question, stream=True)
                    for _resp_chunk in run_response:
                        # Display tool calls if available
                        if _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        # Display response
                        if _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    add_message(
                        "assistant", response, agentic_rag_agent.run_response.tools
                    )
                except Exception as e:
                    error_message = f"Sorry, I encountered an error: {str(e)}"
                    add_message("assistant", error_message)
                    st.error(error_message)

    ####################################################################
    # Session selector
    ####################################################################
    #session_selector_widget(agentic_rag_agent, model_id)
    #rename_session_widget(agentic_rag_agent)

    ####################################################################
    # About section
    ####################################################################
    #about_widget()

    st.html(
        """
    <style>
        .stChatMessage:has(.chat-assistant) {
            flex-direction: row-reverse;
            text-align: right;
        }
    </style>
    """
    )

    # CSS = """
    # .stChatMessage:has([data-testid="stChatMessageAvatarUser"]) {
    #     display: flex;
    #     flex-direction: row-reverse;
    #     align-itmes: end;
    # }

    # [data-testid="stChatMessageAvatarUser"] + [data-testid="stChatMessageContent"] {
    #     text-align: right;
    # }
    # """
    # st.html(f"<style>{CSS}</style>")

main()