import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper ## wrappers for langchain inbuilt tools
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun, GoogleSearchRun ## here are the inbuilt tools
from langchain_classic.agents import initialize_agent, AgentType
from langchain_classic.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()

## to run inbuilt tools first we need to define some customization using wrapper and then we will initialize the query runner
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wikipedia = WikipediaQueryRun(api_wrapper=wiki_wrapper)
## for searching we will use google
search = DuckDuckGoSearchRun(name="search")

st.title("Langchain - Chat With Search")
st.sidebar.title("settings")
api_key = st.sidebar.text_input("Enter your groq api key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "hey there"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="what is machine learning?"):
    st.session_state.messages.append({"role": "user", "content":prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-8b-instant", streaming=True)
    tools = [arxiv, search, wikipedia]
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors=True) ## agent types are just used to define that with which type the agent will manage the history and the memory

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False) ## it's a callback handler which will just show agent updates and thoughts in the container everytime
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant", "content":response})
        st.write(response)
