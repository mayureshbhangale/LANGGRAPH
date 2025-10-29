from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent_executor
from langchain_community.tools import TavilySearchResults

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-pro")

search_tool = TavilySearchResults(search_depth="basic")
tools = [search_tool]

agent = initialize_agent_executor(
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True
)

response = agent.invoke("Give a tweet about today's weather in New York City.")
print(response)