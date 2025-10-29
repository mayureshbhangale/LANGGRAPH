from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from datetime import datetime
from langchain.tools import Tool

def get_today(_: str) -> str:
    return datetime.today().strftime("%B %d, %Y")


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

search = TavilySearchAPIWrapper()
search_tool = Tool.from_function(
    name="Search",
    func=search.results,
    description="Use this tool to search the web for current events, recent information, or factual queries. Input should be a search query."
)
date_tool = Tool.from_function(
    name="GetCurrentDate",
    func=get_today,
    description="Use this to get today's date."
)
tools = [date_tool, search_tool]

agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True,
    handle_parsing_errors=True
)


response = agent.invoke("when was SpaceX's last launch and how many days was that from this instant")
#print(response)