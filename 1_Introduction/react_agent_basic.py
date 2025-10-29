from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.tools import Tool
from langchain.utilities.serpapi import SerpAPIWrapper

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

search = SerpAPIWrapper()
search_tool = Tool.from_function(
    name="Search",
    func=search.run,
    description="Useful for answering questions about current events or factual queries."
)

tools = [search_tool]

agent = initialize_agent(
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    tools=tools,
    llm=llm,
    verbose=True
)

response = agent.invoke("Give a tweet about today's weather in New York City.")
print(response)