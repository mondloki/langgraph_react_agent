from dotenv import load_dotenv

from langchain import hub
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent

load_dotenv()

react_prompt :  PromptTemplate =  hub.pull("hwchase17/react")

@tool
def triple(num: float):
    """ used to multiply the given number by 3"""

    return 3 * float(num)

tools = [TavilySearchResults(max_results=1), triple]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

react_agent_runnable = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)




