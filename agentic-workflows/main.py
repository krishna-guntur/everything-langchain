from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.tools import tool

from tavily import TavilyClient

import logging

logging.basicConfig(
    filename=r'logs\agentic-workflows.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)

tavily = TavilyClient()


@tool
def get_weather(query: str) -> str:

    """
    Searches the web for current information, such as the weather in a specific location.
    Use this tool when you need real-time facts outside your training data.

    Args:
        query (str): The specific search query to execute. It should be highly 
                     targeted, for example: "weather in Singapore" or "top restaurants in Hyderabad".

    Returns:
        str: The search results containing the requested information.

    """

    logging.info("using get_weather tool")
    return tavily.search(query=query)

@tool
def get_job_postings(query: str) -> str:
    
    """
    
    Searches the web for current job postings related to the position and the work location.
    Use this tool when you need real-time job postings at a specific location and for a specific position outside your training data.

    Args:
        query (str): The specific search query to execute. This should be highly targeted,
        for example: "AI engineer positions in Hyderabad" or "Sales manager job postings in New York"

    Returns:
        str: The search results containing the requested information


    """   
    logging.info("using get_job_postings tool")
    return tavily.search(query=query)

   

granite = ChatOllama(temperature=0, model="granite3.3:latest")
tools = [get_weather, get_job_postings]

agent = create_agent(model=granite, tools=tools)

def main():
    
    result = agent.invoke({"messages": HumanMessage(content=input("What's your query?"))})
    logging.info(result)

if __name__ == "__main__":
    main()