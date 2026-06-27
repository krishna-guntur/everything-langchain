from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from dotenv import load_dotenv
import os
from prompts.summaryTemplate import SummaryTemplate

load_dotenv()


def main():

    entity = input("Give an entity name: ")
    
    summary_prompt_template = PromptTemplate(
        input_variables=['entity'],
        template=SummaryTemplate
    )
    llm = ChatOllama(temperature=0, model="granite3.3:latest")
    chain = summary_prompt_template | llm

    response = chain.invoke(input={"entity": entity})
    print(response.content)


if __name__ == "__main__":
    main()
