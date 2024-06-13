from langchain_openai import AzureChatOpenAI
from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
import os
from dotenv import load_dotenv
load_dotenv()

llm_deployement_name = os.getenv("llm_deployement_name")
azure_endpoint = os.getenv("azure_endpoint")

llm = AzureChatOpenAI(deployment_name=llm_deployement_name, model_name="gpt-35-turbo-16k", azure_endpoint=azure_endpoint)

graph = Neo4jGraph()

chain = GraphCypherQAChain.from_llm(
    llm, graph=graph, verbose=True
)


result = chain.invoke({"query": "Does James work for the same company as Emily ?"})

print(result)