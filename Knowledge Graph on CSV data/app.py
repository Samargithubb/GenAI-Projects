from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

import os
from dotenv import load_dotenv
load_dotenv()

llm_deployement_name = os.getenv("llm_deployement_name")
azure_endpoint = os.getenv("azure_endpoint")
llm = AzureChatOpenAI(deployment_name=llm_deployement_name, model_name="gpt-35-turbo-16k", azure_endpoint=azure_endpoint)

graph = Neo4jGraph()


chain = GraphCypherQAChain.from_llm(graph=graph,llm=llm,verbose=True)

response = chain.invoke({"query":"What was the cast of the Casino?"})
print(response)