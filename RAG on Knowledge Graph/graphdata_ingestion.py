from langchain_openai import AzureChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
import os
from dotenv import load_dotenv
load_dotenv()

llm_deployement_name = os.getenv("llm_deployement_name")
azure_endpoint = os.getenv("azure_endpoint")
llm = AzureChatOpenAI(deployment_name=llm_deployement_name, model_name="gpt-35-turbo-16k", azure_endpoint=azure_endpoint)

graph = Neo4jGraph()


text = """Emily is an employee at TechNova, a leading technology company based in Silicon Heights. She has been working there for the past four years as a software developer. James is also an employee at TechNova, where he works as a data analyst. He joined the company three years ago after completing his undergraduate studies. TechNova is a renowned technology company that specializes in developing innovative software solutions and advanced artificial intelligence systems. The company boasts a diverse team of talented professionals from various fields. Both Emily and James are highly skilled experts who contribute significantly to TechNova's achievements. They collaborate closely with their respective teams to create cutting-edge products and services that cater to the dynamic needs of the company's clients."""


documents = [Document(page_content=text)]
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(texts)

print(f"Nodes:{graph_documents[0].nodes}")
print("-----------------------------------------------------------------")
print(f"Relationships:{graph_documents[0].relationships}")


graph.add_graph_documents(graph_documents)
print("Documents successfully added to Graph DataBase")
