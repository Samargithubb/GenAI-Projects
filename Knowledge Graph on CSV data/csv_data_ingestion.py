from langchain_openai import AzureChatOpenAI
from langchain_community.graphs import Neo4jGraph

import os
from dotenv import load_dotenv
load_dotenv()

llm_deployement_name = os.getenv("llm_deployement_name")
azure_endpoint = os.getenv("azure_endpoint")
llm = AzureChatOpenAI(deployment_name=llm_deployement_name, model_name="gpt-35-turbo-16k", azure_endpoint=azure_endpoint)

graph = Neo4jGraph()

movies_query = """
LOAD CSV WITH HEADERS FROM
'https://raw.githubusercontent.com/Samargithubb/GenAI-Projects/main/Knowledge%20Graph%20on%20CSV%20data/movies.csv'
AS row
MERGE(m:Movie {id:row.movieId})
SET m.released = date(row.released),
    m.title = row.title,
    m.imdbRating =toFloat(row.imdbRating)
FOREACH (director in split(row.director, '|') |
    MERGE (p:Person {name:trim(director)})
    MERGE (p)-[:DIRECTED]->(m))
FOREACH (actor in split(row.actors,'|') |
    MERGE (p:Person {name:trim(actor)})
    MERGE (p)-[:ACTED_IN]->(m))
FOREACH (genre in split(row.genres, '|') |
    MERGE (g:Genre {name:trim(genre)})
    MERGE (m)-[:IN_GENRE]->(g))    
"""

graph.query(movies_query)
graph.refresh_schema()
print(graph.schema)

