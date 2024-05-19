import langchain_community
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def retrieve_docs(query: str, search_type: str = "similarity_score_threshold", k: int = 1) -> list:
    # Load the documents from the vector store
    loader = TextLoader("../../state_of_the_union.txt")
    documents = loader.load()

    # Split the texts into chunks and generate embeddings
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # Create a retriever from the vector store
    retriever = db.as_retriever(search_type=search_type, search_kwargs={"k": k})

    # Invoke the retriever with the query and return the results
    docs = retriever.invoke(query)
    return docs

query = "what did he say about ketanji brown jackson"
search_type = "similarity_score_threshold"
k = 1
docs = retrieve_docs(query, search_type, k)
print(docs)
