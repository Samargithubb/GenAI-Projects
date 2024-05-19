import langchain_community
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

def load_data_from_vectors(threshold, top_k=None):
    """
    A function that loads data from a vector store using similarity score threshold retrieval.

    Args:
        threshold (float): The minimum similarity score required for a document to be retrieved.
        top_k (int, optional): The number of documents to retrieve. Defaults to None.

    Returns:
        list: A list of retrieved documents.
    """
    # Load the data from the vector store
    loader = TextLoader("../../state_of_the_union.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(texts, embeddings)

    # Create a retriever object using the similarity score threshold search type
    retriever = db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": threshold})

    # Invoke the retriever with the given query
    docs = retriever.invoke("what did he say about ketanji brown jackson")

    # Return the top k documents if specified, otherwise return all retrieved documents
    if top_k:
        return docs[:top_k]
    else:
        return docs

# Example usage of the function
threshold = 0.5
top_k = 3
print(load_data_from_vectors(threshold, top_k))