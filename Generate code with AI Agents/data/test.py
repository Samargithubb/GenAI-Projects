from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader



loader = TextLoader("../../state_of_the_union.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()
docs = retriever.invoke("what did he say about ketanji brown jackson")


# Maximum marginal relevance retrieval
#By default, the vector store retriever uses similarity search. If the underlying vector store supports maximum marginal relevance search, you can specify that as the search type.
retriever = db.as_retriever(search_type="mmr")
docs = retriever.invoke("what did he say about ketanji brown jackson")



#Similarity score threshold retrieval
retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
)
docs = retriever.invoke("what did he say about ketanji brown jackson")



#Specifying top k
#You can also specify search kwargs like k to use when doing retrieval.
retriever = db.as_retriever(search_kwargs={"k": 1})
docs = retriever.invoke("what did he say about ketanji brown jackson")
len(docs)