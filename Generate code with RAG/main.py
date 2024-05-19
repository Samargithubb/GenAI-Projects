from  langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import Language
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
import warnings
warnings.filterwarnings('ignore')
import os

code_doc = []
def code_reader_func(file_name):
    try:
        with open(file_name, "r") as f:
            content = f.read()
            doc = Document(page_content=content, metadata= {"filename": file_name,"file_index":0})
            code_doc.append(doc)
            return code_doc

    except Exception as e:
        return {"error": str(e)}

code_reader_func("data/test.py")

text_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,chunk_size=2000, chunk_overlap=200
)
texts = text_splitter.split_documents(code_doc)

embeddings = OllamaEmbeddings(model="llama2:7b")


doc_result = embeddings.embed_documents([code_doc])


db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever(
    search_type="similarity", search_kwargs={"k": 2},)

code_llm = Ollama(model="codellama",)

user_question = input("Enter Query to generate Code: ")

# RAG template
prompt_RAG = """
    You are a proficient python developer. Respond with the syntactically correct code for to the question below. Make sure you follow these rules:

    1. Use context to understand the APIs and how to use it & apply.
    2. Ensure all the requirements in the question are met.
    3. Ensure the output code syntax is correct.
    4. All required dependency should be imported above the code.

    Question:
    {question}

    Context:
    {context}

    Helpful Response :
    """

prompt_RAG_tempate = PromptTemplate(
    template=prompt_RAG, input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_llm(
    llm=code_llm, prompt=prompt_RAG_tempate, retriever=retriever, return_source_documents=True,
)

results = qa_chain({"query": user_question})


filename = "code.py"
try:
    with open(os.path.join("output", filename), "w") as f:
        f.write(str(results["result"]))
    print("Code Successfully Generated and Saved in code.py", filename)
except Exception as e:
    print("Error saving file...",e)
