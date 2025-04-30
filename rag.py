from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "text-embedding-3-small"
# DOC_PATH = "./policies/store.pdf"


def create_retriever(doc_path):
    # 1. Load documents
    loader = PyPDFLoader(file_path=doc_path)
    raw_documents = loader.load()

    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
    documents = text_splitter.split_documents(raw_documents)

    # 3. Create an embedding function
    embedding_function = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,  # Or another embedding model
    )

    # 4. Create a vector store
    # Builds a vector store with this policy information
    vectorstore = Chroma.from_documents(
        documents=documents, embedding=embedding_function
    )

    # 5. Create the retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}  # Return top 6 most relevant chunks
    )

    return retriever

    # prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    # The store's policy on packaging tape is that they prefer customers to responsibly secure their own boxes but are willing to apply a little piece of tape once in a while out of generosity. They do not want to see their customer's boxes compromised and believe that proper communication can help customers better secure their packages. However, they no longer offer unlimited tape usage due to some customers taking advantage of the service. Each store may have a unique tape policy, as franchise-owned stores can differ in their approaches.
