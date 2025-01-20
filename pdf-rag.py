import fitz  # PyMuPDF
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Function to load PDF content
def load_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Read and process PDF file
pdf_text = load_pdf('sample.pdf')

# Load PDF text as documents
pdf_loader = TextLoader(text_content=pdf_text)
pdf_documents = pdf_loader.load()

# Split the PDF documents into chunks
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=200, chunk_overlap=0)
pdf_texts = text_splitter.split_documents(pdf_documents)

# Use SentenceTransformer for text embedding
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a Chroma vector store
pdf_db = Chroma.from_documents(pdf_texts, embeddings)

# Set up a retriever to get relevant documents from PDF
pdf_retriever = pdf_db.as_retriever(search_kwargs={"k":3})

# Retrieve relevant documents and provide an answer
question = input("Enter your question: ")
pdf_docs = pdf_retriever.get_relevant_documents(question)
pdf_relevant_text = ''.join([doc.page_content for doc in pdf_docs])
print(pdf_relevant_text)
