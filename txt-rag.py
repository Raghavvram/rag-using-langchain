from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

# Load documents from a text file
loader = TextLoader('sample.txt')
documents = loader.load()

# Split the text documents into chunks
text_splitter = CharacterTextSplitter(separator=" ", chunk_size=200, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Use SentenceTransformer for text embedding
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a Chroma vector store
db = Chroma.from_documents(texts, embeddings)

# Set up a retriever to get relevant documents from text file
retriever = db.as_retriever(search_kwargs={"k":3})

# Retrieve relevant documents and provide an answer
question = input("Enter your question: ")
docs = retriever.get_relevant_documents(question)
relevant_text = ''.join([doc.page_content for doc in docs])
print(relevant_text)
