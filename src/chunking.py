from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL
)

class Chunker:
    def __init__(self, documents):
        self.documents = documents

    def recursive_chunking(self):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 100,
        )

        chunks = text_splitter.split_documents(self.documents)

        print("Documents are chunked!")
        print(f"Recursive Chunked documents: {len(chunks)}")

        return chunks
    
    def semantic_chunking(self):

        text_splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )

        chunks = text_splitter.split_documents(self.documents)

        print("Documents are chunked!")
        print(f"Semantic Chunked documents: {len(chunks)}")

        return chunks
    
    def para_chunk(self):
        paragraph = CharacterTextSplitter(
            separator="\n\n",
            chunk_size = 4000,
            chunk_overlap = 0
        )

        chunks = paragraph.split_documents(self.documents)

        print("Paragraph Chunks Created")
        print(f"No of Paragrah Chunks: {len(chunks)}")

        return chunks
    
    def fixed_chunk(self):
        splitter = CharacterTextSplitter(chunk_size = 500)

        chunks = splitter.split_documents(self.documents)

        print("Fixed Chunks Created")
        print(f"No of Fixed Chunks: {len(chunks)}")

        return chunks