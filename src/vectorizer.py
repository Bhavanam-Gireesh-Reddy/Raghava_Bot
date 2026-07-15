from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os
from pathlib import Path
from langchain_community.retrievers import TFIDFRetriever
import pickle
from langchain_community.graph_vectorstores import CassandraGraphVectorStore
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

embeddings = HuggingFaceEmbeddings(
    model_name = EMBEDDING_MODEL
)

class Vectorizer:
    def __init__(self, chunks = None):
        self.chunks = chunks

    def dense_vector(self):
        CHROMA_DIR = "dense_db"
        if Path(CHROMA_DIR).exists():
            print("Using Existing Dense Db ...")
            db = Chroma(
                persist_directory=CHROMA_DIR,
                embedding_function=embeddings
            )

        else:
            print("Ctreating New Dense DB...")
            db = Chroma.from_documents(
                self.chunks,
                embeddings,
                persist_directory=CHROMA_DIR
            )

        print("Dense Chroma DB is created Sucessfully!")
        return db
    
    def sparse_vector(self):
        retriever = TFIDFRetriever.from_documents(self.chunks)
        with open("tfidf_retriever.pkl", "wb") as f:
            pickle.dump(retriever, f)
            print("Sparse Vector Db created Sucessfully!")


    def create_graph_vectorstore(self, threshold = 0.5):

        GRAPH_PATH = "graph.pkl"
        VECTOR_DB_PATH = "dense_db"
        
        os.makedirs("dense_db", exist_ok=True)

        # Load existing databases
        if os.path.exists(GRAPH_PATH):

            vector_store = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )

            with open(GRAPH_PATH, "rb") as file:
                graph = pickle.load(file)

            print("Graph vector store loaded")

            return vector_store, graph

        # Create vector database
        db = Chroma.from_documents(
            documents=self.chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DB_PATH
        )

        # Create graph
        graph = nx.Graph()

        texts = [chunk.page_content for chunk in self.chunks]
        vectors = embeddings.embed_documents(texts)

        # Add nodes
        for i, chunk in enumerate(self.chunks):
            graph.add_node(
                i,
                text=chunk.page_content,
                metadata=chunk.metadata
            )

        # Add similarity edges
        for i in range(len(self.chunks)):
            for j in range(i + 1, len(self.chunks)):

                score = cosine_similarity(
                    [vectors[i]],
                    [vectors[j]]
                )[0][0]

                if score >= threshold:
                    graph.add_edge(
                        i,
                        j,
                        weight=float(score)
                    )

        # Save graph locally
        with open(GRAPH_PATH, "wb") as file:
            pickle.dump(graph, file)

        print("Graph vector store created")

        return db, graph