from src.file_loader import MainLoader
from src.chunking import Chunker
from src.vectorizer import Vectorizer

a = MainLoader()
doc = a.document_loader()

ch = Chunker(doc)
b = ch.recursive_chunking()
c = ch.semantic_chunking()
d = ch.para_chunk()
e = ch.fixed_chunk()

ve = Vectorizer(b)
# ve.dense_vector()
ve.sparse_vector()
ve.create_graph_vectorstore()