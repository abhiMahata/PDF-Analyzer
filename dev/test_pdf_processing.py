from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

try:
    print("Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Embeddings loaded.")
    
    print("Testing FAISS...")
    knowledge_base = FAISS.from_texts(["hello world", "test string"], embeddings)
    print("FAISS loaded.")
except Exception as e:
    import traceback
    traceback.print_exc()
