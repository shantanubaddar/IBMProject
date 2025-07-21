from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
import torch


loader = PyPDFLoader("/content/Syllabus.pdf")  
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(pages)


embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = FAISS.from_documents(chunks, embedding)


device = 0 if torch.cuda.is_available() else -1
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
tokenizer.model_max_length = 1024  
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

qa_model = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)

print("✅ PDF Q&A system is ready (using flan-t5-large on GPU)!")


while True:
    query = input("\n🔎 Ask a question (or type 'exit'): ").strip()
    if query.lower() in ["exit", "quit"]:
        break
    
    docs = db.similarity_search(query, k=3)
    
    context = "\n\n".join([doc.page_content[:1000] for doc in docs])
    prompt = f"Answer the question based on the text below:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    
    result = qa_model(prompt, max_new_tokens=256)[0]['generated_text']
    print("\n💬 Answer:", result)
