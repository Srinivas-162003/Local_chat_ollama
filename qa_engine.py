from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from processor import get_retriever

llm = Ollama(model="llama3.1:8b")
retriever = get_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

def ask_query():
    print("\n💬 Ask questions based on uploaded documents. Type 'exit' to stop.\n")
    while True:
        query = input(">> ")
        if query.lower() in ["exit", "quit"]:
            break
        try:
            answer = qa_chain.run(query)
            print("🤖", answer)
        except Exception as e:
            print("❌ Error:", e)
