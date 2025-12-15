import logging
logging.basicConfig(level=logging.WARNING)
from processor import get_retriever
from qa_engine import answer_query

retriever = get_retriever()
query = "education"
try:
	docs = retriever.get_relevant_documents(query)
except Exception:
	# Newer LangChain retrievers may support invoke()
	docs = retriever.invoke(query)

print("=" * 80)
print("RAW CHUNK #1 FROM VECTOR DB:")
print("=" * 80)
if not docs:
	print("<no chunks retrieved>")
else:
	for i, doc in enumerate(docs[:3]):
		print(f"\n--- Chunk {i+1} (source={doc.metadata.get('source_file','unknown')}) ---")
		print(doc.page_content[:500])

print("\n" + "=" * 80)
print("LLAMA'S ANSWER TO: 'Tell me about education'")
print("=" * 80)
answer = answer_query('Tell me about education')
print(answer)
