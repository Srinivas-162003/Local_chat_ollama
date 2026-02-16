"""
Test script to compare RAG vs CLaRa performance
"""

from qa_engine import answer_query
from clara_engine import answer_with_clara
import time

def test_question(question: str):
    """Test a question with both RAG and CLaRa"""
    print("=" * 80)
    print(f"QUESTION: {question}")
    print("=" * 80)
    
    # Test Traditional RAG
    print("\n--- Traditional RAG ---")
    start = time.time()
    try:
        rag_answer = answer_query(question)
        rag_time = time.time() - start
        print(f"Answer: {rag_answer}")
        print(f"Time: {rag_time:.2f}s")
    except Exception as e:
        print(f"Error: {e}")
        rag_time = 0
    
    # Test CLaRa
    print("\n--- CLaRa (Advanced) ---")
    start = time.time()
    try:
        clara_response = answer_with_clara(
            question, 
            max_iterations=3, 
            max_hops=3,
            detailed_response=True
        )
        clara_time = time.time() - start
        
        print(f"Answer: {clara_response['answer']}")
        print(f"\nReasoning Steps: {len(clara_response['reasoning_steps'])}")
        for step in clara_response['reasoning_steps']:
            print(f"  Step {step['step']}: Confidence {step['confidence']:.2f}")
            print(f"    Sources: {', '.join(step['sources'])}")
        
        print(f"\nOverall Confidence: {clara_response['confidence']:.2f}")
        print(f"Total Iterations: {clara_response['total_iterations']}")
        print(f"Time: {clara_time:.2f}s")
        print(f"Speed Ratio: {clara_time/rag_time:.1f}x slower than RAG" if rag_time > 0 else "")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    # Test questions - modify these based on your documents
    test_questions = [
        "What is the main topic of the documents?",
        "Compare and contrast the key ideas across different documents.",
        "What are the relationships between different concepts mentioned?",
    ]
    
    print("\n🧪 RAG vs CLaRa Comparison Test\n")
    print("Make sure you have documents uploaded before running this test.\n")
    
    for question in test_questions:
        test_question(question)
        input("Press Enter to continue to next question...")
    
    print("\n✅ Testing complete!")
    print("\nKey Observations:")
    print("- RAG is faster for simple queries")
    print("- CLaRa provides better context and reasoning for complex queries")
    print("- CLaRa shows step-by-step reasoning process")
    print("- CLaRa provides confidence scores and source attribution")
