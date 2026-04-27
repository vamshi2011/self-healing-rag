import os
from typing import TypedDict
from dotenv import load_dotenv
import chromadb
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

load_dotenv()

# --- Step 2: Define the Agent State ---
class AgentState(TypedDict):
    question: str
    rewritten_question: str
    documents: list[str]
    answer: str
    grade: str
    retry_count: int

# --- Step 3: Initialize LLM and Vector Store ---
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.environ.get("GROQ_API_KEY")
)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

# FIX: Health check on startup — immediately warn if the knowledge base is empty.
# The most common cause of "Found 0 chunks" is forgetting to run ingest.py first.
def check_knowledge_base():
    count = collection.count()
    if count == 0:
        print("\n" + "="*60)
        print("WARNING: Your ChromaDB knowledge base is EMPTY!")
        print("You must run ingest.py before using this agent.")
        print("Steps:")
        print("  1. Create a 'docs/' folder next to your scripts")
        print("  2. Put your .txt files inside docs/")
        print("  3. Run: python ingest.py")
        print("  4. Then re-run this script")
        print("="*60 + "\n")
    else:
        print(f"Knowledge base loaded: {count} chunks ready.")

# --- Step 4: The Retrieve Node ---
def retrieve(state: AgentState) -> AgentState:
    """Search ChromaDB for documents relevant to the current question."""
    question = state.get("rewritten_question") or state["question"]
    print(f"\n[RETRIEVE] Searching for: {question}")

    try:
        results = collection.query(
            query_texts=[question],
            n_results=3
        )
        documents = results["documents"][0]
        print(f"[RETRIEVE] Found {len(documents)} chunks.")
    except Exception as error:
        print(f"[RETRIEVE] ChromaDB query failed: {error}")
        documents = []

    return {**state, "documents": documents}

# --- Step 5: The Generate Node ---
def generate(state: AgentState) -> AgentState:
    """Generate an answer using retrieved documents as context."""
    question = state.get("rewritten_question") or state["question"]
    documents = state.get("documents", [])

    print(f"\n[GENERATE] Generating answer for: {question}")

    if not documents:
        return {**state, "answer": "I don't know. No relevant documents were found."}

    context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(documents)])

    system_prompt = """You are a helpful assistant that answers questions 
    based strictly on the provided documents. If the documents do not contain 
    enough information to answer the question, say so honestly by saying "I don't know". 
    Do not use any knowledge outside of the provided documents."""

    user_prompt = f"Documents:\n{context}\n\nQuestion: {question}\n\nAnswer based only on the documents above:"

    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        answer = response.content
        print(f"[GENERATE] Answer generated ({len(answer)} chars).")
    except Exception as error:
        print(f"[GENERATE] LLM call failed: {error}")
        answer = "An error occurred while generating the answer."

    return {**state, "answer": answer}

# --- Step 6: The Grade Node ---
def grade_answer(state: AgentState) -> AgentState:
    """Grade whether the answer is grounded in the retrieved documents."""
    question = state["question"]
    documents = state.get("documents", [])
    answer = state.get("answer", "")

    print("\n[GRADE] Evaluating answer quality...")

    if "I don't know" in answer:
        print("[GRADE] Answer is 'I don't know'. Forcing a FAIL to trigger retry.")
        return {**state, "grade": "fail"}

    context = "\n\n".join(documents) if documents else "No documents."

    grading_prompt = f"""You are a strict grader evaluating whether an answer 
    is grounded in the provided documents.
    
    Documents:
    {context}
    
    Answer to evaluate:
    {answer}
    
    If the answer contains ANY information not found in the documents, or if it is unhelpful, respond with exactly the word "fail". 
    If the answer is completely supported by the documents, respond with exactly the word "pass"."""

    response = llm.invoke([HumanMessage(content=grading_prompt)])
    grade = response.content.strip().lower()

    if "pass" in grade:
        grade_result = "pass"
        print("[GRADE] Result: PASS ✅")
    else:
        grade_result = "fail"
        print("[GRADE] Result: FAIL ❌")

    return {**state, "grade": grade_result}

# --- Step 7: The Rewrite Node ---
def rewrite_question(state: AgentState) -> AgentState:
    """Rewrite the question to try and get better search results."""
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    print("\n[REWRITE] Rephrasing the question...")

    # FIX: Explicitly tell the LLM NOT to wrap the query in quotes.
    # The original prompt caused the LLM to return `"Definition of Python"` (with quotes),
    # which are then passed literally to ChromaDB and hurt vector search quality.
    rewrite_prompt = f"""Look at this initial question: {question}
    Formulate an improved, alternative search query that might yield better results from a vector database.
    Respond ONLY with the new query as plain text. Do NOT wrap it in quotes or add any explanation."""

    response = llm.invoke([HumanMessage(content=rewrite_prompt)])
    new_question = response.content.strip().strip('"').strip("'")  # Extra safety strip

    print(f"[REWRITE] New question: {new_question}")

    return {
        **state,
        "rewritten_question": new_question,
        "retry_count": retry_count + 1
    }

# --- Step 8: Edge Routing Functions ---
def check_grade(state: AgentState) -> str:
    if state["grade"] == "pass":
        return "end"
    else:
        return "rewrite"

def check_retries(state: AgentState) -> str:
    if state.get("retry_count", 0) < 2:
        return "retrieve"
    else:
        return "give_up"

def give_up(state: AgentState) -> AgentState:
    print("\n[GIVE UP] Max retries reached.")
    return {**state, "answer": "🤷 I couldn't find a good answer in the documents, even after trying different searches."}

# --- Step 9: Build the Graph ---
workflow = StateGraph(AgentState)

workflow.add_node("retrieve", retrieve)
workflow.add_node("generate", generate)
workflow.add_node("grade_answer", grade_answer)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("give_up", give_up)

workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", "grade_answer")

workflow.add_conditional_edges(
    "grade_answer",
    check_grade,
    {"end": END, "rewrite": "rewrite_question"}
)

workflow.add_conditional_edges(
    "rewrite_question",
    check_retries,
    {"retrieve": "retrieve", "give_up": "give_up"}
)

workflow.add_edge("give_up", END)

app = workflow.compile()

# --- Run the App ---
if __name__ == "__main__":
    print("\n🤖 Self-Healing RAG Agent Initialized!")

    # FIX: Check if the knowledge base has data before accepting questions
    check_knowledge_base()

    while True:
        user_input = input("\nAsk a question (or type 'quit'): ")
        if user_input.lower() == 'quit':
            break

        # FIX: Always reset rewritten_question to empty string for each new query.
        # Without this, a rewritten question from a previous session can persist
        # in state and be used as the search query for the next unrelated question.
        inputs = {"question": user_input, "rewritten_question": "", "retry_count": 0}

        for output in app.stream(inputs):
            for key, value in output.items():
                pass  # Node logs will print automatically

        print(f"\n✅ Final Answer: {value.get('answer', '')}")