"""
Core RAG System - OpenRouter.ai Version (FREE!)
"""

import os
from typing import Dict, List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

class RAGSystem:
    """Retrieval-Augmented Generation System using OpenRouter (FREE)"""
    
    def __init__(self, db_path: str = "vectordb"):
        """Initialize the RAG system"""
        self.db_path = db_path
        self.vectordb = None
        self.qa_chain = None
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def load_database(self):
        """Load the vector database"""
        if not os.path.exists(self.db_path):
            raise Exception(f"Database not found at {self.db_path}. Run build_database.py first!")
        
        print("📚 Loading vector database...")
        self.vectordb = FAISS.load_local(
            self.db_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        print("✅ Database loaded!")
        
    def setup_qa_chain(self):
        """Setup the question-answering chain with OpenRouter"""
        
        # Create custom prompt
        template = """You are a helpful Deep Learning course assistant.
Answer the question using ONLY the context provided from course materials.
If the answer is not in the context, say "I cannot find this information in the course materials."
Always cite the source when possible.

Context from course materials:
{context}

Question: {question}

Detailed Answer with citations:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Get API credentials
        api_key = os.getenv("OPENAI_API_KEY")
        api_base = os.getenv("OPENAI_API_BASE", "https://openrouter.ai/api/v1")
        
        if not api_key:
            raise Exception("OPENAI_API_KEY not found in .env file!")
        
        print(f"🔗 Connecting to OpenRouter...")
        
        # Initialize OpenRouter LLM
        # Using gpt-3.5-turbo through OpenRouter (FREE!)
        llm = OpenAI(
            temperature=0,
            model_name="openai/gpt-3.5-turbo-instruct",
            openai_api_key=api_key,
            openai_api_base=api_base,
            max_tokens=500
        )
        
        print("✅ OpenRouter connected!")
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectordb.as_retriever(
                search_kwargs={"k": 4}  # Retrieve top 4 chunks
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        print("✅ QA chain ready!")
        
    def initialize(self):
        """Initialize the complete system"""
        self.load_database()
        self.setup_qa_chain()
        
    def ask(self, question: str) -> Dict:
        """Ask a question and get answer with sources"""
        
        if not self.qa_chain:
            raise Exception("System not initialized! Call initialize() first.")
        
        # Get answer from chain
        result = self.qa_chain({"query": question})
        
        # Extract source information
        sources = []
        for doc in result['source_documents']:
            source_info = {
                'content': doc.page_content[:300] + "...",
                'source': doc.metadata.get('source', 'Unknown'),
                'page': doc.metadata.get('page', 'N/A')
            }
            sources.append(source_info)
        
        return {
            'question': question,
            'answer': result['result'],
            'sources': sources
        }

# Test the system
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 TESTING RAG SYSTEM - OpenRouter Version")
    print("="*60 + "\n")
    
    # Initialize system
    rag = RAGSystem()
    rag.initialize()
    
    # Test questions
    test_questions = [
        "What is backpropagation?",
        "Explain CNNs",
        "What is the difference between SGD and Adam optimizer?"
    ]
    
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"❓ Question: {q}")
        print('='*60)
        
        result = rag.ask(q)
        
        print(f"\n💡 Answer:\n{result['answer']}")
        print(f"\n📚 Sources:")
        for i, source in enumerate(result['sources'], 1):
            print(f"\n[{i}] {os.path.basename(source['source'])} - Page {source['page']}")
            print(f"    {source['content'][:150]}...")