"""
Streamlit Web Interface - OpenRouter Version (FREE!)
Run: streamlit run src/app.py
"""

import streamlit as st
import os
from rag_system import RAGSystem

# Page configuration
st.set_page_config(
    page_title="Deep Learning Assistant",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = None
    st.session_state.initialized = False

# Header
st.title("🧠 Deep Learning RAG Assistant")
st.markdown("*Powered by OpenRouter.ai - FREE!* 🆓")

# Sidebar
with st.sidebar:
    st.header("⚙️ System Status")
    
    # Check if database exists
    db_exists = os.path.exists("vectordb")
    
    if db_exists:
        st.success("✅ Database found")
    else:
        st.error("❌ Database not found")
        st.warning("Run: `python src/build_database.py`")
    
    # Check if .env exists
    env_exists = os.path.exists(".env")
    if env_exists:
        st.success("✅ API key configured")
    else:
        st.error("❌ .env file missing")
        st.info("Create .env with your OpenRouter key")
    
    st.divider()
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary", disabled=not (db_exists and env_exists)):
        with st.spinner("Loading system..."):
            try:
                st.session_state.rag = RAGSystem()
                st.session_state.rag.initialize()
                st.session_state.initialized = True
                st.success("✅ System ready!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Status indicator
    if st.session_state.initialized:
        st.success("🟢 System Online")
    else:
        st.info("⚪ System Offline")
    
    st.divider()
    
    st.header("📚 Features")
    st.markdown("""
    - ✅ 100% FREE (OpenRouter)
    - ✅ Fast responses (2-3 sec)
    - ✅ Citation-backed answers
    - ✅ Semantic search
    - ✅ Course-aligned responses
    """)
    
    st.divider()
    
    # Team info
    st.markdown("### 👥 Team")
    st.markdown("""
    - Muhammad Hammad
    - Preet Sawari Mandhwani
    - Tanzeela Memon
    
    **Instructor:** Dr. Ismail
    """)

# Main content
if not st.session_state.initialized:
    # Welcome screen
    st.info("👈 Click 'Initialize System' in the sidebar to start")
    
    st.markdown("""
    ### 📋 How to Use:
    
    1. **Setup Complete** ✅
       - Database built
       - OpenRouter API key configured
    
    2. **Initialize System**: Click the button in sidebar
    
    3. **Ask Questions**: Type your questions below
    
    4. **Get Answers**: With citations from your course materials!
    """)
    
    # Example questions
    st.markdown("### 💡 Example Questions:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - What is backpropagation?
        - Explain convolutional neural networks
        - How does dropout work?
        """)
    
    with col2:
        st.markdown("""
        - Difference between SGD and Adam?
        - What are activation functions?
        - Explain gradient descent
        """)

else:
    # Question interface
    st.header("💬 Ask a Question")
    
    # Sample questions as buttons
    with st.expander("📝 Quick Questions"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔹 Explain Backpropagation"):
                st.session_state.current_question = "Explain backpropagation in detail"
        
        with col2:
            if st.button("🔹 What are CNNs?"):
                st.session_state.current_question = "What are Convolutional Neural Networks?"
        
        with col3:
            if st.button("🔹 SGD vs Adam"):
                st.session_state.current_question = "What is the difference between SGD and Adam optimizer?"
    
    # Question input
    question = st.text_area(
        "Your Question:",
        value=st.session_state.get('current_question', ''),
        height=100,
        placeholder="e.g., How does gradient descent work?"
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Clear", use_container_width=True)
    
    if clear_button:
        st.session_state.current_question = ""
        st.rerun()
    
    # Process question
    if ask_button and question:
        with st.spinner("🔎 Searching course materials..."):
            try:
                result = st.session_state.rag.ask(question)
                
                # Display answer
                st.markdown("### 📖 Answer")
                st.markdown(result['answer'])
                
                # Display sources
                st.markdown("### 🔗 Sources")
                st.markdown("*Click to expand and view source content*")
                
                for i, source in enumerate(result['sources'], 1):
                    source_name = os.path.basename(source['source'])
                    with st.expander(f"📄 Source {i}: {source_name} (Page {source['page']})"):
                        st.text(source['content'])
                
                # Success message
                st.success("✅ Answer retrieved successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <small>Deep Learning RAG Assistant | Powered by OpenRouter.ai (FREE) | November 2025</small>
</div>
""", unsafe_allow_html=True)