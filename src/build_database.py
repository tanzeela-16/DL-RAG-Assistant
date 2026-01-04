"""
Step 1: Build Vector Database from PDFs
Run this ONCE to create your database: python src/build_database.py
"""

import os
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_database():
    """Build vector database from course materials"""
    
    print("\n" + "="*60)
    print("🚀 BUILDING VECTOR DATABASE")
    print("="*60)
    
    # Configuration
    docs_path = "course_materials"
    db_path = "vectordb"
    
    # Check if PDFs exist
    pdf_files = list(Path(docs_path).glob("*.pdf"))
    if not pdf_files:
        print(f"❌ ERROR: No PDF files found in {docs_path}/")
        print(f"Please add your PDFs to the {docs_path}/ folder first!")
        return
    
    print(f"\n✅ Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    
    # Step 1: Load PDFs
    print("\n📚 Step 1: Loading PDF documents...")
    loader = DirectoryLoader(
        docs_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")
    
    # Step 2: Split into chunks
    print("\n✂️  Step 2: Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    # Step 3: Create embeddings
    print("\n🧠 Step 3: Creating embeddings (this takes 3-5 minutes)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("✅ Embedding model loaded")
    
    # Step 4: Build FAISS vector database
    print("\n🔨 Step 4: Building FAISS vector database...")
    vectordb = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector database created")
    
    # Step 5: Save to disk
    print(f"\n💾 Step 5: Saving database to {db_path}/...")
    os.makedirs(db_path, exist_ok=True)
    vectordb.save_local(db_path)
    print("✅ Database saved successfully!")
    
    print("\n" + "="*60)
    print("🎉 DATABASE BUILD COMPLETE!")
    print("="*60)
    print(f"\n📊 Summary:")
    print(f"   - PDFs processed: {len(pdf_files)}")
    print(f"   - Total pages: {len(documents)}")
    print(f"   - Total chunks: {len(chunks)}")
    print(f"   - Database location: {db_path}/")
    print(f"\n✅ You can now run: streamlit run src/app.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    try:
        build_database()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Make sure PDFs are in course_materials/ folder")
        print("2. Check if all packages are installed: pip install -r requirements.txt")
        print("3. Make sure PDFs are not password-protected")