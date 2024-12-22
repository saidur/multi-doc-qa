from flask import Flask, request, jsonify, render_template
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

# Initialize embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
vectorstore = None

# Initialize vector store
def initialize_vectorstore():
    global vectorstore
    vectorstore = Chroma(embedding_function=embeddings)

initialize_vectorstore()

# Crawl website using Beautiful Soup
def crawl_website(url, max_pages=10):
    visited = set()
    to_visit = [url]
    crawled_data = []

    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue

        try:
            response = requests.get(current_url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text from the page
                text = soup.get_text(strip=True)
                crawled_data.append(Document(page_content=text, metadata={"url": current_url}))

                # Find new links to visit
                for a_tag in soup.find_all('a', href=True):
                    href = a_tag['href']
                    if href.startswith('/') or href.startswith(url):
                        full_url = requests.compat.urljoin(url, href)
                        if full_url not in visited:
                            to_visit.append(full_url)
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")

        visited.add(current_url)

    return crawled_data

# Process uploaded PDF
def process_pdf(file):
    loader = PyPDFLoader(file)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    return splitter.split_documents(documents)

# Add documents to the vector store
def add_to_vectorstore(docs):
    global vectorstore
    vectorstore.add_documents(docs)

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['file']
    if not file.filename.endswith('.pdf'):
        return jsonify({"error": "Invalid file format. Only PDFs are supported."}), 400

    # Process and add the PDF to the vector store
    documents = process_pdf(file)
    add_to_vectorstore(documents)
    return jsonify({"message": "PDF uploaded and processed successfully!"})

@app.route('/crawl', methods=['POST'])
def crawl():
    url = request.json.get("url")
    max_pages = int(request.json.get("max_pages", 10))

    if not url:
        return jsonify({"error": "No URL provided"}), 400

    # Crawl website and add data to the vector store
    crawled_docs = crawl_website(url, max_pages)
    add_to_vectorstore(crawled_docs)
    return jsonify({"message": f"Crawled {len(crawled_docs)} pages and added to knowledge base!"})

@app.route('/query', methods=['POST'])
def query():
    question = request.json.get("question")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Use the retrieval QA system to answer the question
    retriever = vectorstore.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm="openai-gpt-3.5-turbo", chain_type="stuff", retriever=retriever)
    answer = qa_chain.run(question)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
