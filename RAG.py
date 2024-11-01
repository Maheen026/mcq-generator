import os
import streamlit as st
from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from fpdf import FPDF
import tempfile

# Set up the Streamlit UI
st.set_page_config(page_title="Tech MCQ Generator", page_icon="📄", layout="wide")

# Sidebar for file upload and Groq API key
st.sidebar.header("📁 Upload your PDF or Word document")
uploaded_file = st.sidebar.file_uploader("Upload your file", type=["pdf", "docx"])

st.sidebar.header("🔑 Groq API Key")
groq_api_key = st.sidebar.text_input("🔑 Enter your Groq API key", type="password")

st.sidebar.markdown("---")

# Main UI section
st.title("📄 Tech MCQ Generator")
st.markdown("Generate Multiple Choice Questions (MCQs) from tech-related documents using Groq and LangChain models.")

# Function to check if the document is tech-related using Groq
def is_tech_related(content, groq_api_key):
    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Classify if the following content is tech-related:\n\n{content[:1700]}"}],
        model="gemma-7b-it",
    )
    result = response.choices[0].message.content.strip().lower()
    return "yes" in result

# Function to generate MCQs using Groq
def generate_mcqs(content, difficulty_level, num_questions, groq_api_key):
    template = """
    You are an AI designed to generate high-quality tech-related multiple-choice questions (MCQs) from the provided document.
    Focus only on tech-related content, such as software, networking, AI, cybersecurity, and cloud computing.

    - Easy: Basic recall questions.
    - Medium: Comprehension or application questions.
    - Hard: Critical thinking and integration of concepts.

    Context: {context}
    Generate {num_questions} MCQs of {difficulty_level} difficulty based on the document.
    """
    prompt = PromptTemplate(template=template, input_variables=["context", "difficulty_level", "num_questions"])
    formatted_prompt = prompt.format(context=content, difficulty_level=difficulty_level, num_questions=num_questions)

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": formatted_prompt}],
        model="gemma-7b-it") #, "llama3-groq-8b-8192-tool-use-preview", "llama-3.1-70b-versatile")
    return response.choices[0].message.content.strip()

# Function to convert the MCQs to PDF
def convert_to_pdf(mcqs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in mcqs.split("\n"):
        pdf.multi_cell(0, 10, line)
    return pdf

# Main functionality
if uploaded_file and groq_api_key:
    with st.spinner("Processing document..."):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        # Load and process the PDF content
        doc_loader = PyPDFLoader(temp_file_path)
        doc_content = doc_loader.load()

        # Convert to plain text for easier processing
        doc_text = " ".join([doc.page_content for doc in doc_content])

        # Check if the document is tech-related
        if is_tech_related(doc_text, groq_api_key):
            st.success("The document is classified as tech-related!")

            # Difficulty level selection
            difficulty = st.selectbox("Select difficulty level", ["Easy", "Medium", "Hard"])
            num_questions = st.slider("Number of questions", 5, 20)
            model_name = st.selectbox("model_name", ["llama3-groq-8b-8192-tool-use-preview","gemma2-9b-it","llama-3.1-70b-versatile"])

            if st.button("Generate MCQs"):
                # Generate MCQs based on the document content
                mcqs = generate_mcqs(doc_text, difficulty, num_questions, groq_api_key)

                # Display the MCQs in the app
                st.subheader("Generated MCQs")
                st.write(mcqs)

                # Provide option to download as PDF
                if st.button("Download MCQs as PDF"):
                    pdf = convert_to_pdf(mcqs)
                    pdf_output = pdf.output(dest="S").encode("latin1")
                    st.download_button("Download PDF", data=pdf_output, file_name="mcqs.pdf", mime="application/pdf")
        else:
            st.error("The document is not tech-related.")
else:
    st.info("Please upload a document and enter your Groq API key to get started.")

