import google.generativeai as genai
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain  

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)  # Corrected typo
    chunks = text_splitter.split_text(text)  # Fixed typo here
    return chunks

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """  # Changed variable name to avoid confusion
        Answer the questions as detailed as possible from provided context, make sure to provide all the details, 
        if the answer is not available in provided context just say "Answer is not available in provided context", dont
        provide wrong answers.

        Context: \n{context}\n
        Question: \n{question}\n

        Answer:                       
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Fixed prompt initialization
    chain = load_qa_chain(llm = model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True  # Correct usage of return_only_outputs
    )

    st.write("", response["output_text"])
    return response

def main():
    st.set_page_config(
        page_title="Gemini PDF Chatbot",
        page_icon="👾"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with PDF files 🧠🇦🇮👾")
    st.write("Welcome to the chat!")
    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload some pdfs and ask me a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                print(response)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
# import google.generativeai as genai
# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter  # Corrected typo
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.prompts import PromptTemplate  # Corrected typo
# from dotenv import load_dotenv
# from langchain.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain  # Corrected typo in 'chains'

# # Load environment variables
# load_dotenv()
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# def get_pf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200)  # Corrected typo
#     chunks = text_splitter.split_text(text)  # Fixed typo here
#     return chunks

# def get_vector_store(text_chunks):
#     embedding = GoogleGenerativeAIEmbeddings(model="gemini-1.5-embedding")
#     vector_store = FAISS.from_texts(text_chunks, embedding)  # Corrected parameter passing
#     vector_store.save_local("faiss_index")

# def get_conversational_chain():
#     prompt_template = """  # Changed variable name to avoid confusion
#         Answer the questions as detailed as possible from provided context, make sure to provide all the details, 
#         if the answer is not available in provided context just say "Answer is not available in provided context", dont
#         provide wrong answers.

#         Context: \n{context}\n
#         Question: \n{question}\n

#         Answer:                       
#     """
#     model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
#     prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])  # Fixed prompt initialization
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model="gemini-1.5-embedding")
#     new_db = FAISS.load_local("faiss_index", embeddings)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

#     response = chain(
#         {"input_documents": docs, "question": user_question},
#         return_only_outputs=True  # Correct usage of return_only_outputs
#     )

#     print(response)
#     st.write("Reply: ", response["output_text"])

# def main():
#     st.set_page_config("Chat with PDF")
#     st.header("Chat with PDF Using Gemini-Pro")

#     user_question = st.text_input("Ask Question")

#     with st.sidebar:
#         st.title("Menu")
#         pdf_docs = st.file_uploader("Upload PDF file and click on submit", type="pdf", accept_multiple_files=True)  # Added type for clarity
#         if st.button("Submit"):
#             with st.spinner("Preprocessing..."):
#                 raw_text = get_pf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)  # Fixed variable assignment
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()

# genai.configure(api_key="YOUR_API_KEY")
# model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("Explain how AI works")
# print(response.text)