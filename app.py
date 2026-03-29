






import time
import base64
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader







from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser






def process_pdf(pdf_file):
    """Extracts text from a single PDF, chunks it, and generates local embeddings."""
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text



    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base






def main():
    load_dotenv(override=True)
    st.set_page_config(page_title="Automated PDF Analyzer", layout="wide")
    
    #CSSintegration
    st.markdown("""
    <style>
        /* Pitch black background globally */
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
        }
        [data-testid="stHeader"] {
            background-color: #000000;
        }
        
        /* Make file uploader look massive */
        [data-testid="stFileUploader"] > div > div {
            border: 3px dashed #555 !important;
            border-radius: 20px !important;
            padding: 100px 50px !important;
            background-color: #0a0a0a !important;
        }
        
        /* General text coloring to pop on black */
        * { color: #f0f0f0; }
        
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)












    #messagestates
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "pdf_bytes" not in st.session_state:
        st.session_state.pdf_bytes = None
    if "new_query" not in st.session_state:
        st.session_state.new_query = None

    # Function to clear search bar but grab value
    def handle_submit():
        st.session_state.new_query = st.session_state.query_input
        st.session_state.query_input = ""

    #querybar
    #alogn with title
    st.text_input(
        "ASK QUESTIONS FROM YOUR FILE!!!", 
        placeholder="Type your query when ready!", 
        key="query_input", 
        on_change=handle_submit
    )
    
    current_query = st.session_state.new_query
    # Immediately wipe new_query flag so it doesn't double-trigger when interacting with other widgets
    st.session_state.new_query = None

    st.markdown("---")

    #uploader
    if st.session_state.knowledge_base is None:
        st.markdown("<h2 style='text-align: center; color: white;'>Initialize Scanner</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Drop your document into the massive dropzone below.</p>", unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("", type="pdf", label_visibility="collapsed")
        
        if pdf_file is not None:
            with st.spinner("Scanning..."):
                # Save bytes so we can open it alongside later
                st.session_state.pdf_bytes = pdf_file.getvalue()
                st.session_state.knowledge_base = process_pdf(pdf_file)
                st.rerun()

    #pdfchatsplitview
    if st.session_state.knowledge_base is not None:
        
        #reset?
        if st.sidebar.button("Start Over / Upload New"):
            st.session_state.knowledge_base = None
            st.session_state.messages = []
            st.session_state.pdf_bytes = None
            st.rerun()

        col1, col2 = st.columns([1.2, 1])
        
        #docreference
        with col1:
            st.markdown("### Document Reference")
            base64_pdf = base64.b64encode(st.session_state.pdf_bytes).decode('utf-8')
            # Iframe embedding to view PDF in browser
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf" style="border-radius: 10px; border: 1px solid #333; background: #fff;"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)

        #chat
        with col2:
            st.markdown("### chat")
            
            #enter=process
            if current_query:
                # Add question to history
                st.session_state.messages.append({"role": "user", "content": current_query})
                
                with st.spinner("Retrieving facts..."):
                    docs = st.session_state.knowledge_base.similarity_search(current_query)
                    context = "\n\n".join([doc.page_content for doc in docs])

                    # Build conversational memory string
                    chat_history_text = ""
                    # exclude the latest user message we just appended
                    for msg in st.session_state.messages[:-1]:
                        role = "User" if msg["role"] == "user" else "AI"
                        chat_history_text += f"{role}: {msg['content']}\n"

                    prompt = PromptTemplate.from_template(
                        "You are a highly intelligent tutor and assistant.\n"
                        "Use the provided document context and conversation history to inform your answers.\n"
                        "However, if the user asks you to explain topics, restructure concepts, brainstorm, or if the answer isn't fully in the text, you are highly encouraged to use your own general LLM knowledge to provide a comprehensive, structured, and easy-to-understand response!\n\n"
                        "Conversation History:\n{chat_history}\n\n"
                        "Document Context:\n{context}\n\n"
                        "Question: {question}\n\n"
                        "Answer:"
                    )
                    
                    llm = ChatGroq(model="llama-3.1-8b-instant")
                    chain = prompt | llm | StrOutputParser()
                    
                    st.markdown(f"**You:** {current_query}")
                    st.markdown("**AI Response:**")
                    
                    # Stream logic directly to screen
                    response = st.write_stream(chain.stream({
                        "context": context, 
                        "question": current_query,
                        "chat_history": chat_history_text
                    }))
                    
                    # Display sources immediately below
                    sources_list = [d.page_content for d in docs]
                    with st.expander(f"View {len(sources_list)} Source Passages Used"):
                        for i, source in enumerate(sources_list):
                            st.markdown(f"**Source {i+1}:**\n{source}")
                            st.markdown("---")
                    
                    # answersandsources
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response,
                        "sources": sources_list
                    })
            
            # Or if they clicked a button, just show the most recent Q&A to keep it on screen
            elif len(st.session_state.messages) > 0:
                last_msg_user = st.session_state.messages[-2]
                last_msg_ai = st.session_state.messages[-1]
                
                st.markdown(f"**You:** {last_msg_user['content']}")
                st.markdown("**AI Response:**")
                st.write(last_msg_ai['content'])
                
                # Show sources if available
                if "sources" in last_msg_ai:
                    with st.expander(f"View {len(last_msg_ai['sources'])} Source Passages Used"):
                        for i, source in enumerate(last_msg_ai["sources"]):
                            st.markdown(f"**Source {i+1}:**\n{source}")
                            st.markdown("---")

            st.markdown("<br><br>", unsafe_allow_html=True)
            
            #history
            with st.expander("Expand Conversation History", expanded=False):
                if len(st.session_state.messages) <= 2:
                    st.write("No history .")
                else:
                    # Show everything EXCEPT the last two messages (which are already on the screen)
                    # We reverse them so newest history is at the top
                    for msg in reversed(st.session_state.messages[:-2]):
                        if msg["role"] == "user":
                            st.markdown(f"**You:** {msg['content']}")
                        else:
                            st.markdown(f" **AI:** {msg['content']}")
                            if "sources" in msg:
                                st.caption(f"*(Based on {len(msg['sources'])} retrieved document passages)*")
                            st.markdown("---")

























if __name__ == "__main__":
    main()