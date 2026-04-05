#to run type streamlit run app.py







import time
import io
import base64
from dotenv import load_dotenv
import streamlit as st
import fitz
import easyocr
import numpy as np
from streamlit_pdf_viewer import pdf_viewer







from langchain_text_splitters import CharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage






@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'], gpu=False)


def vision_transcribe_page(pix) -> str:
    """Use Groq vision LLM to transcribe a page image — handles handwriting well."""
    try:
        load_dotenv(override=True)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        vision_llm = ChatGroq(model="llama-3.2-11b-vision-preview")
        msg = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    "You are a precise document transcription assistant. "
                    "Transcribe ALL text visible in this image exactly as written, "
                    "including handwriting, printed text, numbers, and special characters. "
                    "Preserve line breaks and structure. Output ONLY the transcribed text, nothing else."
                )
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"}
            }
        ])
        result = vision_llm.invoke([msg])
        return result.content.strip()
    except Exception as e:
        return ""


def process_pdf(pdf_bytes):
    """Extracts text from a PDF using a 3-tier strategy:
    1. Native text extraction (fastest, for digital PDFs)
    2. EasyOCR (for scanned/printed documents)
    3. Groq Vision LLM fallback (best for handwriting)
    """
    text = ""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # --- Tier 1: Native text extraction ---
    page_texts = []
    for page in doc:
        page_text = page.get_text()
        page_texts.append(page_text.strip() if page_text else "")
        if page_text and page_text.strip():
            text += page_text + "\n"

    # --- Tier 2 & 3: OCR + Vision LLM for pages with no/little text ---
    pages_needing_ocr = [i for i, t in enumerate(page_texts) if not t]

    if pages_needing_ocr or not text.strip():
        try:
            reader = get_ocr_reader()
        except Exception:
            reader = None

        for page_idx in pages_needing_ocr:
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=200)

            # Tier 2: Try EasyOCR first (faster for printed text)
            easyocr_text = ""
            if reader:
                try:
                    img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                    if pix.n == 4:
                        img_data = img_data[:, :, :3]
                    results = reader.readtext(img_data, detail=0)
                    easyocr_text = "\n".join(results).strip()
                except Exception:
                    easyocr_text = ""

            if easyocr_text and len(easyocr_text) > 30:
                # EasyOCR got a reasonable result — use it
                text += easyocr_text + "\n"
            else:
                # Tier 3: Vision LLM fallback — handles handwriting
                vision_text = vision_transcribe_page(pix)
                if vision_text:
                    text += vision_text + "\n"
                elif easyocr_text:
                    # Vision failed but EasyOCR had something (even if sparse)
                    text += easyocr_text + "\n"

    doc.close()

    if not text.strip():
        raise RuntimeError("No text could be extracted from this PDF. It may be empty or corrupted.")

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    if not chunks:
        raise RuntimeError("Document text split resulted in zero chunks.")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    return knowledge_base






def main():
    load_dotenv(override=True)
    st.set_page_config(page_title="Automated PDF Analyzer", layout="wide")
    
    #CSSintegration
    st.markdown("""
    <style>
       
        [data-testid="stAppViewContainer"] {
            background-color: #000000;
        }
        [data-testid="stHeader"] {
            background-color: #000000;
        }
        
        
        [data-testid="stFileUploader"] > div > div {
            border: 3px dashed #555 !important;
            border-radius: 20px !important;
            padding: 100px 50px !important;
            background-color: #0a0a0a !important;
        }
        
        
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
    if "llm" not in st.session_state:
        try:
            st.session_state.llm = ChatGroq(model="llama-3.1-8b-instant")
        except Exception as e:
            st.warning(f"LLM initialization issue: {e}")
            st.session_state.llm = None

    # clearsearchbarandgrabvalue
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
    if st.session_state.pdf_bytes is None:
        st.markdown("<h2 style='text-align: center; color: white;'>Initialize Scanner</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #888;'>Drop your document into the massive dropzone below.</p>", unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("", type="pdf", label_visibility="collapsed")
        
        if pdf_file is not None:
            with st.spinner("Scanning..."):
                # Save bytes so we can open it alongside later
                st.session_state.pdf_bytes = pdf_file.getvalue()
                try:
                    st.session_state.knowledge_base = process_pdf(st.session_state.pdf_bytes)
                    st.success("PDF processed successfully.")
                except Exception as e:
                    st.error(f"Failed to process PDF: {e}")
                    st.session_state.knowledge_base = None
                st.rerun()


    #pdfchatsplitview
    if st.session_state.pdf_bytes is not None:
        
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
            if st.session_state.pdf_bytes:
                pdf_viewer(input=st.session_state.pdf_bytes, height=800)
            else:
                st.warning("PDF content is not available for viewing right now.")

        #chat
        with col2:
            st.markdown("### chat")
            
            #enter=process
            if st.session_state.knowledge_base is None:
                st.error("AI is unavailable for this document. It may be an image-only PDF and system OCR tools are missing. But you can still safely view it on the left.")
            elif current_query:
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
                    
                    if st.session_state.llm is None:
                        st.error("LLM is unavailable. Please check connection or model setup.")
                        return
                    chain = prompt | st.session_state.llm | StrOutputParser()
                    
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
            
            
            elif len(st.session_state.messages) >= 2:
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
            elif len(st.session_state.messages) == 1:
                st.markdown("**You:** " + st.session_state.messages[0]['content'])
                st.markdown("**AI Response:** Waiting for AI response...")
            else:
                st.info("Ask a question to start the conversation.")

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