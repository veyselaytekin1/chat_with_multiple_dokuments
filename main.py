import streamlit as st
import langchain 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI, VectorDBQA
from langchain.chains import RetrievalQAWithSourcesChain
import PyPDF2

# This function will parse pdf and extract and return list of page texts.
def read_and_textify(files):
    text_list = []
    sources_list = []
    for file in files:
        pdf_reader = PyPDF2.PdfReader(file)
        #print("Page Number: ", len(pdfReader.pages))
        for i in range(len(pdf_reader.pages)):
            page_obj = pdf_reader.pages[i]
            text = page_obj.extract_text()
            page_obj.clear()

            text_list.append(text)
            sources_list.append(file.name + '_page' + str(i))
    return [text_list, sources_list]

st.set_page_config(layout="centered", page_title='Chat mit Ihre PDF')

tabs= ["Chat","PDF Files","About me"]

page = st.sidebar.radio("Tabs",tabs)

if page == "Chat":
    # Custom CSS to create a rainbow divider effect
    st.markdown("""
        <style>
        .rainbow {
            height: 4px;
            background: linear-gradient(to right, violet, indigo, blue, green, yellow, orange, red);
        }
        h1 {
            text-align: center;
        }
        </style>
        """, unsafe_allow_html=True)
    # Your header
    st.header("Chat mit deine PDFs")
    # Rainbow divider
    st.markdown('<div class="rainbow"></div>', unsafe_allow_html=True)
    st.write("---")
    # file uploader
    uploaded_files = st.file_uploader('Sie können Ihre PDFs hochladen.', accept_multiple_files=True, type=["text", "pdf"])
    st.write("---")

    
    if uploaded_files is None:
        st.info(f"""Bitte laden Sie Ihre PDF hoch.""")
    elif uploaded_files:
        if len(uploaded_files) == 1:
            st.write(str(len(uploaded_files)) + " Dokument ist hochgeladen")
        else:
            st.write(str(len(uploaded_files)) + " Dokumente sind hochgeladen")
        
        textify_output = read_and_textify(uploaded_files)
        documents = textify_output[0]
        sources = textify_output[1]

        # extract embeddings
        embeddings = OpenAIEmbeddings(openai_api_key = st.secrets['openai_api_key'])

        # vector_store with metadata. Here we will store page number.
        vector_store = Chroma.from_texts(documents, embeddings, metadatas = [{"source": s} for s in sources])

        #deciding model
        model_name = "gpt-4"

        retriver = vector_store.as_retriver()
        retriver.search_kwargs = {'k':2 }

        # initiate model
        llm = OpenAI(model_name = model_name, openai_api_key= st.secrets["openai_api_key"], streaming=True)
        model = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type='stuff', retriver=retriver)

        st.header('Frage an deine PDFs')
        user_q = st.text_area("Geben Sie hier Ihre Frage ein")

        if st.button("Antwort erhalten"):
            try:
                with st.spinner('LLM Modell arbeitet daran...'):
                    result = model({"question" : user_q}, return_only_outputs =True)
                    st.subheader('Deine Antwort: ')
                    st.write(result['answer'])
                    st.subheader("Source Seiten: ")
                    st.writer(result['sources'])
            except Exception as e:
                st.error(f"Ein Problem ist aufgetreten")
                st.error("Hoppla, die GPT-Antwort führte zu einem Fehler : Bitte versuchen Sie es mit einer anderen Frage noch einmal.")































