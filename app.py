import os
import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.document_loaders import PyPDFDirectoryLoader


PDF_FILE_PATH = 'docs_pdf/'
CHROMA_PERSIST_DIR = 'chroma/'
RETRIEVAL_MAP = {"mmr": "Maximum Marginal Relevance",
                 "sqr": "Self Query Retrieval", "similarity": "Similarity Score Threshold"}
CHAIN_TYPE_MAP = {"stuff": "Stuff", "map_reduce": "Map Reduce"}


def wipe_pdf_directory():
    for file_name in os.listdir(os.path.join(os.getcwd(), PDF_FILE_PATH)):
        file_path = os.path.join(os.getcwd(), PDF_FILE_PATH, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)


def wipe_chroma_directory():
    if os.path.exists(CHROMA_PERSIST_DIR):
        for file_name in os.listdir(os.path.join(os.getcwd(), CHROMA_PERSIST_DIR)):
            file_path = os.path.join(os.getcwd(), CHROMA_PERSIST_DIR, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)


def save_pdf_to_directory(files):
    file_list = []
    for file in files:
        with open(os.path.join(PDF_FILE_PATH, file.name), mode="wb") as f:
            f.write(file.getbuffer())
        file_list.append(file.name)
    return file_list


def load_models(query_model="mistral", chat_model="llama3"):
    # Instantiate LLM models
    llm = ChatOllama(model=chat_model, temperature=0)
    llm_query = Ollama(model=query_model, temperature=0)
    return llm, llm_query


def load_pdf_to_docs(pdf_file_path="docs_pdf/"):
    # Load PDF file(s) to docs
    loader = PyPDFDirectoryLoader(pdf_file_path)
    docs = loader.load()

    return docs


def split_docs_to_chunks(docs, chunk_size=1000, chunk_overlap=50):
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    return splits


def create_vectordb(splits):

    embedding = HuggingFaceEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=CHROMA_PERSIST_DIR
    )

    return vectordb


def create_self_query_retriever(llm_query, vectordb, document_content_description, file_list):
    file_path_list = [os.path.join(PDF_FILE_PATH, file) for file in file_list]
    metadata_field_info = [
        AttributeInfo(
            name="source",
            description=f"The name of the document. Should be in the list: {file_path_list}",
            type="string",
        ),
        AttributeInfo(
            name="page",
            description="The page from the document",
            type="integer",
        ),
    ]
    retriever = SelfQueryRetriever.from_llm(
        llm_query,
        vectordb,
        document_content_description,
        metadata_field_info,
        verbose=True,
        search_type="mmr"
    )

    return retriever


def create_chain(llm, retriever, chain_type):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa_chain


def run_chain(qa_chain, question, chat_history):
    result = qa_chain.invoke({"question": question, "chat_history": chat_history})
    print(result["source_documents"])  # print this for debugging purposes
    print(result["generated_question"])  # print this for debugging purposes
    return result


# streamlit app main() function
def main():
    # Set page config and title
    st.set_page_config(page_title="Chat with your PDF document(s)",
                       page_icon=":llama:", layout="wide")
    st.title("Chat with your PDF document(s)")
    st.subheader(
        "*Powered by instructor text embeddings, mistral (for self query retrieval), and llama3 (for chat) :sunglasses: *")
    st.subheader("*Note: Refreshing the page will erase all input*")
    st.divider()

    # Initialize file list
    if 'file_list' not in st.session_state:
        st.session_state.file_list = []

    # Initialize document_content_description
    if 'document_content_description' not in st.session_state:
        st.session_state.document_content_description = None

    # Initialize chat_history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Initialize retrieval_type
    if 'retrieval_type' not in st.session_state:
        st.session_state.retrieval_type = None

    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = False

    if 'show_source' not in st.session_state:
        st.session_state.show_source = False

    # Create pdf directory if not exists
    if not os.path.exists(PDF_FILE_PATH):
        os.mkdir(PDF_FILE_PATH)

    with st.sidebar:
        with st.form("Upload PDF(s)", border=False):
            if st.session_state.uploaded == True:
                disabled = True
                chunk_size = st.session_state.chunk_size
                chunk_overlap = st.session_state.chunk_overlap

                files = st.file_uploader("Upload the PDF file(s) you would like to chat with:", type=[
                                         "pdf"], accept_multiple_files=True, disabled=disabled)

                st.selectbox("Vector Database Retrieval Method", [
                             RETRIEVAL_MAP[st.session_state.retrieval_type]], disabled=True)

            else:
                disabled = False
                chunk_size = 1024
                chunk_overlap = 128
                # Input pdf file path
                files = st.file_uploader("Upload the PDF file(s) you would like to chat with:", type=[
                                         "pdf"], accept_multiple_files=True)

                st.session_state.retrieval_type = st.selectbox("Vector Database Retrieval Method", RETRIEVAL_MAP.keys(
                ), format_func=lambda x: RETRIEVAL_MAP[x], placeholder="Choose an option")
                st.session_state.document_content_description = st.text_input("Brief description of the PDF file(s) contents:",
                                                                              max_chars=50, disabled=disabled,
                                                                              help="Only required for self query retrieval*")
            st.session_state.chunk_size = st.slider("Chunk Size", min_value=512, max_value=4096, step=32, value=chunk_size, disabled=disabled,
                                                    help="""The maximum number of characters in a chunk, 
                                                    to be used for document splitting in vector database generation. 
                                                    Adjusting this value may affect the quality of answers. Default value:1000""")
            st.session_state.chunk_overlap = st.slider("Chunk Overlap", min_value=48, max_value=512, step=16, value=chunk_overlap,  disabled=disabled,
                                                       help="""The number of characters to overlap between chunks, 
                                                    to be used for document splitting in vector database generation. 
                                                    Adjusting this value may affect the quality of answers. Default value:50""")

            uploaded = st.form_submit_button("Upload", disabled=disabled)
            if uploaded:
                if len(files) == 0:
                    uploaded = False
                    st.warning("Please upload at least 1 file!")
                if st.session_state.retrieval_type == None:
                    uploaded = False
                    st.warning("Please select a vector database retrieval method!")
                if st.session_state.retrieval_type == "sqr" and len(st.session_state.document_content_description) == 0:
                    uploaded = False
                    st.warning("Please input the document content description!")
                if uploaded:
                    with st.spinner("Creating vector database..."):
                        wipe_pdf_directory()  # Clear the directory so we only process the files uploaded this time
                        wipe_chroma_directory()  # Clear the vectordb so we only process the files uploaded this time
                        st.session_state.vectordb = None  # Clear the vectordb first
                        st.session_state.uploaded = True
                        st.session_state.file_list = save_pdf_to_directory(files)
                        docs = load_pdf_to_docs(PDF_FILE_PATH)
                        splits = split_docs_to_chunks(
                            docs, chunk_size=st.session_state.chunk_size, chunk_overlap=st.session_state.chunk_overlap)
                        st.session_state.vectordb = create_vectordb(splits)
                        # print(st.session_state.vectordb._collection.count())
                    st.success("Finished loading data into vector database!")
                    st.rerun()

    if st.session_state.file_list:
        st.sidebar.write("**Retrieval method:**", RETRIEVAL_MAP[st.session_state.retrieval_type])
        st.sidebar.write("**Chunk size:**", st.session_state.chunk_size)
        st.sidebar.write("**Chunk Overlap:**", st.session_state.chunk_overlap)
        st.sidebar.write("**Uploaded files:**")
        for file in st.session_state.file_list:
            st.sidebar.write(file)
        if st.session_state.document_content_description:
            st.sidebar.write("**PDF file content description:**",
                             st.session_state.document_content_description)
        chain_type = st.sidebar.selectbox(
            "Chain Type", CHAIN_TYPE_MAP.keys(), format_func=lambda x: CHAIN_TYPE_MAP[x])
        st.session_state.show_source = st.sidebar.toggle("Show answer source")
        # Display warning message
        if "map_reduce" in chain_type:
            st.sidebar.warning(
                f"Map reduce chain type may take several minutes to generate answer due to prompt latency!")

    for turn in st.session_state["chat_history"]:
        st.write("**User:**", turn[0])
        st.write("**Bot:**", turn[1])

    # Prompt input
    question = st.text_input("Enter prompt:", disabled=not disabled,
                             placeholder="Ask something about the documents")

    # Enter button
    if st.button("Enter", disabled=not disabled):
        with st.spinner("Processing"):
            llm, llm_query = load_models()
            if st.session_state.retrieval_type == "sqr":
                retriever = create_self_query_retriever(
                    llm_query, st.session_state.vectordb, st.session_state.document_content_description, st.session_state.file_list)
            elif st.session_state.retrieval_type == "mmr":
                retriever = st.session_state.vectordb.as_retriever(search_type="mmr")
            else:
                retriever = st.session_state.vectordb.as_retriever(search_type="similarity_score_threshold", search_kwargs={
                                                                   "score_threshold": 0.1})  # Use similarity score threshold search
            qa_chain = create_chain(llm, retriever, chain_type)
            # Run qa chain
            result = run_chain(qa_chain, question, st.session_state["chat_history"])
            st.session_state["chat_history"].extend(
                [(question, result["answer"], result["source_documents"])])

            print(st.session_state["chat_history"])

        st.rerun()

    if st.session_state.show_source:
        st.write("Answer source:", st.session_state["chat_history"][-1][-1])


if __name__ == "__main__":
    main()
