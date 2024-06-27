# Chat with your PDF document(s)

![demo-screen](https://github.com/SarahWSY/rag-webapp/assets/47151064/c239c0f4-75d3-4f50-b085-f95b91e5ad0e)


#### Description: 
This project is a web application which allows the user to “chat with” and quickly extract information from one or more of their PDF documents. It also allows some flexibility for the user to choose their preferred retrieval method (maximal marginal relevance/self query retrieval/similarity threshold) and prompting strategy (stuff/map reduce) which may work better for their data. The default options are Maximal Marginal Relevance for retrieval and Stuff for prompting strategy. Users also have the option to modify the chunk size and chunk overlap which are set to 1024 and 128 by default.

#### Technical Details: 
The application was built using LangChain for the primary functionality and Streamlit for the web application front-end. 

**LangChain** is a framework for building applications powered by LLMs. It offers modular components and various chains, which can be easily modified and implemented. It also has many third-party integrations for using models, vector database retrieval and agent tools. This application uses the third-party integration of Ollama to enable local usage of LLMs. It uses 2 models – Mistral (for self query retrieval) and Llama3 (for chat). It uses the third-party integrated PyPDFDirectoryLoader to load in PDF files from a given directory as Document objects. It then uses a LangChain RecursiveCharacterTextSplitter to split the documents into chunks. The chunks are then embedded with HuggingFace embeddings (based on Instructor model) and then stored in a Chroma vector database. The application allows users to select the retrieval method to be base on maximal marginal relevance (MMR), similarity threshold, or self query retrieval. For MMR and similarity threshold, the retriever is created from the vector database using the as_retriever() method. For self query retrieval, the retriever is created with LangChain’s SelfQueryRetriever. For the QA chain, the Langchain ConversationalRetrievalChain is used as it can take in the conversation history to reformulate the query into one that also considers chat history, perform retrieval from the vector database and finally query the LLM using the selected prompting strategy (stuff or map reduce in this application). 

**Streamlit** was selected as the front-end framework as this web application only required a simple single-page site as the chatbot interface. A Streamlit form is used on the left sidebar to collect the necessary user input. When the user clicks “Upload”, the files which were uploaded using Streamlit’s file uploader will be read and saved to the server’s pdf file directory, which will be accessed subsequently to create the vector database. The other variables such as vector database retrieval method, document content description, chunk size and chunk overlap are stored as Streamlit session state attributes for later access. In the main section of the page, the user can input a question about the documents. The conversation questions and answers from each turn will be stored in a session state attribute called "chat_history" and displayed on the main page. Once there is at least 1 conversation turn, a toggle button will appear on the left sidebar, which allows the user to view the source chunks of the current answer and the document and page it was taken from.

#### Usage: 
1. Upload the PDF file(s) on the left sidebar. 
2. Select the vector database retrieval method in the drop down list. 
3. Input a brief description of the uploaded document(s). 
4. Adjust chunk size and chunk overlap (optional) 
5. Click on 'Upload' to create the vector database. 
6. Enter your question on the main page text box and chat with your document(s). 

*Note: If you would like to view the part of the source document(s) which the answer was derived from, you can activate the "Show answer source" toggle button on the left sidebar and the raw source chunks will be displayed on the main page below the "Enter" button.*

#### Limitations:
This application does not have to ability to answer questions unrelated to the uploaded documents or conduct general conversation with users. It is only able to answer questions that are related to the document(s) uploaded, whether in single-turn or multiple turn exchanges. 

Also, for self query retrieval I have used the Mistral model instead of the recommended OpenAI gpt-3.5-turbo-instruct model to avoid incurring API costs so the self query retrieval may fail in some cases. 

#### Installation:
The required packages can be installed using the following commands (alternatively use the requirements.txt file to install dependencies):

`pip install streamlit`
`pip install langchain`
`pip install langchain-community`
`pip install pypdf`
`pip install pycryptodome==3.15.0`
`pip install sentence-transformers`
`pip install chromadb`
`pip install lark`

To run the application, use the command below:

`streamlit run app.py`

