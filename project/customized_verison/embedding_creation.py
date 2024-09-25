from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

class EmbeddingCreation():
    def __init__(self) -> None:
     pass
    
    #laods the embedding model 
    def intialize_emb(self,model_path):
        """
        Intializes the embedding model
        """
        embedding=HuggingFaceEmbeddings(model_name = model_path)
        #returning model intiation for embedding creation
        return embedding


    def create_vector_index(self,documents_path,vector_db_path,embedding_model_path,chunksize,chunkoverlap):
        """
        creates the vector database
        """

        loader = DirectoryLoader(documents_path, glob="./*.txt", loader_cls=TextLoader)
        documents1 = loader.load()
    

        loader = DirectoryLoader(documents_path, glob="./*.pdf", loader_cls=PyPDFLoader)
        documents2 = loader.load()

        whole_docs = documents1+documents2
        print("whole documents",whole_docs)

        #splitting the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size= chunksize, chunk_overlap=chunkoverlap)
        texts = text_splitter.split_documents(whole_docs)
        #getting folderpath and intitaing the persistant directory to save the embedding database
        
        
        embedding = self.intialize_emb(embedding_model_path)
        global vectordb
        vectordb= Chroma.from_documents(documents=texts,
                                        embedding=embedding,
                                        persist_directory=vector_db_path)
        
        vectordb.persist()
        print("created db ")