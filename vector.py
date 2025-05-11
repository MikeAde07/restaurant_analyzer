#Vector search
#Vectorzing our documents. Embedding our documents and then looking them up
#Embedding model can take text and convert it into a vector. This is essentially numbers that we can use to lookup data efficiently 
#Chroma is our vector store
#Create documents and then pass these to our Chroma database
from langchain_ollama import OllamaEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document 
import os
import pandas as pd 

#load in csv file
df = pd.read_csv("realistic_restaurant_reviews.csv")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")


#file location to store vector database
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)


if add_documents :
    documents = []
    ids = []

    for i, row in df.iterrows() :
        document = Document(
            #info to query the data
            page_content = row["Title"] + " " + row["Review"],
            #additional info attached with the data but not used for querying
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id =str(i)
        )
        ids.append(str(i))
        documents.append(document)

#create vector store
vector_store = Chroma(
    #specify location and collection
    collection_name = "restaurant_reviews",
    #permanent store
    persist_directory = db_location,
    embedding_function = embeddings

)

if add_documents :
    vector_store.add_documents(documents=documents, ids=ids)

#retriever allows us to lookup documents and we can pass the documents into the prompt for our LLM
retriever = vector_store.as_retriever(
    search_kwargs = {"k" : 5}
)