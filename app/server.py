import os
from fastapi import FastAPI
import langserve
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser


app = FastAPI()

llm = ChatGroq(
    temperature=0,
    model_name="mixtral-8x7b-32768",
    groq_api_key=os.environ["GROQ_API_KEY"])

embeddings_model = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=os.environ["COHERE_API_KEY"]
)

faiss_index = FAISS.load_local(
    "./langserve_index", #"../langserve_index",
    embeddings_model,
    allow_dangerous_deserialization=True)

retriever = faiss_index.as_retriever()

prompt_template = """\
Use the provided context to answer the user's question. If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}"""


rag_prompt = ChatPromptTemplate.from_template(prompt_template)

entry_point_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)

rag_chain = entry_point_chain | rag_prompt | llm | StrOutputParser()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

# Add rag chain to a route
add_routes(app, rag_chain, path='/rag')

# Add Mixtral model to a route
add_routes(app, llm, path='/mixtral')

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
