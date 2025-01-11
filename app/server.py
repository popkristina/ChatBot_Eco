import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from langchain.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser


app = FastAPI()

llm = ChatGroq(temperature=0,
               model_name="mixtral-8x7b-32768",
               groq_api_key="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")

embeddings_model = CohereEmbeddings(
    model="embed-english-v3.0",
    api_key=os.environ["COHERE_API_KEY"]
)

faiss_index = FAISS.load_local("../langserve_index", embeddings_model)
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


# Edit this to add the chain you want to add
add_routes(app, rag_chain, path='/rag')

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
