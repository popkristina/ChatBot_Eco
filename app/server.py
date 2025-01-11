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

# TODO: Fix
llm = HuggingFaceEndpoint(
    endpoint_url="https://oolbderhhrn6klkc.us-east-1.aws.endpoints.huggingface.cloud",
    huggingfacehub_api_token=os.environ["GROQ_API"],
    task="text-generation",
)

# Ova e ok
embeddings_model = CohereEmbeddings(
    model="embed-english-v3.0",
)

# Ova e ok
faiss_index = FAISS.load_local("../langserve_index", embeddings_model)
retriever = faiss_index.as_retriever()

# Ova e ok
prompt_template = """\
Use the provided context to answer the user's question. If you don't know the answer, say you don't know.

Context:
{context}

Question:
{question}"""

# TODO: Check
rag_prompt = ChatPromptTemplate.from_template(prompt_template)

# TODO: Check
entry_point_chain = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
# TODO: Check
rag_chain = entry_point_chain | rag_prompt | hf_llm | StrOutputParser()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


# Edit this to add the chain you want to add
add_routes(app, NotImplemented)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
