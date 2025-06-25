from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
import os

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = OpenAI(
    api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)
# client = OpenAI()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

vector_store = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="PdfRAG",
    embedding=embedding_model,
)


while True:
    query = input(" -> ")
    search_result = vector_store.similarity_search(query=query)
    context = "\n\n\n".join(
        [
            f"Page Content: {result.page_content}\nPage Number: {result.metadata['page_label']}\nFile Location: {result.metadata['source']}"
            for result in search_result
        ]
    )
    SYSTEM_PROMPT = f"""
    You are a helpfull AI Assistant who asnweres user query based on the available context
    retrieved from a PDF file along with page_contents and page number.

    You should only ans the user based on the following context and navigate the user
    to open the right page number to know more.

    always answer in plain text , dont use any thing specific like :
    - ***
    - ```javascript
    - any other things that might be present in the document 
    analyse and make the response clean and easy to read
    Context:
    {context}
    """
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    )

    print(response.choices[0].message.content)
