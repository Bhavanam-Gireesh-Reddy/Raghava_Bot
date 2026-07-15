from langchain_ollama import ChatOllama


def generate_response(prompt):

    llm = ChatOllama(
        model="llama3.2:1b",
        temperature=0,
        num_predict = 100
    )

    response = llm.invoke(prompt)

    return response.content