from generation.ollama_gen import generate_response

gr = generate_response("What is Machine Learning?")
print(gr)