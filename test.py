from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("test2.pdf")
pages = loader.load_and_split()

print(pages[0])