import telebot
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
from dotenv import load_dotenv
import io
from datetime import datetime
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import base64
from google.cloud import vision
from google.oauth2 import service_account

credentials_path = 'api-key.json'
credentials = service_account.Credentials.from_service_account_file(credentials_path)
client = vision.ImageAnnotatorClient(credentials=credentials)
pkl_folder = "pkl_files"
pdf_folder = "pdf_files"
# Ensure the pkl_folder exists
if not os.path.exists(pkl_folder):
    os.makedirs(pkl_folder)
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)


load_dotenv()
# Initialize the Telegram bot
API_TOKEN = os.getenv('TELE_API')
bot = telebot.TeleBot(API_TOKEN)
user_store_name = None
def delete_pdf(user_store_name):
    pdf_file_path = f"{user_store_name}.pdf"
    if os.path.exists(pdf_file_path):
        os.remove(pdf_file_path)
def extract_text_from_pdf(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Process and store the PDF VectorStore
# Process and store the PDF VectorStore
def process_and_store_pdf(pdf_file, store_name):
    images = convert_from_bytes(pdf_file)
    pdf_text = ""

    extracted_text = []
    for image in images:
        image_bytes = image_to_bytes(image)
        response = extract_text_from_image(image_bytes)
        pdf_text+=response.description

        
    print(pdf_text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=pdf_text)

    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
    else:
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
    return VectorStore
    
def image_to_bytes(image):
    img_byte_array = io.BytesIO()
    image.save(img_byte_array, format='PNG')
    return img_byte_array.getvalue()

def extract_text_from_image(image_bytes):
    image = vision.Image(content=image_bytes)
    response = client.text_detection(image=image)
    return response.text_annotations[0]

# Handle document messages (PDF files)
@bot.message_handler(content_types=['document'])
def handle_document(message):
    global user_store_name

    # Check if the document is a PDF
    if not message.document.mime_type == 'application/pdf':
        bot.reply_to(message, "Please upload a PDF file.")
        return

    # Set user_store_name based on user ID and timestamp
    now = datetime.now()
    user_store_name = f"user_{message.from_user.id}_{now.strftime('%Y%m%d%H%M%S')}"

    # Get the file path for the PDF document
    file_info = bot.get_file(message.document.file_id)
    downloaded_file = bot.download_file(file_info.file_path)
    pdf_path = os.path.join(pdf_folder, user_store_name + ".pdf")
    pdf_file_path = pdf_path
    
    with open(pdf_file_path, "wb") as f:
        f.write(downloaded_file)

    # Extract text from the PDF
    #pdf_text = extract_text_from_pdf(pdf_file_path)
    with open(pdf_file_path, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()
    # Process and store the PDF VectorStore
    pkl_path = os.path.join(pkl_folder, user_store_name + ".pkl")
    process_and_store_pdf(pdf_bytes, pkl_path)
    delete_pdf(pdf_file_path)
    bot.reply_to(message, "PDF file processed. You can now start asking questions.")

# Handle text messages
@bot.message_handler(func=lambda message: True)
def handle_text(message):
    global user_store_name

    # Check if user_store_name is set
    if user_store_name is None:
        bot.reply_to(message, "Please upload a PDF and process it first.")
        return
    query = message.text
    pkl_path = os.path.join(pkl_folder, user_store_name + ".pkl")
    # Load the PDF VectorStore based on the provided user_store_name
    with open(pkl_path + ".pkl", "rb") as f:
        VectorStore = pickle.load(f)

    # Perform the LangChain operations and send the response
    docs = VectorStore.similarity_search(query=query, k=3)
    llm = ChatOpenAI(max_tokens=256)
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)

    bot.reply_to(message, response)

# Start the bot's polling
bot.infinity_polling()
