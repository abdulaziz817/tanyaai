# Import library yang dibutuhkan
import gradio as gr  # Gradio untuk antarmuka web
import os  # Mengakses variabel lingkungan
from langchain.chains import ConversationChain  # Untuk membuat rantai percakapan
from langchain.chains.conversation.memory import ConversationBufferWindowMemory  # Memori percakapan
from langchain_groq import ChatGroq  # Model LLM "GROQ" untuk percakapan
from dotenv import load_dotenv  # Untuk memuat variabel lingkungan dari file .env


# Memuat variabel lingkungan dari file .env
load_dotenv()  # Membaca file .env yang menyimpan variabel sensitif seperti kunci API

# Mengambil kunci API GROQ dari variabel lingkungan
groq_api_key = os.environ['GROQ_API_KEY']


# Fungsi untuk inisialisasi percakapan menggunakan model LLM GROQ
def initialize_conversation():
    # Inisialisasi memori percakapan agar chatbot bisa "mengingat" konteks dialog sebelumnya
    memory = ConversationBufferWindowMemory()

    # Membuat objek ChatGroq dengan API key, model, dan pengaturan
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,  # Menggunakan kunci API untuk autentikasi
        model_name="llama3-8b-8192",  # Nama model yang digunakan
        temperature=1  # Kreativitas respons (semakin tinggi, semakin kreatif/jelas jawabannya)
    )

    # Mengembalikan objek ConversationChain yang menggabungkan model dan memori
    return ConversationChain(llm=groq_chat, memory=memory)


# Membuat objek conversation untuk mengelola alur percakapan
conversation = initialize_conversation()


# Fungsi untuk menangani pertanyaan dari pengguna
def chatbot(user_question):
    try:
        # Mengirim pertanyaan ke model dan mendapatkan respons
        response = conversation(user_question)
        # Mengembalikan respons yang dihasilkan oleh model
        return response['response']
    except Exception as e:  # Jika ada error
        # Mengembalikan pesan error agar mudah dilihat
        return f"Terjadi error: {e}"


# Membuat antarmuka dengan Gradio
iface = gr.Interface(
    fn=chatbot,  # Fungsi backend yang dipanggil ketika pengguna mengirim pertanyaan
    inputs="textbox",  # Input berupa kotak teks
    outputs="textbox",  # Output berupa kotak teks
    title="Groq Chat App",  # Judul aplikasi
    description="Ask a question and get a response."  # Deskripsi aplikasi
)

# Menjalankan aplikasi sehingga dapat diakses melalui browser
iface.launch()
