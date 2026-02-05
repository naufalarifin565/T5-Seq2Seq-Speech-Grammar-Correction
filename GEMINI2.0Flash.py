# Imports: Library untuk STT, grammar correction, dan utilitas
import google.generativeai as genai         # Google Gemini API client
from happytransformer import HappyTextToText, TTSettings  # Wrapper T5 + settings
import sounddevice as sd                     # Rekam audio dari mikrofon
from scipy.io.wavfile import write           # Simpan rekaman sebagai WAV
import os                                    # Operasi file system
import time                                  # Delay/wait handling

# Configuration: API keys & model setup
# Konfigurasi Gemini 2.0 STT 
GOOGLE_API_KEY = "AIzaSyCYHojfGvnB7X6qCAzN_wR_u0Eq4xIQi8c"
MODEL_ID        = "models/gemini-2.0-flash"
LANGUAGE_PROMPT = "Transcribe the following audio in English:"

# Grammar Correction Model (T5)
happy_tt    = HappyTextToText("T5", "saved_model/")  # Load fine-tuned T5 dari folder
tt_settings = TTSettings(max_length=20)              # Batasi output hingga 20 token


# Function: record_audio
def record_audio(filename="temp_recording.wav", duration=5, fs=44100):
    """
    Rekam audio dari mikrofon selama `duration` detik
    dan simpan sebagai WAV di `filename`.
    """
    print(f"\n🎤 Start speaking... Recording for {duration} seconds.")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, recording)
    print(f"✅ Recording saved as {filename}")
    return filename

# Function: transcribe_audio_gemini
def transcribe_audio_gemini(file_path):
    """
    Upload file WAV ke Google Gemini API, terima hasil
    transkripsi sebagai teks, lalu hapus resource di server.
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {os.path.abspath(file_path)}")
        return None

    try:
        genai.configure(api_key=GOOGLE_API_KEY)

        print("⬆️ Uploading audio file to Google AI...")
        audio_file_resource = genai.upload_file(path=file_path)
        time.sleep(2)

        print("📝 Requesting transcription from Gemini API...")
        model    = genai.GenerativeModel(model_name=MODEL_ID)
        response = model.generate_content([LANGUAGE_PROMPT, audio_file_resource])

        # Hapus resource file di server untuk efisiensi
        try:
            print("🗑️ Deleting uploaded file resource...")
            genai.delete_file(name=audio_file_resource.name)
        except Exception as delete_err:
            print(f"⚠️ Warning: Could not delete uploaded file: {delete_err}")

        return response.text

    except Exception as e:
        print(f"⚠️ Error during transcription: {e}")
        if 'audio_file_resource' in locals() and audio_file_resource:
            try:
                print("🗑️ Attempting to delete file resource after error...")
                genai.delete_file(name=audio_file_resource.name)
            except Exception as delete_err:
                print(f"⚠️ Warning: Could not delete after error: {delete_err}")
        return None

# Function: grammar_correction
def grammar_correction(text):
    """
    Koreksi grammar menggunakan HappyTextToText (T5).
    Gunakan TTSettings untuk membatasi panjang output
    dan ambil hanya kalimat pertama hasil koreksi.
    """
    print("✏️ Correcting grammar using T5 model...")
    result = happy_tt.generate_text(f"grammar: {text}", args=tt_settings)
    
    # Post-processing: hanya kalimat pertama
    corrected_text = result.text.split(".")[0].strip() + "."
    return corrected_text

# Main Execution
if __name__ == "__main__":
    # 1️⃣ Rekam audio langsung (durasi 6 detik)
    audio_file = record_audio(duration=6)

    # 2️⃣ Transkripsi ke teks (STT Gemini 2.0)
    transcription = transcribe_audio_gemini(audio_file)

    if transcription:
        print("\n--- 📝 Raw Transcription (English) ---")
        print(transcription)

        # 3️⃣ Koreksi grammar
        corrected_text = grammar_correction(transcription)
        print("\n--- ✅ Grammar Corrected Text ---")
        print(corrected_text)
        print("-----------------------------------")
    else:
        print("\n❌ Transcription failed.")
