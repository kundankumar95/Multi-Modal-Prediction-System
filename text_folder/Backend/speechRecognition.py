import speech_recognition as sr

def convert_speech_to_text(audio_file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file_path) as source:
        print("Listening...")
        audio_data = recognizer.record(source)
        print("Recognizing...")
        text = recognizer.recognize_google(audio_data)
        return text

