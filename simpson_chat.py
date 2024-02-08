from openai import OpenAI
import streamlit as st
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json


# Hier laden wir die Labels aus einer JSON-Datei
def load_labels(path='labels.json'):
    with open(path, 'r') as f:
        labels = json.load(f)
    return labels


# Hier speichern wir die Labels in einer Variable
labels = load_labels()

# Hier laden wir das Modell, das wir für die Simpsons verwenden
model = load_model('simpsons_20_epochen.h5')


# Diese Funktion klassifiziert ein Bild
def classify_image(uploaded_file):
    # Hier laden wir das Bild und bereiten es vor
    img = image.load_img(uploaded_file, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array /= 255.0  # Hier normalisieren wir das Bild wie im Training
    img_batch = np.expand_dims(img_array, axis=0)

    # Hier klassifizieren wir das Bild
    prediction = model.predict(img_batch)
    predicted_class_index = np.argmax(prediction)
    predicted_class = labels[predicted_class_index]
    predicted_probability = prediction[0][predicted_class_index]

    # Hier zeigen wir die Klasse mit der höchsten Wahrscheinlichkeit an
    st.write(f"Vorhersage: {predicted_class}")
    st.write(f"Wahrscheinlichkeit: {predicted_probability * 100:.2f}%")

    # Hier zeigen wir das Bild an
    st.image(img, caption='Hochgeladenes Bild.')

    # Hier geben wir die vorhergesagte Klasse zurück
    return predicted_class


# Hier definieren wir eine Klasse für die Nachrichten der Charaktere
class CharacterMessages:
    def __init__(self, name, avatar, message):
        self.name = name
        self.avatar = avatar
        self.message = message


# Diese Funktion zeigt den Chatverlauf an
def show_chat_history():
    if 'chat_history' in st.session_state:
        for msg in st.session_state.chat_history:
            st.chat_message(msg.name, avatar=msg.avatar).write(msg.message)


# Diese Funktion fügt eine Nachricht hinzu
def add_message(msg):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    st.session_state.chat_history.append(msg)


# Diese Funktion generiert Nachrichten
def generate_messages(prompt):
    messages = []
    if 'chat_history' in st.session_state:
        for msg in st.session_state.chat_history:
            m = {"role": msg.name, "content": msg.message}
            messages.append(m)
    u = {"role": "user", "content": prompt}
    messages.append(u)
    return messages


# Diese Funktion holt die Antwort des Bots
def get_bot_response(client, prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=generate_messages(prompt)
    )
    return response.choices[0].message.content


# Hier beginnt die Hauptfunktion
def main():
    # Hier holen wir den API-Schlüssel
    APIKEY = st.secrets["openai_api_key"]
    if APIKEY is None:
        raise ValueError("Die Umgebungsvariable OPENAI_API_KEY ist nicht gesetzt.")
        return

    # Hier erstellen wir den Client
    client = OpenAI(api_key=APIKEY)
    st.write("Your Simpson Character")

    # Hier laden wir die Datei hoch
    uploaded_file = st.file_uploader("Bitte einen Simpson Character hochladen", type=["jpg", "png"])
    bot_name = "Assistant"

    classify = st.button("Klassifikation starten")
    if classify and uploaded_file is not None:
        bot_name = classify_image(uploaded_file)
        add_message(CharacterMessages("system", "👩", f"You are a simpson character: {bot_name}. You act as this character."))

    # Hier setzen wir den Reset-Knopf
    reset = st.button("Zurücksetzen")

    # Wenn der Reset-Knopf gedrückt wird, löschen wir den Chatverlauf
    if reset:
        st.session_state.chat_history = []
        st.rerun()

    # Hier zeigen wir den Chatverlauf an
    show_chat_history()

    # Hier geben wir den Prompt für den Chat ein
    prompt = st.chat_input("Frage dein Simpson-Charakter etwas.")

    # Wenn es einen Prompt gibt, fügen wir die Nachrichten hinzu und holen die Antwort des Bots
    if prompt:
        add_message(CharacterMessages("user", "👩‍💻", prompt))
        bot_response = get_bot_response(client, prompt)
        add_message(CharacterMessages("assistant", "🤖", f"{bot_response}"))
        st.rerun()


# Hier starten wir das Programm
if __name__ == "__main__":
    main()
