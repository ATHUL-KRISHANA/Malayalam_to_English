from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from gtts import gTTS
import os
import speech_recognition as sr


app = Flask(__name__)
language = 'en'
# Load model and tokenizer
model_name = "athul-krishna/malayalam-english"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def translate_text(input_text, max_length=128):
    # Get the device (CPU or GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    
    # Generate the translation
    outputs = model.generate(inputs["input_ids"], max_length=max_length)
    
    # Decode the generated output
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text

@app.route('/')
def home():
    return render_template('demo.html')

@app.route('/translate', methods=['POST'])
def translate():
    try:
        text = request.form.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided', 'success': False})
        
        # Translate using the custom model
        translation = translate_text(text)
        tts = gTTS(text=text, lang=language, slow=False)
        # print(translation)
        
        return jsonify({
            'original': text,
            'translation': translation,
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })
import uuid  # For unique filenames

@app.route('/speak', methods=['POST'])
def speak():
    try:
        translation = request.form.get('text', '')
        if not translation:
            return jsonify({'error': 'No text provided', 'success': False})
        
        # Generate audio for the translated text
        tts = gTTS(text=translation, lang=language, slow=False)
        filename = f"{uuid.uuid4().hex}.mp3"
        filepath = os.path.join("static", filename)
        tts.save(filepath)

        return jsonify({
            'audio_url': f"/static/{filename}",
            'success': True
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })
@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    recognizer = sr.Recognizer()
    audio_file = request.files['audio']

    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            # Recognize Malayalam speech
            text = recognizer.recognize_google(audio, language='ml-IN')
            return jsonify({'success': True, 'text': text})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True,port=8000)
