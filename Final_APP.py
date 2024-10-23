import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from collections import Counter
import torch
from nltk.corpus import wordnet
import nltk
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException
from gtts import gTTS
from io import BytesIO
import tempfile
import PyPDF2
import docx
from io import BytesIO

nltk.download('wordnet')

# Load BART tokenizer and model for summarization
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Check if GPU is available and use it if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Function to summarize key points
def extract_main_points(text, num_points=5):
    summary = summarize_bart(text, max_length=150, min_length=80)
    sentences = summary.split('. ')
    points = [f"• {sentence.strip()}." for sentence in sentences[:num_points] if sentence.strip()]
    return "\n".join(points)

# Function to extract key nouns and adjectives from the text
def extract_key_terms(text, num_concepts=10):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    relevant_words = [word for word, pos in pos_tags if pos in ["NN", "JJ"] and len(word) > 3]
    most_common_words = [word for word, _ in Counter(relevant_words).most_common(num_concepts)]
    return most_common_words

# Function to generate a list of concepts with their definitions (for English input)
def extract_concepts_with_definitions(text, num_concepts=10):
    try:
        language = detect(text)
    except LangDetectException:
        return "Could not detect the language of the text."
    
    # Ensure this feature only works for English input
    if language != 'en':
        return "The 'Concepts List' feature only works with English input."

    key_terms = extract_key_terms(text, num_concepts)
    definitions = []
    
    for term in set(key_terms):
        synsets = wordnet.synsets(term)
        if synsets:
            definition = synsets[0].definition()
            definitions.append(f"• {term} = {definition.strip()}")
        else:
            continue

    if not definitions:
        return "No definitions found."

    return "\n".join(definitions)

# Function to translate the summarized text
def translate_summary(text, target_lang):
    if target_lang != "Original":  # Only translate if a language other than "Original" is selected
        translator = GoogleTranslator(source="en", target=target_lang)
        translated_text = translator.translate(text)
        return translated_text
    return text  # If "Original" is selected, return the original text

# Function to summarize the text using the BART model
def summarize_bart(input_text, max_length, min_length):
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=min_length, do_sample=False)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# Function to read DOCX files
def read_docx(file):
    doc = docx.Document(file)  # Create a Document object from the uploaded file
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

# Function to read PDF files
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    return text

# Updated generate_summary function to handle file reading
def generate_summary(input_text, url, file, format_type, source, target_lang):
    if source == "Text Input":
        content = input_text
    elif source == "Web Page URL":
        content = url
    elif source == "File Upload" and file is not None:
        # Check file extension to determine the file type
        if file.name.endswith('.pdf'):
            content = read_pdf(file)
        elif file.name.endswith('.docx'):
            content = read_docx(file)
        else:
            return "Unsupported file type.", content
    else:
        content = ""

    if not content:
        return "No content to summarize.", content

    try:
        if format_type == "Main Points":
            summary = extract_main_points(content)
        elif format_type == "Concepts List (Only works with English Input)":
            summary = extract_concepts_with_definitions(content)
        elif format_type == "Short Summary":
            summary = summarize_bart(content, max_length=80, min_length=40)
        elif format_type == "Medium Summary":
            summary = summarize_bart(content, max_length=200, min_length=100)
        elif format_type == "Long Summary":
            summary = summarize_bart(content, max_length=400, min_length=200)
        else:
            summary = content

        # Translate the summary to the selected language if needed
        translated_summary = translate_summary(summary, target_lang)
        
    except Exception as e:
        return f"Error during summarization: {str(e)}", content

    return translated_summary, content


# Language map for TTS (adjust if needed)
tts_language_map = {
    'en': 'en',  # English
    'es': 'es',  # Spanish
    'fr': 'fr',  # French
    'de': 'de',  # German
    'fi': 'fi',  # Finnish
    # Add more mappings as needed
}

# Text-to-speech function, now with language detection and automatic TTS language adjustment
def text_to_speech(input_text, summary_text, summary_generated):
    text_to_read = summary_text if summary_generated else input_text
    if not text_to_read.strip():
        return None  # No text to convert to speech

    try:
        # Detect the language of the text
        detected_lang = detect(text_to_read)
        tts_lang = tts_language_map.get(detected_lang, 'en')  # Default to English if not supported

        # Generate speech using gTTS in the detected language
        tts = gTTS(text_to_read, lang=tts_lang)
        audio_file = BytesIO()
        tts.write_to_fp(audio_file)
        audio_file.seek(0)

        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            return temp_audio_file.name  # Return the audio file path

    except LangDetectException:
        return "Could not detect the language of the text."  # If language detection fails
    except Exception as e:
        return None  # Handle any other exceptions

# Dynamic input visibility control based on source type
def dynamic_input(source):
    if source == "Text Input":
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
    elif source == "Web Page URL":
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)
    elif source == "File Upload":
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Summarization, Translation, and Text-to-Speech App")

    # Input type selection
    source = gr.Dropdown(["Text Input", "Web Page URL", "File Upload"], label="Input Type", value="Text Input", interactive=True)

    # Input fields
    text_input_box = gr.Textbox(label="Enter Text", visible=True)
    file_input_box = gr.File(label="Upload a File", visible=False, file_types=[".pdf", ".docx"])
    url_input_box = gr.Textbox(label="Give Website URL", visible=False)

    # Language selection for translation
    target_lang = gr.Dropdown(["Original", "fi", "es", "fr", "de"], label="Choose Output Language", value="Original")

    # Summary type selection
    format_type = gr.Dropdown(
        ["Main Points", "Concepts List (Only works with English Input)", "Short Summary", "Medium Summary", "Long Summary"],
        label="Choose Summary Type",
        value="Main Points"
    )

    # Output box for the generated summary
    summary_output_box = gr.Textbox(label="Generated Summary", visible=True, lines=10)

    # Button to trigger summary generation
    summary_button = gr.Button("Generate Summary")

    # Button to trigger text-to-speech generation
    tts_button = gr.Button("Text to Speech")

    # Flag for whether summary has been generated
    summary_generated = gr.State(False)

    # Trigger the summary generation
    summary_button.click(
        fn=generate_summary,
        inputs=[text_input_box, url_input_box, file_input_box, format_type, source, target_lang],
        outputs=[summary_output_box, summary_generated]
    )

    # Trigger the TTS generation
    tts_button.click(
        fn=text_to_speech,
        inputs=[text_input_box, summary_output_box, summary_generated],
        outputs=gr.Audio(label="Generated Speech", type="filepath")
    )

    # Dynamically update input visibility based on source type
    source.change(fn=dynamic_input, inputs=source, outputs=[text_input_box, file_input_box, url_input_box])

# Launch the interface
demo.launch()
