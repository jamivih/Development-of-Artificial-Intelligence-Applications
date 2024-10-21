import gradio as gr
from deep_translator import GoogleTranslator  # Replaced googletrans with deep-translator
from gtts import gTTS
from io import BytesIO
import tempfile  # To create a temporary file for the audio

# Simple text summarization (example)
def summarize_text(input_text, format_type):
    words = input_text.split()

    if format_type == "Main Points":
        summary = "Main points: " + " ".join(words[:30]) + "..."
    elif format_type == "Concepts List":
        summary = "Concepts: " + ", ".join(set(words[:30]))  # Very basic concept listing
    elif format_type == "Short Summary":
        summary = " ".join(words[:50]) + "..." if len(words) > 50 else input_text
    elif format_type == "Medium Summary":
        summary = " ".join(words[:100]) + "..." if len(words) > 100 else input_text
    elif format_type == "Long Summary":
        summary = " ".join(words[:150]) + "..." if len(words) > 150 else input_text
    else:
        summary = input_text

    return summary

# Text translation using deep-translator
def translate_text(input_text, dest_language):
    try:
        if dest_language == "original":
            return input_text  # No translation needed
        # Use deep-translator's GoogleTranslator for translation
        translation = GoogleTranslator(target=dest_language).translate(input_text)
        return translation
    except Exception as e:
        return f"Translation failed: {str(e)}"

# Text-to-speech (using gTTS)
def text_to_speech(input_text):
    if not input_text.strip():  # Check for empty or whitespace-only input
        return None  # Return None if no text is provided

    try:
        # Generate the speech using gTTS
        tts = gTTS(input_text)
        audio_file = BytesIO()  # In-memory file for the audio
        tts.write_to_fp(audio_file)  # Write the audio content to the in-memory file
        audio_file.seek(0)  # Rewind the file pointer to the start

        # Save the audio temporarily to a file path
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
            tts.save(temp_audio_file.name)
            return temp_audio_file.name  # Return the file path for Gradio Audio component
    except Exception as e:
        return None  # Return None if there's an error

# Dynamic input box rendering based on source selection
def dynamic_input(source):
    if source == "Text Input":
        return gr.update(visible=True, label="Enter Text"), gr.update(visible=False), gr.update(visible=False)
    elif source == "Web Page URL":
        return gr.update(visible=True, label="Enter Web Page URL"), gr.update(visible=False), gr.update(visible=False)
    elif source == "File Upload":
        return gr.update(visible=False), gr.update(True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

# Process file input
def process_file(file):
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            file_content = f.read()
        return file_content
    except Exception as e:
        return f"Error reading file: {str(e)}"

# Generate summary and translation based on input method
def generate_summary(input_text, url, file, format_type, lang, source):
    if source == "Text Input":
        content = input_text
    elif source == "Web Page URL":
        content = url  # In a real implementation, you'd fetch the web page content
    elif source == "File Upload":
        content = process_file(file)
    else:
        content = ""

    if not content:
        return "No content to summarize.", gr.update(visible=False)

    summary = summarize_text(content, format_type)
    translated_summary = translate_text(summary, lang) if lang != "original" else summary
    return translated_summary, gr.update(visible=True, lines=10, label="Generated Summary", value=translated_summary)

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("### Summarization, Translation, and Text-to-Speech App")

    # Input method selection (default is "Text Input")
    source = gr.Dropdown(["Text Input", "Web Page URL", "File Upload"], label="Input Method", value="Text Input", interactive=True)

    # Dynamic input box based on source selection
    text_input_box = gr.Textbox(label="Enter Text", visible=True)  # Make visible by default
    file_input_box = gr.File(label="Upload File", visible=False, file_types=[".txt"])
    url_input_box = gr.Textbox(label="Enter Web Page URL", visible=False)

    # Summary types
    format_type = gr.Dropdown(
        ["Main Points", "Concepts List", "Short Summary", "Medium Summary", "Long Summary"],
        label="Select Summary Type"
    )

    # Translation language
    lang = gr.Dropdown(["original", "en", "fi"], label="Translation Language")

    # Output summary text area (large text box)
    summary_output_box = gr.Textbox(label="Generated Summary", visible=False, lines=10)

    # Buttons
    summary_button = gr.Button("Generate Summary")
    tts_button = gr.Button("Text to Speech")
    redo_button = gr.Button("Re-do", visible=False)  # Initially hidden

    # Dynamic input change based on source selection
    source.change(dynamic_input, inputs=source, outputs=[text_input_box, file_input_box, url_input_box])

    # When 'Generate Summary' button is clicked
    summary_button.click(
        fn=generate_summary,
        inputs=[text_input_box, url_input_box, file_input_box, format_type, lang, source],
        outputs=[summary_output_box, redo_button]
    )

    # Text-to-speech button action, plays the text from the "Enter Text" input in audio format
    tts_button.click(
        fn=text_to_speech,
        inputs=text_input_box,  # Take input from the "Enter Text" box directly
        outputs=gr.Audio(label="Generated Speech", type="filepath")  # Output the file path
    )

demo.launch()
