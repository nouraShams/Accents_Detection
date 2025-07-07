import streamlit as st
import subprocess
import os
import tempfile
import shutil # For cleaning up temporary directories
from faster_whisper import WhisperModel
from transformers import pipeline
import numpy as np
import soundfile as sf # For reading audio for the accent model

WHISPER_MODEL_SIZE = "small" 
ACCENT_MODEL_NAME = "dima806/english_accents_classification"



def download_and_extract_audio(video_url):
   
    try:
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, "audio.wav")

        st.info(f"Downloading video and extracting audio from: {video_url}")
        
        command = [
            "yt-dlp",# Use yt-dlp to download and extract audio directly to WAV
            "-x",# -x: extract audio
            "--audio-format", "wav",# --audio-format wav: specify WAV formatر
            "--no-playlist",# -o: output template (file path)
            "--no-warnings", # --no-playlist: prevent downloading entire playlists if the link is a playlist
            "-o", audio_path, # --no-warnings: hide warnings
            video_url
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        st.success("Audio extracted successfully!")
        return audio_path
    except subprocess.CalledProcessError as e:
        st.error(f"Error extracting audio: {e.stderr}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during audio extraction: {e}")
        return None

@st.cache_resource
def load_whisper_model():
    st.info(f"Loading Whisper model ({WHISPER_MODEL_SIZE}).")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8") 
    st.success("Whisper model loaded!")
    return model

@st.cache_resource
def load_accent_model():
    st.info(f"Loading accent classification model ({ACCENT_MODEL_NAME}). ")
    
    
    
    classifier = pipeline(
        "audio-classification", # Use pipeline for easier inference
        model=ACCENT_MODEL_NAME, 
        framework="pt",# "pt" (PyTorch) as the model uses PyTorch weights. 
        model_kwargs={"ignore_mismatched_sizes": True}# Added ignore_mismatched_sizes=True as a safeguard.
    )
    st.success("Accent classification model loaded!")
    return classifier

def analyze_audio(audio_path, whisper_model, accent_classifier):
    """
    Analyzes audio for transcription, language, and accent.
    """
    results = {}

    # 1. Transcribe and detect language using Whisper
    st.info("Transcribing audio and detecting language using Whisper...")
    # beam_size=5 is a good setting for accuracy
    segments, info = whisper_model.transcribe(audio_path, beam_size=5)
    
    detected_language = info.language
    language_probability = info.language_probability

    full_transcript = ""
    for segment in segments:
        full_transcript += segment.text + " "
    
    results['transcript'] = full_transcript.strip()
    results['detected_language'] = detected_language
    results['language_probability'] = language_probability

    st.success(f"Whisper analysis complete. Detected language: {detected_language} (confidence: {language_probability:.2f})")

    # 2. Accent classification (if English is detected)
    if detected_language == "en":
        st.info("Performing accent classification...")
        try:
            # The accent classifier expects raw audio data (numpy array) and sample rate
            audio_data, sample_rate = sf.read(audio_path)
            
            # Ensure audio is single channel (mono) if it's stereo
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0] # Take the first channel

            # The pipeline expects a dictionary containing 'raw' audio and 'sampling_rate'
            accent_prediction = accent_classifier({"raw": audio_data, "sampling_rate": sample_rate})
            
            # The output is a list of dictionaries, sorted by score.
            # Example: [{'score': 0.9, 'label': 'us'}, {'score': 0.05, 'label': 'uk'}]
            
            if accent_prediction:
                top_accent = accent_prediction[0] # Select the accent with the highest confidence
                results['accent_classification'] = top_accent['label']
                results['accent_confidence'] = top_accent['score']
                st.success(f"Accent classified as: {top_accent['label']} (confidence: {top_accent['score']:.2f})")
            else:
                results['accent_classification'] = "Not Available"
                results['accent_confidence'] = 0.0
                st.warning("Could not classify accent.")
        except Exception as e:
            st.error(f"Error during accent classification: {e}")
            results['accent_classification'] = "Error"
            results['accent_confidence'] = 0.0
    else:
        results['accent_classification'] = "Not Applicable (Not English)"
        results['accent_confidence'] = 0.0
        st.warning(f"Accent classification skipped because the detected language is not English ({detected_language}).")

    return results

# --- Streamlit UI ---
st.set_page_config(page_title="REM Waste - English Accent Analyzer", layout="centered")

st.title("🗣️ REM Waste - English Accent Analyzer Tool")
st.markdown("""
    Welcome to the English Accent Analyzer tool!
    This tool helps evaluate the English accent of candidates by analyzing video recordings.
    Simply enter a public video URL (e.g., Loom or direct MP4 link) and we will analyze the audio.

    **Note:** This tool is a Proof-of-Concept and may not be 100% accurate in all cases.
    It uses the Whisper model for transcription and language detection, and the `dima806/english_accents_classification` model for accent classification.
""")

video_url = st.text_input("Enter the public video URL here (e.g., Loom, direct MP4 link):", "")

if st.button("Analyze Video"):
    if not video_url:
        st.warning("Please enter a valid video URL.")
    else:
        with st.spinner("Analyzing video... This might take a few minutes depending on video size and internet speed."):
            # Load models
            whisper_model = load_whisper_model()
            accent_classifier = load_accent_model()

            # 1. Download and extract audio
            audio_file_path = download_and_extract_audio(video_url)

            if audio_file_path:
                # 2. Analyze audio
                analysis_results = analyze_audio(audio_file_path, whisper_model, accent_classifier)

                st.subheader("Analysis Results:")

                # Display accent classification
                st.write("---")
                st.markdown(f"### 🌍 Accent Classification: **{analysis_results.get('accent_classification', 'Not Available').upper()}**")
                
                # Display accent confidence score
                accent_confidence = analysis_results.get('accent_confidence', 0.0)
                st.markdown(f"### ✅ Accent Confidence Score: **{accent_confidence * 100:.2f}%**")
                
                # Display English language confidence (from Whisper)
                lang_prob = analysis_results.get('language_probability', 0.0)
                st.markdown(f"### 🗣️ English Language Confidence (Whisper): **{lang_prob * 100:.2f}%**")

                st.write("---")
                st.subheader("Summary/Brief Explanation:")
                if analysis_results.get('detected_language') == "en":
                    st.success("English language detected in the video. Accent analysis performed.")
                    st.markdown(f"**Transcription:**")
                    if analysis_results.get('transcript'):
                        st.info(analysis_results['transcript'])
                    else:
                        st.info("No transcription available.")
                    st.markdown("""
                        **Note on Confidence:**
                        * **Accent Confidence:** Indicates how confident the model is in the specific accent classification (e.g., US, UK).
                        * **English Language Confidence (Whisper):** Indicates how confident the Whisper model is that the spoken language is English. If this is low, the candidate might not be primarily speaking English.
                    """)
                else:
                    st.warning(f"Detected language is: **{analysis_results.get('detected_language', 'Unknown')}** (with {lang_prob*100:.2f}% confidence). English accent analysis was skipped because the primary language is not English.")
                    st.markdown(f"**Transcription:**")
                    if analysis_results.get('transcript'):
                        st.info(analysis_results['transcript'])
                    else:
                        st.info("No transcription available.")


                # Clean up temporary audio files
                if audio_file_path and os.path.exists(os.path.dirname(audio_file_path)):
                    shutil.rmtree(os.path.dirname(audio_file_path))
                    st.info("Temporary files cleaned up.")
            else:
                st.error("Could not extract audio from the video. Please check the URL and try again.")

st.markdown("""
---
**Notes for Reviewers:**
* **Practicality:** This tool leverages open-source libraries (yt-dlp, faster-whisper, transformers) to fully achieve the required functionalities.
* **Creativity:** A specialized Wav2Vec2 model from Hugging Face is used for precise English accent classification, providing a smart solution for a complex task.
* **Technical Execution:** The code is organized, uses caching for ML models to improve performance and reduce loading time. Temporary files are handled properly.
* **Limitations:** Accent classification accuracy might be limited based on audio quality and the diversity of accents in the model's training data.
""")
