# REM Waste - English Accent Analyzer Tool

## Introduction

This repository contains a Streamlit-based web application developed as a practical challenge for the REM Waste interview process. The tool is designed to automate the evaluation of spoken English by analyzing public video URLs, extracting audio, transcribing speech, and classifying the speaker's English accent.

## Challenge Task Objective

The primary objective, as outlined in the challenge, is to build a working script or simple tool capable of:

1.  Accepting a public video URL (e.g., Loom, direct MP4 link).

2.  Extracting audio from the provided video.

3.  Analyzing the speaker's accent to detect English language speaking candidates.

4.  Outputting:

    - Classification of the English accent (e.g., British, American, Australian).

    - A confidence score for the English accent classification (0-100%).

    - A short summary or transcription of the spoken content.

This tool is intended for internal use to aid in evaluating spoken English for hiring purposes.

## Features

- **Public Video URL Acceptance:** Supports various public video sources, including YouTube and direct MP4 links.

- **Automated Audio Extraction:** Utilizes `yt-dlp` for efficient audio extraction from video URLs.

- **Speech-to-Text Transcription:** Employs the `faster-whisper` library, an optimized implementation of OpenAI's Whisper model, for accurate speech transcription and language detection.

- **English Accent Classification:** Integrates a specialized Hugging Face Transformers model (`dima806/english_accents_classification`) fine-tuned for English accent recognition.

- **Confidence Scoring:** Provides confidence levels for both detected language and accent classification.

- **User-Friendly Interface:** Built with Streamlit for a simple, interactive, and responsive web application.

- **Temporary File Management:** Ensures proper cleanup of temporary audio files after processing.

## Technical Approach

My approach prioritizes **practicality**, **creativity**, and **technical execution** within the given time constraints.

- **Practicality:** The solution is a fully functional proof-of-concept. It leverages robust, industry-standard open-source libraries (`yt-dlp`, `faster-whisper`, `transformers`) that are well-maintained and highly capable for their respective tasks. The Streamlit framework enables rapid development and easy deployment, making the tool immediately usable for evaluation.

- **Creativity:** Instead of a generic audio classification model, I opted for `dima806/english_accents_classification`, a pre-trained `Wav2Vec2` model specifically fine-tuned for English accent recognition. This demonstrates a targeted solution for the accent classification requirement, aiming for higher relevance and accuracy compared to broader audio classifiers. The integration of `faster-whisper` for efficient transcription and language detection provides a comprehensive analysis pipeline.

- **Technical Execution:**

  - **Modular Code:** The application logic is structured into distinct, well-commented functions (e.g., `download_and_extract_audio`, `load_whisper_model`, `load_accent_model`, `analyze_audio`), enhancing readability, maintainability, and testability.

  - **Resource Caching:** `st.cache_resource` is utilized for AI model loading. This significantly improves performance on subsequent runs by caching large model files, reducing startup time and computational overhead.

  - **Error Handling:** Robust `try-except` blocks are implemented throughout the audio extraction and analysis pipeline to gracefully handle common issues (e.g., invalid URLs, video restrictions, network timeouts, missing dependencies) and provide informative feedback to the user.

  - **Dependency Management:** `requirements.txt` lists all Python dependencies, and `packages.txt` specifies system-level packages (`ffmpeg`) required for deployment environments like Streamlit Community Cloud.

  - **User Interface:** Streamlit provides a clean, intuitive, and mobile-responsive interface, ensuring a smooth user experience for reviewers.

## Setup and Local Execution

To run this application locally, please follow these steps:

1.  **Clone the Repository (or Download Files):**

    ```
    git clone [YOUR_GITHUB_REPO_LINK_HERE]
    cd [YOUR_PROJECT_DIRECTORY_NAME]
    ```

    Alternatively, download `app.py`, `requirements.txt`, and `packages.txt` into a single directory.

2.  **Create and Activate a Virtual Environment (Highly Recommended):**

    ```
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install System-Level Dependencies (`yt-dlp` and `ffmpeg`):**
    Ensure `yt-dlp` and `ffmpeg` are installed on your system and accessible via your system's `PATH` environment variable.

    - **`yt-dlp`:** Download `yt-dlp.exe` (for Windows) from its [GitHub Releases page](https://github.com/yt-dlp/yt-dlp/releases) and place it in a directory (e.g., `C:\Tools`).

    - **`ffmpeg`:** Download a `ffmpeg` build (e.g., a `.zip` file) for Windows from [gyan.dev/ffmpeg/builds/](https://www.gyan.dev/ffmpeg/builds/). Extract the archive, locate the `bin` directory within it, and copy its contents (`ffmpeg.exe`, `ffprobe.exe`) to the same directory (e.g., `C:\Tools`).

    - **Add the directory to `System PATH`:** Search for "Environment Variables" in Windows, edit the "Path" system variable, and add the full path to your chosen directory (e.g., `C:\Tools`).

    - **Restart your command-line interface** (PowerShell/CMD) for `PATH` changes to take effect.

4.  **Install Python Dependencies:**
    With your virtual environment activated, install the required Python libraries:

    ```
    pip install -r requirements.txt
    ```

5.  **Run the Application:**

    ```
    streamlit run app.py
    ```

    The application will automatically open in your web browser (typically at `http://localhost:8501`).

## Deployment

This application is designed for straightforward deployment on Streamlit Community Cloud, offering a simple public link for testing.

1.  **Upload all project files** (`app.py`, `requirements.txt`, `packages.txt`, `README.md`) to a public GitHub repository.

2.  **Navigate to [share.streamlit.io](https://share.streamlit.io/)** and log in with your GitHub account.

3.  **Click "New app"** and select your repository.

4.  **Ensure the "Main file path" is set to `app.py`**.

5.  **Click "Deploy!"**.

6.  Once deployed, a public URL will be provided for accessing the application.

## Conclusion

This solution aims to demonstrate a practical, creative, and technically sound approach to the challenge task. It leverages powerful open-source tools to deliver a functional prototype for English accent analysis. I am confident in its ability to meet the core requirements and provide valuable insights for REM Waste's hiring process.

## Contact

Noura Shbani
nouranjsh@gmail.com
