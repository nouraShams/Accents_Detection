REM Waste Accent Analyzer
 Setup
1. Install system dependencies:  
   - Ubuntu: `sudo apt install ffmpeg`  
   - Mac: `brew install ffmpeg`  
   - Windows: Download [FFmpeg](https://ffmpeg.org/) and add to PATH

2. Install Python packages:  
   `pip install -r requirements.txt`

3. Run the app:  
   `streamlit run app.py`

 Usage
- Enter public video URL (Loom, YouTube, direct MP4)
- Click "Analyze Video"
- View accent classification and confidence score

Limitations
- Accent model covers: US, UK, Canadian, Australian, Indian accents
- Best results with clear English speech >30 seconds
- May not recognize heavy non-native accents
