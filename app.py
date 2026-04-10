import streamlit as st
import os
import subprocess
import json
import requests
import librosa
import cv2
import math
import uuid
import shutil
import yt_dlp
import time
import numpy as np
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings

# --- CONFIGURATION ---
change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"})

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
JOB_DIR = "jobs"

# MediaPipe Face Detection Setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

if not os.path.exists(JOB_DIR):
    os.makedirs(JOB_DIR)

# --- INGESTION LAYER ---
def download_youtube(url, job_folder):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': os.path.join(job_folder, 'source_video.%(ext)s'),
        'quiet': True,
        'no_warnings': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return os.path.join(job_folder, f"source_video.{info['ext']}")

def extract_audio(video_path, job_folder):
    audio_path = os.path.join(job_folder, "audio.mp3")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-ar", "16000", "-ac", "1", "-b:a", "32k",
        audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

# --- AUDIO PROCESSING LAYER (DEEPGRAM) ---
def get_deepgram_transcript(audio_path):
    url = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true&utterances=true&timestamps=true"
    headers = {"Authorization": f"Token {DEEPGRAM_API_KEY}", "Content-Type": "audio/mp3"}
    try:
        with open(audio_path, "rb") as f:
            response = requests.post(url, headers=headers, data=f, timeout=60)
        return response.json() if response.status_code == 200 else {}
    except:
        return {}

# --- FACE TRACKING & ACTIVE SPEAKER DETECTION ---
def get_smooth_face_center(video_path, start_t, end_t):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, start_t * 1000)
    
    centers = []
    frames_to_process = int((end_t - start_t) * fps)
    
    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret: break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_frame)
        
        if results.detections:
            best_detection = max(results.detections, key=lambda d: d.score[0])
            bbox = best_detection.location_data.relative_bounding_box
            center_x = bbox.xmin + (bbox.width / 2)
            centers.append(center_x)
        else:
            centers.append(0.5) 
            
    cap.release()
    
    if not centers: return np.array([0.5])
    window_size = max(1, int(fps // 2))
    smoothed_centers = np.convolve(centers, np.ones(window_size)/window_size, mode='same')
    return smoothed_centers

# --- SEGMENTATION LAYER (STRICT 60-90s) ---
def find_viral_clips(dg_data):
    try:
        words = dg_data['results']['channels'][0]['alternatives'][0]['words']
    except:
        return []
    
    clips = []
    n = len(words)
    if n == 0: return []
    
    i = 0
    while i < n and len(clips) < 50:
        start_time = words[i]['start']
        # Look for a closing word between 60s and 90s from start
        target_min = start_time + 60
        target_max = start_time + 90
        
        end_idx = i
        # Fast forward to the 60s mark
        while end_idx < n and words[end_idx]['end'] < target_min:
            end_idx += 1
            
        # Find best break point before 90s
        best_break = end_idx
        while end_idx < n and words[end_idx]['end'] < target_max:
            # Prefer end of sentences
            if words[end_idx]['word'].endswith(('.', '!', '?')):
                best_break = end_idx
                break
            best_break = end_idx
            end_idx += 1
            
        if best_break < n:
            actual_end = words[best_break]['end']
            duration = actual_end - start_time
            
            if 60 <= duration <= 90.5:
                segment_words = words[i:best_break+1]
                clips.append({
                    "start": start_time,
                    "end": actual_end,
                    "words": segment_words,
                    "transcript": " ".join([w['word'] for w in segment_words])
                })
        
        # Advance i to avoid heavy overlap, moving ~30s forward
        advance_target = start_time + 30
        while i < n and words[i]['start'] < advance_target:
            i += 1
            
    return clips

# --- AUTO FRAMING & RENDER ENGINE ---
def render_clip(video_path, clip_data, job_folder, idx):
    clip_dir = os.path.join(job_folder, f"clip_{idx}")
    os.makedirs(clip_dir, exist_ok=True)
    output_file = os.path.join(clip_dir, f"clip_{idx}.mp4")

    smooth_x_path = get_smooth_face_center(video_path, clip_data['start'], clip_data['end'])
    video = VideoFileClip(video_path).subclip(clip_data['start'], clip_data['end'])
    w, h = video.size
    
    target_ratio = 9/16
    crop_w = h * target_ratio
    
    def scroll_frame(get_frame, t):
        frame = get_frame(t)
        frame_idx = min(int(t * video.fps), len(smooth_x_path) - 1)
        center_x_pct = smooth_x_path[frame_idx]
        
        pixel_center_x = center_x_pct * w
        x1 = max(0, min(w - crop_w, pixel_center_x - (crop_w / 2)))
        
        cropped = frame[:, int(x1):int(x1+crop_w)]
        return cv2.resize(cropped, (1080, 1920))

    transformed_vid = video.fl(scroll_frame)

    subtitle_clips = []
    for w_data in clip_data['words']:
        start_t = max(0, w_data['start'] - clip_data['start'])
        end_t = min(video.duration, w_data['end'] - clip_data['start'])
        
        if start_t >= video.duration: continue

        txt = TextClip(
            w_data['word'].upper(),
            font='Impact', fontsize=95, color='yellow',
            stroke_color='black', stroke_width=2,
            method='caption', size=(900, None)
        ).set_start(start_t).set_end(end_t).set_position(('center', 1450))
        
        subtitle_clips.append(txt)

    final_clip = CompositeVideoClip([transformed_vid] + subtitle_clips)
    final_clip.write_videofile(
        output_file, 
        fps=30, 
        codec="libx264", 
        audio_codec="aac", 
        preset="ultrafast", 
        threads=MAX_WORKERS, 
        logger=None
    )
    
    video.close()
    return output_file

# --- STREAMLIT UI ---
st.set_page_config(page_title="Opus CLUS AI", layout="wide")
st.title("🧠 Opus CLUS - Local Clipping AI")

source_type = st.radio("Source", ["YouTube URL", "Local File"])
input_val = st.text_input("Enter URL") if source_type == "YouTube URL" else st.file_uploader("Upload Video")

if st.button("Start Pipeline") and input_val:
    job_id = str(uuid.uuid4())[:8]
    job_folder = os.path.join(JOB_DIR, job_id)
    os.makedirs(job_folder, exist_ok=True)
    
    with st.status("Processing...") as status:
        st.write("Ingesting Video...")
        if source_type == "YouTube URL":
            video_path = download_youtube(input_val, job_folder)
        else:
            video_path = os.path.join(job_folder, "source.mp4")
            with open(video_path, "wb") as f: f.write(input_val.getbuffer())
        
        st.write("Transcribing Audio...")
        audio_path = extract_audio(video_path, job_folder)
        dg_data = get_deepgram_transcript(audio_path)
        
        st.write("Detecting Viral Segments (60s-90s)...")
        clips = find_viral_clips(dg_data)
        st.write(f"Found {len(clips)} eligible clips.")
        
        for i, clip in enumerate(clips):
            st.write(f"Rendering Clip {i+1}/{len(clips)} (Duration: {int(clip['end']-clip['start'])}s)")
            out = render_clip(video_path, clip, job_folder, i+1)
            st.video(out)
            
        status.update(label="All Clips Processed!", state="complete")