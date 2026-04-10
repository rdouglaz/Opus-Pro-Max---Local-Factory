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
import bisect
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip
from moviepy.config import change_settings

# --- CONFIGURATION --- 
change_settings({"IMAGEMAGICK_BINARY": "magick"})   # ← changed for Linux/Docker

DEEPGRAM_API_KEY = os.environ.get("DEEPGRAM_API_KEY", "")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")  # ← NEW: now configurable via Railway env var

MAX_WORKERS = max(1, min(4, (os.cpu_count() or 2) // 2))
JOB_DIR = "jobs"

STORY_PATTERNS = ["i thought", "i was wrong", "then", "but", "until", "this changed", "after that"]
CONTROVERSY_WORDS = ["lie", "truth", "scam", "fake", "exposed", "wrong", "problem", "bad", "hate", "never", "don't", "stop", "why"]
EMOTION_WORDS = ["crazy", "insane", "love", "angry", "shocked", "secret", "destroy"]
HOOK_WORDS = ["listen", "look", "here is", "the secret", "step", "how to"]

if not os.path.exists(JOB_DIR):
    os.makedirs(JOB_DIR)

# --- NEW: SCENE DETECTION ---
def detect_scenes_opencv(video_path, threshold=30.0):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    scenes = [0.0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            if diff.mean() > threshold:
                scenes.append(frame_idx / fps)

        prev_frame = gray
        frame_idx += 1

    cap.release()
    return sorted(list(set(scenes)))

# --- INGESTION & PREPROCESSING ---
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

# --- AI & NLP ---
def get_deepgram_transcript(audio_path):
    url = "https://api.deepgram.com/v1/listen?model=nova-2&punctuate=true&timestamps=true"
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/mp3"
    }
    try:
        with open(audio_path, "rb") as f:
            response = requests.post(url, headers=headers, data=f, timeout=60)
        if response.status_code != 200:
            return {}
        return response.json()
    except:
        return {}

def analyze_audio_peaks(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        peaks = librosa.util.peak_pick(onset_env, 3, 3, 3, 5, 0.5, 10)
        return sorted(list(librosa.frames_to_time(peaks, sr=sr)))
    except:
        return []

def generate_metadata_ollama(transcript):
    prompt = f"Act as a viral clip creator. Create a highly specific, non-generic title, description, and hook based ONLY on this transcript. Return ONLY valid JSON with keys 'title', 'description', 'hook'. Transcript: {transcript[:1200]}"
    payload = {"model": "llama3", "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=20).json()
        data = json.loads(response.get("response", "{}"))
        title = data.get("title", "").strip()
        if not title:
            title = transcript[:80]
        return {
            "title": title,
            "description": data.get("description", ""),
            "hook": data.get("hook", "")
        }
    except:
        return {"title": transcript[:80], "description": "", "hook": ""}

# --- HYBRID SEGMENTATION ---
def find_viral_clips(dg_data, peak_times, scene_times, min_dur, max_dur, max_clips):
    try:
        words = dg_data['results']['channels'][0]['alternatives'][0]['words']
    except:
        return []

    if not words:
        return []

    clips = []
    n = len(words)

    step = max(10, int(n / 300))  # FIXED

    i = 0
    while i < n:
        start_time = words[i].get('start', 0)
        if scene_times:
            start_time = min(scene_times, key=lambda x: abs(x - start_time))

        target_end = start_time + max_dur

        end_idx = i + 20
        while end_idx < n and words[end_idx].get('end', 0) < target_end:
            end_idx += 5

        if end_idx >= n:
            break

        end_time = words[end_idx].get('end', 0)
        future_scenes = [s for s in scene_times if s > end_time]
        if future_scenes:
            end_time = future_scenes[0]

        duration = end_time - start_time

        if min_dur <= duration <= max_dur:
            segment_words = [w for w in words if start_time <= w['start'] <= end_time]

            clips.append({
                "start": start_time,
                "end": end_time,
                "words": segment_words,
                "score": 0,
                "transcript": " ".join([w.get('punctuated_word', '') for w in segment_words])
            })

        i += step

    final_clips = []
    for c in clips:
        overlap = False
        for fc in final_clips:
            if max(0, min(c['end'], fc['end']) - max(c['start'], fc['start'])) > 5:
                overlap = True
                break
        if not overlap:
            final_clips.append(c)
        if len(final_clips) >= max_clips:
            break

    return final_clips

# --- RENDER ENGINE ---
def render_clip(video_path, clip_data, job_folder, idx):
    clip_dir = os.path.join(job_folder, f"clip_{idx}")
    os.makedirs(clip_dir, exist_ok=True)

    metadata = generate_metadata_ollama(clip_data['transcript'])

    clean_title = "".join([c for c in metadata.get('title', '') if c.isalnum() or c == ' ']).strip()
    if not clean_title:
        clean_title = f"Clip_{idx}"

    filename = clean_title.replace(' ', '_')[:80]
    output_file = os.path.join(clip_dir, f"{filename}.mp4")

    with open(os.path.join(clip_dir, "metadata.md"), "w") as f:
        f.write(f"# {metadata.get('title')}\n\nHook: {metadata.get('hook')}\n\nScore: {clip_data['score']}")

    video = VideoFileClip(video_path).subclip(clip_data['start'], clip_data['end'])
    w, h = video.size

    crop_w = int(w * 0.8)
    cropped_vid = video.crop(x_center=w/2, y_center=h/2, width=crop_w, height=h)

    resized_vid = cropped_vid.resize(height=1920)
    if resized_vid.w > 1080:
        resized_vid = resized_vid.resize(width=1080)

    final_base = resized_vid.on_color(size=(1080, 1920), color=(0, 0, 0), pos=('center', 'center'))

    subtitle_clips = []
    chunk = []
    chunk_start = None

    for i, w_data in enumerate(clip_data['words']):
        if chunk_start is None:
            chunk_start = max(0, w_data.get('start', 0) - clip_data['start'])

        word_text = w_data.get('punctuated_word', '')
        chunk.append(word_text.upper())

        if len(chunk) >= 3 or word_text.endswith((".", "?", "!")) or i == len(clip_data['words']) - 1:
            end_t = min(video.duration, w_data.get('end', 0) - clip_data['start'])
            text_str = " ".join(chunk)

            color = 'yellow' if any(kw in text_str.lower() for kw in CONTROVERSY_WORDS + EMOTION_WORDS) else 'white'

            txt_clip = TextClip(
                text_str,
                font='Impact',
                fontsize=70,
                color=color,
                stroke_color='black',
                stroke_width=2,
                method='caption',
                size=(900, None)
            ).set_start(chunk_start).set_end(end_t).set_position(('center', 1400))

            subtitle_clips.append(txt_clip)
            chunk = []
            chunk_start = None

    final_clip = CompositeVideoClip([final_base] + subtitle_clips)

    final_clip.write_videofile(
        output_file,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        preset="medium",
        threads=MAX_WORKERS,
        ffmpeg_params=[
            "-crf", "23",
            "-maxrate", "4M",
            "-bufsize", "8M",
            "-pix_fmt", "yuv420p"
        ],
        logger=None
    )

    final_clip.close()
    video.close()

    with open(output_file, "rb") as f:
        return output_file, f.read()

# --- ZIP ---
def create_zip(files):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as z:
        for path, data in files:
            z.writestr(os.path.basename(path), data)
    zip_buffer.seek(0)
    return zip_buffer

# --- STREAMLIT UI ---
st.set_page_config(page_title="Opus Pro Max", layout="wide")
st.title("✂️ Opus Pro Max - Local CPU")

col1, col2 = st.columns([1, 2])

with col1:
    source_type = st.radio("Source", ["YouTube URL", "Local File"])

    clip_length_option = st.selectbox("Clip Length", ["15-30s", "30-60s", "61-90s"])
    max_clips_option = st.selectbox("Maximum Clips", [10, 20, 30, 50, 100])

    if source_type == "YouTube URL":
        input_val = st.text_area("YouTube Links (one per line)")
    else:
        input_val = st.file_uploader("Upload Videos", accept_multiple_files=True)

    if st.button("Generate Viral Clips") and input_val:

        if clip_length_option == "15-30s":
            min_dur, max_dur = 15, 30
        elif clip_length_option == "30-60s":
            min_dur, max_dur = 30, 60
        else:
            min_dur, max_dur = 61, 90

        batch_items = input_val if source_type != "YouTube URL" else [u.strip() for u in input_val.split("\n") if u.strip()]

        batch_progress = st.progress(0)
        batch_status = st.empty()

        all_outputs = []
        batch_start = time.time()

        for b_idx, item in enumerate(batch_items):
            job_folder = os.path.join(JOB_DIR, str(uuid.uuid4())[:8])
            os.makedirs(job_folder, exist_ok=True)

            with col2:
                progress_bar = st.progress(0)
                status = st.empty()

                status.text(f"Job {b_idx+1}/{len(batch_items)} - Ingesting...")

                if source_type == "YouTube URL":
                    video_path = download_youtube(item, job_folder)
                else:
                    video_path = os.path.join(job_folder, "source.mp4")
                    with open(video_path, "wb") as f:
                        f.write(item.getbuffer())

                progress_bar.progress(10)

                status.text("Extracting Audio & Transcribing...")
                audio_path = extract_audio(video_path, job_folder)
                peak_times = analyze_audio_peaks(audio_path)
                scene_times = detect_scenes_opencv(video_path)
                dg_data = get_deepgram_transcript(audio_path)

                progress_bar.progress(40)

                status.text("Segmenting Clips...")
                viral_clips = find_viral_clips(
                    dg_data,
                    peak_times,
                    scene_times,
                    min_dur,
                    max_dur,
                    max_clips_option
                )

                progress_bar.progress(60)

                rendered_files = []
                start_time = time.time()

                if viral_clips:
                    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                        futures = [executor.submit(render_clip, video_path, c, job_folder, i+1) for i, c in enumerate(viral_clips)]
                        for i, future in enumerate(futures):
                            rendered_files.append(future.result())

                            elapsed = time.time() - start_time
                            eta = int((elapsed / (i+1)) * (len(viral_clips)-(i+1))) if i+1 else 0

                            progress_bar.progress(60 + int(((i+1)/len(viral_clips))*40))
                            status.text(f"Rendering {i+1}/{len(viral_clips)} | ETA: {eta}s")
                else:
                    progress_bar.progress(100)
                    status.text("No clips found")

                all_outputs.extend(rendered_files)

                batch_elapsed = time.time() - batch_start
                batch_eta = int((batch_elapsed / (b_idx+1)) * (len(batch_items)-(b_idx+1))) if b_idx+1 else 0

                batch_progress.progress(int(((b_idx+1)/len(batch_items))*100))
                batch_status.text(f"Batch {b_idx+1}/{len(batch_items)} | ETA: {batch_eta}s")

        st.success("Batch Complete!")

        for path, vid_bytes in all_outputs:
            st.video(vid_bytes)

        zip_file = create_zip(all_outputs)
        st.download_button("📦 Download All Clips (ZIP)", data=zip_file, file_name="clips.zip")
