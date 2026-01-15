import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import parselmouth
from parselmouth.praat import call
import tempfile
import os


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Frailty Audio Analysis",
    layout="centered",
    page_icon="🎤"
)

# -----------------------------
# CUSTOM STYLING
# -----------------------------
st.markdown("""
<style>
body {
    background-color: #f5f7fb;
}

.main-title {
    color: #2c3e50;
    font-size: 40px;
    font-weight: 700;
    text-align: center;
}

.subtitle {
    color: #555;
    font-size: 18px;
    text-align: center;
    margin-bottom: 25px;
}

.instruction-box {
    background-color: #e8f0fe;
    padding: 20px;
    border-radius: 10px;
    border-left: 6px solid #4285f4;
    margin-bottom: 20px;
}

.section-header {
    color: #1f4fd8;
    font-size: 26px;
    font-weight: 600;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# TITLE & DESCRIPTION
# -----------------------------
st.markdown('<div class="main-title">🎤 Frailty Audio Analysis</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload a speech recording to extract frailty-related voice characteristics</div>',
    unsafe_allow_html=True
)

# -----------------------------
# INSTRUCTIONS
# -----------------------------
st.markdown("""
<div class="instruction-box">
<b>📌 Instructions</b><br><br>
1️⃣ Upload a <b>WAV audio file</b> containing speech<br>
2️⃣ The system will analyze voice characteristics related to frailty<br>
3️⃣ You will see the waveform and extracted features<br>
4️⃣ Download the results as a CSV file if needed
</div>
""", unsafe_allow_html=True)


# -----------------------------
# FRAILTY PROCESSOR
# -----------------------------
class FrailtyAudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr

    def preprocess_audio(self, file_path):
        y, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)

        if len(y_trimmed) < self.target_sr * 0.5:
            return None, None

        return y_trimmed, sr

    def extract_biomarkers(self, file_path):
        y, sr = self.preprocess_audio(file_path)
        if y is None:
            return None

        sound = parselmouth.Sound(file_path)
        features = {}

        # A1: Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features["A1_zcr_mean"] = float(np.mean(zcr))

        # A2: Shimmer
        try:
            pp = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            features["A2_shimmer"] = float(
                call([sound, pp],
                     "Get shimmer (local)",
                     0, 0, 0.0001, 0.02, 1.3, 1.6)
            )
        except:
            features["A2_shimmer"] = 0.0

        # A3: F1 Formant Variability
        try:
            formant = sound.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=5,
                maximum_formant=5500
            )

            f1_vals = []
            n_frames = call(formant, "Get number of frames")

            for i in range(1, n_frames + 1):
                t = call(formant, "Get time from frame number", i)
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                if not np.isnan(f1):
                    f1_vals.append(f1)

            features["A3_f1_std"] = float(np.std(f1_vals)) if f1_vals else 0.0
        except:
            features["A3_f1_std"] = 0.0

        # A4: Low-Frequency Energy Ratio
        try:
            S = np.abs(librosa.stft(y)) ** 2
            freqs = librosa.fft_frequencies(sr=sr)
            cutoff = np.argmin(np.abs(freqs - 800))
            total_energy = np.sum(S)

            features["A4_energy_ratio"] = (
                float(np.sum(S[:cutoff]) / total_energy)
                if total_energy > 0 else 0.0
            )
        except:
            features["A4_energy_ratio"] = 0.0

        return features


# -----------------------------
# FILE UPLOAD
# -----------------------------
processor = FrailtyAudioProcessor()
uploaded_file = st.file_uploader("🎧 Upload WAV file", type=["wav"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # -----------------------------
    # WAVEFORM
    # -----------------------------
    st.markdown('<div class="section-header">🔊 Audio Waveform</div>', unsafe_allow_html=True)
    y, sr = librosa.load(tmp_path, sr=16000)

    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    st.pyplot(fig)

    # -----------------------------
    # FEATURES
    # -----------------------------
    st.markdown('<div class="section-header">📊 Frailty Voice Features</div>', unsafe_allow_html=True)
    features = processor.extract_biomarkers(tmp_path)

    if features:
        df = pd.DataFrame([features])
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "frailty_features.csv",
            "text/csv"
        )
    else:
        st.error("❌ Audio file too short or invalid.")

    os.remove(tmp_path)
