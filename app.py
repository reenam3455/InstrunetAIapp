import streamlit as st
import torch
import torch.nn as nn
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import json
import tempfile
import os
from fpdf import FPDF

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="InstruNet AI",
    page_icon="🎵",
    layout="wide"
)

st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 1rem; }
div[data-testid="stVerticalBlock"] { gap: 0.75rem; }
.stButton > button { width: 100%; border-radius: 6px; height: 2.8em; }
</style>
""", unsafe_allow_html=True)

# ================= AUTH =================
USERS_FILE = "users.json"

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

def hash_password(p):
    return hashlib.sha256(p.encode()).hexdigest()

if "users" not in st.session_state:
    st.session_state.users = load_users()
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# ================= LOGIN =================
def login_page():
    st.markdown(
        "<h1 style='text-align:center;'>🎵 InstruNet AI</h1>"
        "<p style='text-align:center;color:gray;'>InstruNet AI Authentication</p>",
        unsafe_allow_html=True
    )

    _, col, _ = st.columns([1, 2, 1])
    with col:
        with st.form("login"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            mode = st.selectbox("Action", ["Login", "Create Account"])
            submit = st.form_submit_button("Continue")

        if submit:
            if not u or not p:
                st.error("All fields required")
                return

            h = hash_password(p)

            if mode == "Create Account":
                if u in st.session_state.users:
                    st.error("User exists")
                    return
                st.session_state.users[u] = h
                save_users(st.session_state.users)
                st.success("Account created. Please login.")
            else:
                if st.session_state.users.get(u) != h:
                    st.error("Invalid credentials")
                    return
                st.session_state.authenticated = True
                st.session_state.current_user = u
                st.rerun()

# ================= CONFIG =================
CFG = {
    "sr": 16000,
    "segment_sec": 3.0,
    "n_mels": 128,
    "hop": 512,
    "max_segments": 10,
    "instruments": [
        "Cello","Clarinet","Flute","Acoustic Guitar",
        "Electric Guitar","Organ","Piano",
        "Saxophone","Trumpet","Violin","Human Voice"
    ]
}
INSTRUMENTS = CFG["instruments"]
ICON_MAP = {
    "Violin": "🎻",
    "Cello": "🎻",
    "Trumpet": "🎺",
    "Saxophone": "🎷",
    "Piano": "🎹",
    "Acoustic Guitar": "🎸",
    "Electric Guitar": "🎸",
    "Flute": "🎶",
    "Clarinet": "🎶",
    "Organ": "🎼",
    "Human Voice": "🗣️"
}

# ================= MODEL =================
class CNN(nn.Module):
    def __init__(self, num_classes=len(INSTRUMENTS)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x.view(B, S, -1)

@st.cache_resource
def load_model():
    m = CNN()
    m.load_state_dict(torch.load("trained_model_fixed.pth", map_location="cpu"))
    m.eval()
    return m

# ================= AUDIO =================
def load_audio(path):
    y,_ = librosa.load(path, sr=CFG["sr"], mono=True)
    return y/(np.max(np.abs(y))+1e-9)

def segment_audio(y):
    L = int(CFG["sr"]*CFG["segment_sec"])
    return [np.pad(y[i:i+L],(0,max(0,L-len(y[i:i+L]))))
            for i in range(0,len(y),L)][:CFG["max_segments"]]

def mel(y):
    m = librosa.feature.melspectrogram(y=y,sr=CFG["sr"],n_mels=CFG["n_mels"],hop_length=CFG["hop"])
    m = librosa.power_to_db(m,ref=np.max)
    return (m+80)/80

def predict(model,path,agg):
    y = load_audio(path)
    segs = segment_audio(y)
    x = torch.tensor(np.stack([mel(s) for s in segs])[:,None]).unsqueeze(0).float()
    with torch.no_grad():
        p = model(x)[0].numpy()
    out = {"mean":p.mean(0),"max":p.max(0),"median":np.median(p,0)}[agg]
    return out,p,y

# ================= VISUALS =================
def plot_spectrogram(y):
    S = librosa.feature.melspectrogram(y=y, sr=CFG["sr"], n_mels=CFG["n_mels"])
    S_db = librosa.power_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8,4))
    img = librosa.display.specshow(S_db, sr=CFG["sr"], hop_length=CFG["hop"], x_axis='time', y_axis='mel', cmap='magma', ax=ax)
    ax.set_title("Mel Spectrogram")
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    return fig

def plot_bars(p,t):
    detected = [(i,v) for i,v in zip(INSTRUMENTS,p) if v>=t]
    if not detected: return None
    labels, values = zip(*detected)
    fig, ax = plt.subplots(figsize=(6,len(labels)*0.45+1))
    ax.barh(labels, values, color="steelblue")
    ax.axvline(t, color="red", ls="--")
    ax.set_xlim(0,1)
    ax.set_title("Confidence")
    return fig

def plot_timeline(probs_segments,t):
    times = np.arange(len(probs_segments))*CFG["segment_sec"]
    fig, ax = plt.subplots(figsize=(6,3))
    for i, inst in enumerate(INSTRUMENTS):
        if probs_segments[:,i].max()>=t:
            ax.plot(times, probs_segments[:,i], label=inst, linewidth=2)
    ax.axhline(t,color="red",ls="--")
    ax.set_ylim(0,1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Confidence")
    ax.set_title("Detected Instruments Timeline")
    ax.legend(ncol=2)
    return fig

# ================= EXPORT =================
def export_json(name, dur, p, t, agg):
    return json.dumps({
        "project": "InstruNet AI",
        "audio_file": name,
        "duration_sec": round(dur, 2),
        "threshold": t,
        "aggregation": agg,
        "instruments": {i: round(float(v), 4) for i, v in zip(INSTRUMENTS, p)}
    }, indent=2)

def export_pdf(name, dur, p, t, agg, y):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 18)
    pdf.cell(0, 10, " InstruNet AI Report", ln=True, align="C")
    pdf.ln(5)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6,
        f"File: {name}\nDuration: {dur:.2f} seconds\nConfidence Threshold: {t}\nAggregation Method: {agg}"
    )
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 6, "Mel Spectrogram", ln=True)
    fig = plot_spectrogram(y)
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    fig.savefig(tmp_img.name, bbox_inches='tight')
    tmp_img.close()
    pdf.image(tmp_img.name, x=15, w=180)
    os.unlink(tmp_img.name)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 6, "Detected Instruments", ln=True)
    pdf.set_font("Arial", "", 12)
    detected = [(i,v) for i,v in zip(INSTRUMENTS,p) if v>=t]
    if not detected:
        pdf.cell(0,6,"None detected above threshold.", ln=True)
    else:
        for i,v in detected:
            pdf.cell(0,6,f"{i}: {v*100:.1f}%", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 6, "Analysis Summary", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 6,
        "This audio file was analyzed using InstruNet AI, a deep learning model designed to "
        "identify musical instruments. The audio was first segmented into short intervals, "
        "converted into mel spectrograms, and then passed through a convolutional neural network. "
        "The model outputs confidence scores for each instrument, which are aggregated according to "
        f"the '{agg}' method. Instruments above the selected threshold ({t}) are considered detected."
    )
    return pdf.output(dest="S").encode("latin-1")

# ================= MAIN APP =================
def main_app():
    model = load_model()
    st.title("🎵 InstruNet AI")
    st.caption(f"Logged in as **{st.session_state.current_user}**")

    # LEFT PANEL: always visible
    left, main = st.columns([1.2, 3], gap="small")
    with left:
        st.subheader("Upload & Settings")
        file = st.file_uploader("Audio", ["wav", "mp3"])

        # Audio preview
        if file:
            st.audio(file, format="audio/wav")

        threshold = st.slider("Confidence Threshold", 0.2, 0.9, 0.5)
        agg = st.selectbox("Aggregation Method", ["mean", "max", "median"])
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

    # MAIN AREA
    with main:
        if not file:
            # Welcome message
            st.markdown(
                f"""
<div style="text-align:center;padding:3rem;border-radius:14px;">
                <h1>🎵 InstruNet AI</h1>
                <p>Automatically identifies musical instruments using deep learning.</p>
                <p style="font-weight:600;">🎼 Supported Instruments</p>
                <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.6rem;">
                    {''.join([f"<div class='instrument-text'>{ICON_MAP[i]} {i}</div>" for i in INSTRUMENTS])}
                </div>
                <p style="margin-top:1.5rem;">👉 Upload an audio file to begin analysis.</p>
            </div>
            """, unsafe_allow_html=True)

            
        else:
            # Analysis layout: center + right
            center, right = st.columns([2.8, 1.4], gap="small")

            with center:
                with st.spinner("Analyzing audio..."):
                    with tempfile.NamedTemporaryFile(delete=False) as tmp:
                        tmp.write(file.getvalue())
                        path = tmp.name

                    probs, probs_segments, y = predict(model, path, agg)
                    duration = librosa.get_duration(y=y, sr=CFG["sr"])
                    os.unlink(path)

                st.pyplot(plot_spectrogram(y), use_container_width=True)
                fig = plot_bars(probs, threshold)
                if fig:
                    st.pyplot(fig, use_container_width=True)

            with right:
                st.subheader("Detected Instruments")
                detected = [(i, v) for i, v in zip(INSTRUMENTS, probs) if v >= threshold]
                if not detected:
                    st.warning("No instruments detected above threshold")
                else:
                    for i, v in detected:
                        st.markdown(f"**{i}** — {v*100:.1f}%")

                st.subheader("Timeline")
                st.pyplot(plot_timeline(probs_segments, threshold), use_container_width=True)

                st.subheader("Export")
                st.download_button(
                    "Download JSON",
                    export_json(file.name, duration, probs, threshold, agg),
                    file_name="prediction.json"
                )
                st.download_button(
                    "Download PDF",
                    export_pdf(file.name, duration, probs, threshold, agg, y),
                    file_name="InstruNet_Report.pdf",
                    mime="application/pdf"
                )

# ================= ROUTER =================
if st.session_state.authenticated:
    main_app()
else:
    login_page()
