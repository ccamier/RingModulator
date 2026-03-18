#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:04:44 2026

@author: cedriccamier
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.io.wavfile import write as wav_write

# ============================================================
# Configuration Streamlit
# ============================================================
st.set_page_config(
    page_title="Itérations i·fp + fm + WAV ADSR",
    page_icon="🎼",
    layout="wide"
)

st.title("Tableau des 12 premières itérations de i·fp + fm")

st.markdown(
    r"""
Cette application permet de saisir :

- une **fréquence porteuse** \(f_p\) ou une **note porteuse** \(n_p\)
- une **fréquence modulante** \(f_m\) ou une **note modulante** \(n_m\)

La sortie affiche les **12 premières valeurs** de :

\[
i \cdot f_p + f_m
\qquad \text{pour } i = 1, 2, \dots, 12
\]

avec la note correspondante au quart de ton près.

L'application génère aussi trois signaux audio avec une **enveloppe ADSR commune** :

\[
x_p(t)=A(t)\sin(2\pi f_p t)
\]

\[
x_m(t)=A(t)\sin(2\pi f_m t)
\]

\[
x_{\Sigma}(t)=A(t)\sum_{i=1}^{12}\sin\!\bigl(2\pi(i f_p+f_m)t\bigr)
\]
"""
)

# ============================================================
# Outils note <-> fréquence
# ============================================================
NOTE_NAMES_FR = ["Do", "Do♯", "Ré", "Ré♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]
MIN_MIDI = 24.0   # C1
MAX_MIDI = 108.0  # C8


def freq_to_midi(freq: float) -> float:
    if freq <= 0:
        raise ValueError("La fréquence doit être strictement positive.")
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def midi_to_freq(midi_value: float) -> float:
    return 440.0 * (2.0 ** ((midi_value - 69.0) / 12.0))


def round_to_quarter_tone_midi(midi_value: float) -> float:
    return round(midi_value * 2.0) / 2.0


def midi_int_to_note_name(midi_int: int) -> str:
    pitch_class = midi_int % 12
    octave = midi_int // 12 - 1
    return f"{NOTE_NAMES_FR[pitch_class]}{octave}"


def midi_quarter_to_label(midi_q: float) -> str:
    midi_floor = int(math.floor(midi_q))
    frac = midi_q - midi_floor
    base_note = midi_int_to_note_name(midi_floor)

    if abs(frac) < 1e-9:
        return base_note
    if abs(frac - 0.5) < 1e-9:
        return f"{base_note} + 1/4 ton"
    return base_note


def freq_to_note_label_quarter(freq: float) -> str:
    midi_real = freq_to_midi(freq)
    midi_q = round_to_quarter_tone_midi(midi_real)
    midi_q = min(max(midi_q, MIN_MIDI), MAX_MIDI)
    return midi_quarter_to_label(midi_q)


def midi_q_to_label(midi_q: float) -> str:
    midi_q = min(max(midi_q, MIN_MIDI), MAX_MIDI)
    return midi_quarter_to_label(midi_q)


NOTE_OPTIONS = []
m = MIN_MIDI
while m <= MAX_MIDI + 1e-9:
    NOTE_OPTIONS.append((m, midi_q_to_label(m)))
    m += 0.5

NOTE_LABELS = [label for _, label in NOTE_OPTIONS]
NOTE_LABEL_TO_MIDI = {label: midi for midi, label in NOTE_OPTIONS}


def note_label_to_midi_q(label: str) -> float:
    return NOTE_LABEL_TO_MIDI[label]


def note_freq_from_label(label: str) -> float:
    return midi_to_freq(note_label_to_midi_q(label))


def nearest_note_label_from_freq(freq: float) -> str:
    midi_real = freq_to_midi(freq)
    midi_q = round_to_quarter_tone_midi(midi_real)
    midi_q = min(max(midi_q, MIN_MIDI), MAX_MIDI)
    return midi_q_to_label(midi_q)


# ============================================================
# Audio / ADSR
# ============================================================
def build_adsr_envelope(
    fs: int,
    attack_ms: float,
    decay_ms: float,
    sustain_ms: float,
    release_ms: float,
    sustain_level: float,
):
    na = max(1, int(round(fs * attack_ms / 1000.0)))
    nd = max(1, int(round(fs * decay_ms / 1000.0)))
    ns = max(1, int(round(fs * sustain_ms / 1000.0)))
    nr = max(1, int(round(fs * release_ms / 1000.0)))

    attack = np.linspace(0.0, 1.0, na, endpoint=False)
    decay = np.linspace(1.0, sustain_level, nd, endpoint=False)
    sustain = np.full(ns, sustain_level)
    release = np.linspace(sustain_level, 0.0, nr, endpoint=True)

    env = np.concatenate([attack, decay, sustain, release])
    t = np.arange(len(env)) / fs
    return t, env


def generate_adsr_sine(freq: float, t: np.ndarray, env: np.ndarray) -> np.ndarray:
    return env * np.sin(2.0 * np.pi * freq * t)


def generate_adsr_sum(freqs: list[float], t: np.ndarray, env: np.ndarray) -> np.ndarray:
    if not freqs:
        return np.zeros_like(t)
    sig = np.zeros_like(t)
    for f in freqs:
        sig += np.sin(2.0 * np.pi * f * t)
    sig /= len(freqs)
    return env * sig


def audio_to_wav_bytes(x: np.ndarray, fs: int) -> bytes:
    peak = np.max(np.abs(x))
    if peak < 1e-12:
        x_norm = x.copy()
    else:
        x_norm = 0.99 * x / peak

    x_int16 = np.int16(np.clip(x_norm, -1.0, 1.0) * 32767)
    buffer = io.BytesIO()
    wav_write(buffer, fs, x_int16)
    buffer.seek(0)
    return buffer.read()


def plot_envelope(t: np.ndarray, env: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t * 1000.0, env, linewidth=1.5)
    ax.set_title("Enveloppe ADSR")
    ax.set_xlabel("Temps (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_signal(t: np.ndarray, x: np.ndarray, title: str, zoom_ms: float = 40.0):
    fs_est = 1.0 / (t[1] - t[0]) if len(t) > 1 else 44100.0
    n = max(10, min(len(t), int((zoom_ms / 1000.0) * fs_est)))

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t[:n] * 1000.0, x[:n], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Temps (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout()
    return fig


# ============================================================
# Initialisation simple
# ============================================================
if "fp_mode" not in st.session_state:
    st.session_state.fp_mode = "Fréquence"
if "fm_mode" not in st.session_state:
    st.session_state.fm_mode = "Fréquence"

if "fp_freq_input" not in st.session_state:
    st.session_state.fp_freq_input = 440.0
if "fm_freq_input" not in st.session_state:
    st.session_state.fm_freq_input = 120.0

if "fp_note_input" not in st.session_state:
    st.session_state.fp_note_input = nearest_note_label_from_freq(440.0)
if "fm_note_input" not in st.session_state:
    st.session_state.fm_note_input = nearest_note_label_from_freq(120.0)

# ============================================================
# Interface d'entrée
# ============================================================
st.subheader("Entrées")

left, right = st.columns(2)

with left:
    st.markdown("### Porteuse")
    fp_mode = st.radio(
        "Mode d'entrée porteuse",
        ["Fréquence", "Note"],
        key="fp_mode",
        horizontal=True,
    )

    fp_freq_input = st.number_input(
        "Fréquence porteuse fp (Hz)",
        min_value=0.001,
        max_value=20000.0,
        step=0.1,
        key="fp_freq_input",
    )

    fp_note_input = st.selectbox(
        "Note porteuse np",
        options=NOTE_LABELS,
        key="fp_note_input",
    )

with right:
    st.markdown("### Modulante")
    fm_mode = st.radio(
        "Mode d'entrée modulante",
        ["Fréquence", "Note"],
        key="fm_mode",
        horizontal=True,
    )

    fm_freq_input = st.number_input(
        "Fréquence modulante fm (Hz)",
        min_value=0.001,
        max_value=20000.0,
        step=0.1,
        key="fm_freq_input",
    )

    fm_note_input = st.selectbox(
        "Note modulante nm",
        options=NOTE_LABELS,
        key="fm_note_input",
    )

# ============================================================
# Résolution des valeurs sans réécriture interdite
# ============================================================
if fp_mode == "Fréquence":
    fp = float(fp_freq_input)
    fp_note = nearest_note_label_from_freq(fp)
else:
    fp_note = fp_note_input
    fp = float(note_freq_from_label(fp_note))

if fm_mode == "Fréquence":
    fm = float(fm_freq_input)
    fm_note = nearest_note_label_from_freq(fm)
else:
    fm_note = fm_note_input
    fm = float(note_freq_from_label(fm_note))

# ============================================================
# Paramètres ADSR communs
# ============================================================
st.subheader("Paramètres ADSR communs")

ad1, ad2, ad3 = st.columns(3)

with ad1:
    fs_audio = st.number_input("Fréquence d'échantillonnage (Hz)", min_value=8000, max_value=96000, value=44100, step=1000)
    attack_ms = st.number_input("Attack (ms)", min_value=1.0, max_value=5000.0, value=10.0, step=1.0)
    decay_ms = st.number_input("Decay (ms)", min_value=1.0, max_value=5000.0, value=100.0, step=1.0)

with ad2:
    sustain_ms = st.number_input("Sustain durée (ms)", min_value=1.0, max_value=5000.0, value=100.0, step=1.0)
    release_ms = st.number_input("Release (ms)", min_value=1.0, max_value=10000.0, value=500.0, step=1.0)

with ad3:
    sustain_level = st.slider("Niveau de sustain", min_value=0.0, max_value=1.0, value=0.7, step=0.01)

# ============================================================
# Résumé
# ============================================================
st.subheader("Valeurs retenues")

c1, c2 = st.columns(2)
with c1:
    st.metric("fp", f"{fp:.3f} Hz")
    st.write(f"Note associée : **{fp_note}**")

with c2:
    st.metric("fm", f"{fm:.3f} Hz")
    st.write(f"Note associée : **{fm_note}**")

# ============================================================
# Tableau de sortie
# ============================================================
rows = []
freqs_12 = []

for i in range(1, 13):
    value = i * fp + fm
    freqs_12.append(value)

    note_label = freq_to_note_label_quarter(value)
    midi_real = freq_to_midi(value)
    midi_q = round_to_quarter_tone_midi(midi_real)
    freq_note = midi_to_freq(midi_q)
    cents_error = 100.0 * (midi_real - midi_q)

    rows.append({
        "i": i,
        "i·fp + fm (Hz)": round(value, 3),
        "Note la plus proche (quart de ton)": note_label,
        "Fréquence de la note (Hz)": round(freq_note, 3),
        "Écart (cents)": round(cents_error, 1),
    })

df = pd.DataFrame(rows)

st.subheader("Tableau de sortie")
st.dataframe(df, use_container_width=True)

# ============================================================
# Génération ADSR commune
# ============================================================
t_adsr, env_adsr = build_adsr_envelope(
    fs=int(fs_audio),
    attack_ms=float(attack_ms),
    decay_ms=float(decay_ms),
    sustain_ms=float(sustain_ms),
    release_ms=float(release_ms),
    sustain_level=float(sustain_level),
)

sig_fp = generate_adsr_sine(fp, t_adsr, env_adsr)
sig_fm = generate_adsr_sine(fm, t_adsr, env_adsr)
sig_sum = generate_adsr_sum(freqs_12, t_adsr, env_adsr)

wav_fp = audio_to_wav_bytes(sig_fp, int(fs_audio))
wav_fm = audio_to_wav_bytes(sig_fm, int(fs_audio))
wav_sum = audio_to_wav_bytes(sig_sum, int(fs_audio))

# ============================================================
# Visualisation
# ============================================================
st.subheader("Visualisation")

g1, g2 = st.columns(2)
with g1:
    st.pyplot(plot_envelope(t_adsr, env_adsr))
with g2:
    st.pyplot(plot_signal(t_adsr, sig_sum, "Somme des 12 sinusoïdes avec ADSR", zoom_ms=40.0))

# ============================================================
# Sorties WAV
# ============================================================
st.subheader("Sorties WAV")

tab1, tab2, tab3 = st.tabs([
    "Signal sur fp",
    "Signal sur fm",
    "Somme des 12 fréquences"
])

with tab1:
    st.pyplot(plot_signal(t_adsr, sig_fp, f"Signal ADSR sur fp = {fp:.3f} Hz", zoom_ms=40.0))
    st.audio(wav_fp, format="audio/wav")
    st.download_button("Télécharger WAV fp", data=wav_fp, file_name="signal_adsr_fp.wav", mime="audio/wav")

with tab2:
    st.pyplot(plot_signal(t_adsr, sig_fm, f"Signal ADSR sur fm = {fm:.3f} Hz", zoom_ms=40.0))
    st.audio(wav_fm, format="audio/wav")
    st.download_button("Télécharger WAV fm", data=wav_fm, file_name="signal_adsr_fm.wav", mime="audio/wav")

with tab3:
    st.pyplot(plot_signal(t_adsr, sig_sum, "Somme des 12 fréquences avec ADSR", zoom_ms=40.0))
    st.audio(wav_sum, format="audio/wav")
    st.download_button("Télécharger WAV somme", data=wav_sum, file_name="signal_adsr_sum_12freqs.wav", mime="audio/wav")