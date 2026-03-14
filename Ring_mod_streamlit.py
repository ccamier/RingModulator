#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:22:02 2026

@author: cedriccamier
"""

import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write as wav_write

# ============================================================
# Configuration générale de la page
# ============================================================
st.set_page_config(
    page_title="Ring Modulator",
    page_icon="🎛️",
    layout="wide"
)

st.title("Illustration d'un ring modulator")
st.markdown(
    """
Cette application illustre la **modulation en anneaux** :

\[
y(t) = x(t) \\, m(t)
\]

où :
- \(x(t)\) est le **signal porteur**
- \(m(t)\) est le **signal modulant**
- \(y(t)\) est le **signal de sortie**
"""
)

# ============================================================
# Paramètres utilisateur
# ============================================================
with st.sidebar:
    st.header("Paramètres")

    fs = st.slider(
        "Fréquence d'échantillonnage (Hz)",
        min_value=8000,
        max_value=96000,
        value=44100,
        step=1000,
    )

    duration = st.slider(
        "Durée du signal (s)",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

    carrier_freq = st.slider(
        "Fréquence du porteur (Hz)",
        min_value=20,
        max_value=4000,
        value=440,
        step=1,
    )

    mod_freq = st.slider(
        "Fréquence du modulant (Hz)",
        min_value=1,
        max_value=4000,
        value=120,
        step=1,
    )

    mod_type = st.selectbox(
        "Type de signal modulant",
        options=["Sinusoïdal", "Carré", "Triangulaire"],
        index=0,
    )

    carrier_amp = st.slider(
        "Amplitude du porteur",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.01,
    )

    mod_amp = st.slider(
        "Amplitude du modulant",
        min_value=0.0,
        max_value=1.0,
        value=1.0,
        step=0.01,
    )

    audio_gain = st.slider(
        "Gain audio de sortie",
        min_value=0.05,
        max_value=1.0,
        value=0.8,
        step=0.05,
    )

# ============================================================
# Fonctions utilitaires
# ============================================================
def generate_modulator(t: np.ndarray, freq: float, waveform: str, amplitude: float) -> np.ndarray:
    """Génère le signal modulant."""
    omega_t = 2.0 * np.pi * freq * t

    if waveform == "Sinusoïdal":
        return amplitude * np.sin(omega_t)
    elif waveform == "Carré":
        return amplitude * signal.square(omega_t)
    elif waveform == "Triangulaire":
        # width=0.5 -> onde triangulaire
        return amplitude * signal.sawtooth(omega_t, width=0.5)
    else:
        raise ValueError(f"Type de signal non reconnu : {waveform}")


def normalize_audio(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """Normalise un signal audio en flottant dans [-peak, peak]."""
    max_abs = np.max(np.abs(x))
    if max_abs < 1e-12:
        return x.copy()
    return peak * x / max_abs


def to_wav_bytes(x: np.ndarray, fs: int) -> bytes:
    """Convertit un signal float [-1,1] en fichier WAV 16 bits en mémoire."""
    x_clip = np.clip(x, -1.0, 1.0)
    x_int16 = (x_clip * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wav_write(buffer, fs, x_int16)
    buffer.seek(0)
    return buffer.read()


def plot_time_signal(t: np.ndarray, sig: np.ndarray, title: str, time_window_s: float = 0.03):
    """Trace un signal temporel sur une fenêtre réduite pour la lisibilité."""
    n = min(len(t), int(time_window_s * fs))
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t[:n] * 1000.0, sig[:n], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Temps (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_spectrogram(sig: np.ndarray, fs: int, title: str, fmax: int = 5000):
    """Trace un spectrogramme."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    _, _, _, im = ax.specgram(
        sig,
        NFFT=2048,
        Fs=fs,
        noverlap=1536,
        mode="magnitude",
        scale="dB",
    )
    ax.set_title(title)
    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Fréquence (Hz)")
    ax.set_ylim(0, min(fmax, fs // 2))
    fig.colorbar(im, ax=ax, label="Niveau (dB)")
    plt.tight_layout()
    return fig


# ============================================================
# Génération des signaux
# ============================================================
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Porteur : sinus
carrier = carrier_amp * np.sin(2.0 * np.pi * carrier_freq * t)

# Modulant
modulator = generate_modulator(t, mod_freq, mod_type, mod_amp)

# Ring modulation
output = carrier * modulator

# Version audio normalisée
audio_out = normalize_audio(output) * audio_gain
wav_bytes = to_wav_bytes(audio_out, fs)

# ============================================================
# Résumé numérique
# ============================================================
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Porteur", f"{carrier_freq} Hz")

with col_info2:
    st.metric("Modulant", f"{mod_freq} Hz")

with col_info3:
    st.metric("Type de modulant", mod_type)

# ============================================================
# Lecture audio
# ============================================================
st.subheader("Écoute")
st.audio(wav_bytes, format="audio/wav")

# ============================================================
# Graphiques
# ============================================================
st.subheader("Graphiques")

col1, col2 = st.columns(2)

with col1:
    fig_carrier = plot_time_signal(
        t,
        carrier,
        f"Signal porteur : sinus à {carrier_freq} Hz"
    )
    st.pyplot(fig_carrier)

with col2:
    fig_mod = plot_time_signal(
        t,
        modulator,
        f"Signal modulant : {mod_type.lower()} à {mod_freq} Hz"
    )
    st.pyplot(fig_mod)

fig_output = plot_time_signal(
    t,
    output,
    "Signal de sortie : modulation en anneaux"
)
st.pyplot(fig_output)

fig_spec = plot_spectrogram(
    output,
    fs,
    "Spectrogramme du signal modulé",
    fmax=max(5000, min(fs // 2, 8000))
)
st.pyplot(fig_spec)

# ============================================================
# Explication pédagogique
# ============================================================
st.subheader("Interprétation")

if mod_type == "Sinusoïdal":
    lower = abs(carrier_freq - mod_freq)
    upper = carrier_freq + mod_freq
    st.markdown(
        f"""
Avec un **modulant sinusoïdal**, la multiplication produit principalement deux composantes :

- \( |f_c - f_m| = |{carrier_freq} - {mod_freq}| = {lower} \\, \\text{{Hz}} \)
- \( f_c + f_m = {carrier_freq} + {mod_freq} = {upper} \\, \\text{{Hz}} \)

C’est le cas classique du **ring modulation idéal**.
"""
    )
elif mod_type == "Carré":
    st.markdown(
        """
Avec un **modulant carré**, le signal modulant contient des **harmoniques impaires**.
On obtient donc plusieurs bandes latérales du type :

- \( f_c \pm f_m \)
- \( f_c \pm 3 f_m \)
- \( f_c \pm 5 f_m \), etc.

Le spectre est donc plus riche et le résultat sonore plus métallique.
"""
    )
else:
    st.markdown(
        """
Avec un **modulant triangulaire**, le signal contient aussi des **harmoniques impaires**,
mais leur amplitude décroît plus vite que pour l’onde carrée.

Le résultat est donc généralement plus riche qu’un modulant sinusoïdal,
mais souvent moins agressif qu’un modulant carré.
"""
    )

st.caption(
    "Remarque : les formes carrée et triangulaire ici sont générées numériquement à partir "
    "des fonctions standard de SciPy. Elles ne sont pas band-limited, donc un peu d’aliasing "
    "peut apparaître à fréquences élevées."
)