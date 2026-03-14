#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 02:03:25 2026

@author: cedriccamier
"""

import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io.wavfile import write as wav_write

# ============================================================
# Configuration générale
# ============================================================
st.set_page_config(
    page_title="Ring Modulator",
    page_icon="🎛️",
    layout="wide"
)

st.title("Illustration d'un ring modulator")

st.markdown(
r"""
Cette application illustre la **modulation en anneaux** :


y(t) = x(t) \* m(t)


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
        min_value=1,
        max_value=800,
        value=440,
        step=1,
    )

    mod_freq = st.slider(
        "Fréquence du modulant (Hz)",
        min_value=0,
        max_value=1500,
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

    max_components = st.slider(
        "Nombre max. de composantes affichées",
        min_value=2,
        max_value=16,
        value=8,
        step=1,
    )

# ============================================================
# Fonctions DSP
# ============================================================
def generate_modulator(t: np.ndarray, freq: float, waveform: str, amplitude: float) -> np.ndarray:
    omega_t = 2.0 * np.pi * freq * t

    if waveform == "Sinusoïdal":
        return amplitude * np.sin(omega_t)
    if waveform == "Carré":
        return amplitude * signal.square(omega_t)
    if waveform == "Triangulaire":
        return amplitude * signal.sawtooth(omega_t, width=0.5)

    raise ValueError(f"Type de signal non reconnu : {waveform}")


def normalize_audio(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    max_abs = np.max(np.abs(x))
    if max_abs < 1e-12:
        return x.copy()
    return peak * x / max_abs


def to_wav_bytes(x: np.ndarray, fs: int) -> bytes:
    x_clip = np.clip(x, -1.0, 1.0)
    x_int16 = (x_clip * 32767).astype(np.int16)

    buffer = io.BytesIO()
    wav_write(buffer, fs, x_int16)
    buffer.seek(0)
    return buffer.read()


def plot_time_signal(t: np.ndarray, sig: np.ndarray, title: str, fs: int, time_window_s: float = 0.03):
    n = min(len(t), int(time_window_s * fs))
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t[:n] * 1000.0, sig[:n], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Temps (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_spectrogram(sig: np.ndarray, fs: int, title: str, fmax: int = 5000):
    fig, ax = plt.subplots(figsize=(10, 4.2))
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
# Fonctions "hauteurs"
# ============================================================
NOTE_NAMES_SHARP = ["Do", "Do♯", "Ré", "Ré♯", "Mi", "Fa", "Fa♯", "Sol", "Sol♯", "La", "La♯", "Si"]


def freq_to_midi(freq: float) -> float:
    """Convertit une fréquence en numéro MIDI réel."""
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def midi_to_freq(midi_value: float) -> float:
    """Convertit un numéro MIDI réel en fréquence."""
    return 440.0 * (2.0 ** ((midi_value - 69.0) / 12.0))


def midi_int_to_note_name(midi_int: int) -> str:
    """Nom de note tempérée standard."""
    pitch_class = midi_int % 12
    octave = midi_int // 12 - 1
    return f"{NOTE_NAMES_SHARP[pitch_class]}{octave}"


def midi_to_quarter_tone_label(midi_value: float) -> str:
    """
    Arrondi au quart de ton le plus proche.
    Ici, 1/4 de ton = 50 cents = 0,5 demi-ton.
    """
    midi_q = round(midi_value * 2.0) / 2.0
    midi_floor = int(np.floor(midi_q))
    frac = midi_q - midi_floor

    base_note = midi_int_to_note_name(midi_floor)

    if np.isclose(frac, 0.0):
        return base_note
    if np.isclose(frac, 0.5):
        return f"{base_note} + 1/4 ton"

    return base_note


def produced_frequencies(fc: float, fm: float, waveform: str, max_components: int = 8):
    """
    Calcule les composantes fréquentielles principales attendues.
    Renvoie une liste de dicts :
    {"freq": ..., "weight": ..., "origin": ...}
    """
    components = []

    if waveform == "Sinusoïdal":
        candidates = [
            (abs(fc - fm), 1.0, "fc - fm"),
            (fc + fm, 1.0, "fc + fm"),
        ]
        for freq, weight, origin in candidates:
            if freq > 0:
                components.append({"freq": freq, "weight": weight, "origin": origin})

    elif waveform == "Carré":
        # harmoniques impaires
        odd_harmonics = list(range(1, 2 * max_components + 1, 2))
        for k in odd_harmonics:
            weight = 1.0 / k
            f1 = abs(fc - k * fm)
            f2 = fc + k * fm
            if f1 > 0:
                components.append({"freq": f1, "weight": weight, "origin": f"fc - {k}fm"})
            if f2 > 0:
                components.append({"freq": f2, "weight": weight, "origin": f"fc + {k}fm"})

    elif waveform == "Triangulaire":
        # harmoniques impaires, avec décroissance plus rapide
        odd_harmonics = list(range(1, 2 * max_components + 1, 2))
        for k in odd_harmonics:
            weight = 1.0 / (k * k)
            f1 = abs(fc - k * fm)
            f2 = fc + k * fm
            if f1 > 0:
                components.append({"freq": f1, "weight": weight, "origin": f"fc - {k}fm"})
            if f2 > 0:
                components.append({"freq": f2, "weight": weight, "origin": f"fc + {k}fm"})

    # Bande utile
    components = [c for c in components if 20.0 <= c["freq"] <= 6000.0]

    # Fusion des fréquences identiques
    merged = []
    for c in sorted(components, key=lambda x: x["freq"]):
        if not merged:
            merged.append(c.copy())
        else:
            if abs(c["freq"] - merged[-1]["freq"]) < 1e-9:
                merged[-1]["weight"] += c["weight"]
                merged[-1]["origin"] += f" ; {c['origin']}"
            else:
                merged.append(c.copy())

    # tri par poids décroissant
    merged.sort(key=lambda x: x["weight"], reverse=True)
    merged = merged[:max_components]

    # tri final par fréquence croissante pour lecture
    merged.sort(key=lambda x: x["freq"])
    return merged


def components_to_dataframe(components):
    rows = []
    for comp in components:
        freq = comp["freq"]
        midi_real = freq_to_midi(freq)
        midi_q = round(midi_real * 2.0) / 2.0
        freq_q = midi_to_freq(midi_q)
        cents_error = 100.0 * (midi_real - midi_q)

        rows.append({
            "Fréquence calculée (Hz)": round(freq, 3),
            "Note la plus proche (quart de ton)": midi_to_quarter_tone_label(midi_real),
            "Fréquence de cette note (Hz)": round(freq_q, 3),
            "Écart à la note (cents)": round(cents_error, 1),
            "Poids relatif": round(comp["weight"], 4),
            "Origine": comp["origin"],
        })

    return pd.DataFrame(rows)

# ============================================================
# Génération des signaux
# ============================================================
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

carrier = carrier_amp * np.sin(2.0 * np.pi * carrier_freq * t)
modulator = generate_modulator(t, mod_freq, mod_type, mod_amp)
output = carrier * modulator

audio_out = normalize_audio(output) * audio_gain
wav_bytes = to_wav_bytes(audio_out, fs)

components = produced_frequencies(
    fc=carrier_freq,
    fm=mod_freq,
    waveform=mod_type,
    max_components=max_components,
)

df_notes = components_to_dataframe(components)

# ============================================================
# Résumé
# ============================================================
col_info1, col_info2, col_info3 = st.columns(3)

with col_info1:
    st.metric("Porteur", f"{carrier_freq} Hz")

with col_info2:
    st.metric("Modulant", f"{mod_freq} Hz")

with col_info3:
    st.metric("Type de modulant", mod_type)

# ============================================================
# Audio
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
        t, carrier, f"Signal porteur : sinus à {carrier_freq} Hz", fs=fs
    )
    st.pyplot(fig_carrier)

with col2:
    fig_mod = plot_time_signal(
        t, modulator, f"Signal modulant : {mod_type.lower()} à {mod_freq} Hz", fs=fs
    )
    st.pyplot(fig_mod)

fig_output = plot_time_signal(
    t, output, "Signal de sortie : modulation en anneaux", fs=fs
)
st.pyplot(fig_output)

fig_spec = plot_spectrogram(
    output, fs, "Spectrogramme du signal modulé", fmax=max(5000, min(fs // 2, 8000))
)
st.pyplot(fig_spec)

# ============================================================
# Tableau des hauteurs
# ============================================================
st.subheader("Hauteurs calculées")

st.markdown(
    """
Le tableau ci-dessous donne les **composantes fréquentielles principales** attendues
et leur **note la plus proche au quart de ton**.
"""
)

st.dataframe(df_notes, use_container_width=True)

# ============================================================
# Explication
# ============================================================
st.subheader("Interprétation")

if mod_type == "Sinusoïdal":
    lower = abs(carrier_freq - mod_freq)
    upper = carrier_freq + mod_freq
    st.markdown(
        f"""
Avec un **modulant sinusoïdal**, la sortie contient principalement deux composantes :

- \( |f_c - f_m| = |{carrier_freq} - {mod_freq}| = {lower} \,\mathrm{{Hz}} \)
- \( f_c + f_m = {carrier_freq} + {mod_freq} = {upper} \,\mathrm{{Hz}} \)

Le tableau affiche leur conversion en **hauteurs musicales approchées au quart de ton**.
"""
    )

elif mod_type == "Carré":
    st.markdown(
        """
Avec un **modulant carré**, l’onde modulante contient des **harmoniques impaires**.
La sortie contient donc plusieurs bandes latérales du type :

- \( f_c \pm f_m \)
- \( f_c \pm 3f_m \)
- \( f_c \pm 5f_m \), etc.

Le tableau liste les composantes principales retenues.
"""
    )

else:
    st.markdown(
        """
Avec un **modulant triangulaire**, on obtient aussi des composantes liées aux **harmoniques impaires**,
mais avec une décroissance plus rapide que pour l’onde carrée.

Le tableau donne les fréquences principales et leur approximation musicale.
"""
    )

st.caption(
    "Remarque : l'arrondi au quart de ton correspond ici à une quantification par pas de 50 cents."
)