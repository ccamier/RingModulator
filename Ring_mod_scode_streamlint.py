#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 01:20:13 2026

@author: cedriccamier
"""

import io
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
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
    r"""
Cette application illustre la **modulation en anneaux** :

\[
y(t) = x(t)\,m(t)
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

    st.subheader("Partition")
    max_notes_score = st.slider(
        "Nombre max. de notes affichées",
        min_value=2,
        max_value=12,
        value=6,
        step=1,
    )

    show_freq_labels = st.checkbox(
        "Afficher les fréquences sous les notes",
        value=True,
    )

# ============================================================
# Fonctions DSP
# ============================================================
def generate_modulator(t: np.ndarray, freq: float, waveform: str, amplitude: float) -> np.ndarray:
    omega_t = 2.0 * np.pi * freq * t

    if waveform == "Sinusoïdal":
        return amplitude * np.sin(omega_t)
    elif waveform == "Carré":
        return amplitude * signal.square(omega_t)
    elif waveform == "Triangulaire":
        return amplitude * signal.sawtooth(omega_t, width=0.5)
    else:
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
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(t[:n] * 1000.0, sig[:n], linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Temps (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_spectrogram(sig: np.ndarray, fs: int, title: str, fmax: int = 5000):
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
# Fonctions "partition image"
# ============================================================
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
LETTER_TO_DEGREE = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}


def freq_to_midi(freq: float) -> float:
    return 69 + 12 * np.log2(freq / 440.0)


def midi_to_freq(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_note_name(midi: int) -> str:
    pc = midi % 12
    octave = midi // 12 - 1
    return f"{NOTE_NAMES_SHARP[pc]}{octave}"


def midi_to_letter_octave(midi: int):
    name = midi_to_note_name(midi)
    if len(name) == 2:
        letter = name[0]
        accidental = ""
        octave = int(name[1])
    else:
        letter = name[0]
        accidental = "#"
        octave = int(name[2])
    return letter, accidental, octave


def diatonic_index(letter: str, octave: int) -> int:
    # C0 -> 0, D0 -> 1, ..., B0 -> 6, C1 -> 7, etc.
    return octave * 7 + LETTER_TO_DEGREE[letter]


def staff_position_from_midi(midi: int) -> int:
    """
    Position verticale en pas de portée (ligne/interligne).
    Référence : E4 = ligne inférieure de la clé de sol -> position 0.
    """
    letter, _, octave = midi_to_letter_octave(midi)
    ref = diatonic_index("E", 4)
    return diatonic_index(letter, octave) - ref


def produced_frequencies(fc: float, fm: float, waveform: str, max_notes: int = 6):
    """
    Renvoie une liste de dicts :
    {"freq": ..., "weight": ...}
    triée par importance décroissante.
    """
    components = []

    if waveform == "Sinusoïdal":
        freqs = [abs(fc - fm), fc + fm]
        weights = [1.0, 1.0]

        for f, w in zip(freqs, weights):
            if f > 0:
                components.append({"freq": f, "weight": w})

    elif waveform == "Carré":
        odd_harmonics = list(range(1, 2 * max_notes + 1, 2))
        for k in odd_harmonics:
            base_weight = 1.0 / k
            f1 = abs(fc - k * fm)
            f2 = fc + k * fm
            if f1 > 0:
                components.append({"freq": f1, "weight": base_weight})
            if f2 > 0:
                components.append({"freq": f2, "weight": base_weight})

    elif waveform == "Triangulaire":
        odd_harmonics = list(range(1, 2 * max_notes + 1, 2))
        for k in odd_harmonics:
            base_weight = 1.0 / (k * k)
            f1 = abs(fc - k * fm)
            f2 = fc + k * fm
            if f1 > 0:
                components.append({"freq": f1, "weight": base_weight})
            if f2 > 0:
                components.append({"freq": f2, "weight": base_weight})

    # Filtrage bande utile
    components = [c for c in components if 20 <= c["freq"] <= 6000]

    # Fusion si fréquences quasi-identiques
    merged = []
    for c in sorted(components, key=lambda x: x["freq"]):
        if not merged:
            merged.append(c.copy())
        else:
            if abs(c["freq"] - merged[-1]["freq"]) < 1e-6:
                merged[-1]["weight"] += c["weight"]
            else:
                merged.append(c.copy())

    # Tri par importance décroissante
    merged.sort(key=lambda x: x["weight"], reverse=True)

    # Limitation
    merged = merged[:max_notes]

    # Tri final par hauteur croissante pour lecture musicale
    merged.sort(key=lambda x: x["freq"])
    return merged


def render_score_image(components, title="Partition simplifiée", show_freqs=True):
    """
    Dessine une portée simplifiée en clé de sol, comme image matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 100)
    ax.set_ylim(-8, 16)
    ax.axis("off")

    # Titre
    ax.text(2, 14.5, title, fontsize=14, fontweight="bold", ha="left", va="center")
    ax.text(2, 13.0, "Clé de sol – notes tempérées les plus proches", fontsize=10, ha="left", va="center")

    # Portée
    y_staff = [0, 2, 4, 6, 8]  # cinq lignes
    for y in y_staff:
        ax.plot([5, 95], [y, y], color="black", linewidth=1.0)

    # Indication clé de sol textuelle simple
    ax.text(7, 4, "𝄞", fontsize=28, ha="center", va="center")

    if not components:
        ax.text(50, 4, "Aucune composante exploitable", fontsize=12, ha="center", va="center")
        return fig

    xs = np.linspace(20, 90, len(components))

    for x, comp in zip(xs, components):
        freq = comp["freq"]
        midi_float = freq_to_midi(freq)
        midi_round = int(np.round(midi_float))
        midi_round = max(40, min(88, midi_round))  # plage raisonnable

        y = staff_position_from_midi(midi_round)

        # Notehead
        note = Ellipse((x, y), width=2.8, height=1.8, angle=-20,
                       facecolor="black", edgecolor="black")
        ax.add_patch(note)

        # Stem
        ax.plot([x + 1.2, x + 1.2], [y, y + 5], color="black", linewidth=1.2)

        # Lignes supplémentaires
        if y < 0:
            for yl in range(y if y % 2 == 0 else y - 1, -1, 2):
                ax.plot([x - 2.5, x + 2.5], [yl, yl], color="black", linewidth=1.0)
        elif y > 8:
            for yl in range(10, y + 1, 2):
                ax.plot([x - 2.5, x + 2.5], [yl, yl], color="black", linewidth=1.0)

        # Nom de note
        note_name = midi_to_note_name(midi_round)
        cents = 100 * (midi_float - midi_round)

        ax.text(x, -4.8, note_name, fontsize=10, ha="center", va="center")
        ax.text(x, -6.2, f"{cents:+.0f} c", fontsize=8, ha="center", va="center", color="dimgray")

        if show_freqs:
            ax.text(x, -7.4, f"{freq:.1f} Hz", fontsize=8, ha="center", va="center", color="dimgray")

    plt.tight_layout()
    return fig

# ============================================================
# Génération des signaux
# ============================================================
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

carrier = carrier_amp * np.sin(2.0 * np.pi * carrier_freq * t)
modulator = generate_modulator(t, mod_freq, mod_type, mod_amp)
output = carrier * modulator

audio_out = normalize_audio(output) * audio_gain
wav_bytes = to_wav_bytes(audio_out, fs)

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
# Graphiques temporels
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
# Partition image
# ============================================================
st.subheader("Partition des hauteurs principales")

components = produced_frequencies(
    fc=carrier_freq,
    fm=mod_freq,
    waveform=mod_type,
    max_notes=max_notes_score,
)

fig_score = render_score_image(
    components,
    title="Partition simplifiée des composantes fréquentielles",
    show_freqs=show_freq_labels,
)
st.pyplot(fig_score)

with st.expander("Détail des fréquences affichées"):
    rows = []
    for comp in components:
        freq = comp["freq"]
        midi_float = freq_to_midi(freq)
        midi_round = int(np.round(midi_float))
        rows.append({
            "Fréquence (Hz)": round(freq, 3),
            "Note tempérée": midi_to_note_name(midi_round),
            "Écart (cents)": round(100 * (midi_float - midi_round), 1),
            "Poids relatif": round(comp["weight"], 4),
        })
    st.dataframe(rows, use_container_width=True)

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

La partition affiche donc les **deux hauteurs principales** associées.
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

La partition affichée est une **simplification pédagogique** des premières composantes dominantes.
"""
    )
else:
    st.markdown(
        """
Avec un **modulant triangulaire**, on obtient aussi des bandes latérales associées aux **harmoniques impaires**,
mais avec une décroissance plus rapide que dans le cas carré.

La partition affichée représente donc les **premières composantes les plus significatives**.
"""
    )

st.caption(
    "La partition montre les notes tempérées les plus proches des fréquences calculées. "
    "Elle constitue une représentation pédagogique des hauteurs principales, "
    "et non une transcription exhaustive du timbre."
)