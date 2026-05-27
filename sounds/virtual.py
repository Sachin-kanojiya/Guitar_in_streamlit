import streamlit as st
import pygame
import cv2
import mediapipe as mp
import time
import numpy as np

# ---------------- PAGE ----------------
st.set_page_config(page_title="Virtual Guitar", layout="centered")

st.title("🎸 Virtual Guitar")

# ---------------- SOUND ----------------
pygame.mixer.init()

chords = {
    "Am": pygame.mixer.Sound("sounds/Am.wav"),
    "C": pygame.mixer.Sound("sounds/C.wav"),
    "D": pygame.mixer.Sound("sounds/D.wav"),
    "G": pygame.mixer.Sound("sounds/G.wav"),
}

last_played = {chord: 0 for chord in chords}

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PLAY FUNCTION ----------------
def play_chord(chord):
    chords[chord].play()
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.history.append(f"{timestamp} - {chord}")

# ---------------- BUTTONS ----------------
st.subheader("Play Guitar Chords")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("Am"):
        play_chord("Am")

with col2:
    if st.button("C"):
        play_chord("C")

with col3:
    if st.button("D"):
        play_chord("D")

with col4:
    if st.button("G"):
        play_chord("G")

# ---------------- HISTORY ----------------
st.subheader("Chord History")

for item in st.session_state.history:
    st.write(item)

# ---------------- WEBCAM ----------------
st.subheader("Hand Detection Guitar")

run = st.checkbox("Start Webcam")

FRAME_WINDOW = st.image([])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = None

if run:
    cap = cv2.VideoCapture(0)

    chord_zones = {
        "Am": (50, 150),
        "C": (151, 250),
        "D": (251, 350),
        "G": (351, 480),
    }

    while run:
        success, frame = cap.read()

        if not success:
            st.error("Webcam not detected")
            break

        frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb)

        # Draw zones
        for chord, (y1, y2) in chord_zones.items():
            cv2.rectangle(frame, (0, y1), (w, y2), (255, 255, 255), 2)
            cv2.putText(
                frame,
                chord,
                (10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2
            )

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                index_tip = hand_landmarks.landmark[8]

                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

                for chord, (y1, y2) in chord_zones.items():

                    if y1 <= y <= y2:

                        if time.time() - last_played[chord] > 1:

                            play_chord(chord)

                            last_played[chord] = time.time()

        FRAME_WINDOW.image(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

    cap.release()