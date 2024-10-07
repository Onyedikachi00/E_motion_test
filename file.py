import streamlit as st
import cv2
from deepface import DeepFace
import time
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import pandas as pd

# Spotify API credentials
client_id = '11aa242441c14a23af907a60c558249e'
client_secret = 'bddb93531191476a8a1790a780b1e5a9'

# Authenticate with Spotify
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

# Function to check internet connection
def check_internet_connection():
    try:
        requests.get('https://www.google.com', timeout=5)
        return True
    except requests.ConnectionError:
        return False

# Function to get top 20 songs based on phrases
def get_songs_by_phrases(phrases):
    query = ' OR '.join(phrases)
    results = sp.search(q=query, type='track', limit=10)
    
    songs = []
    for item in results['tracks']['items']:
        song = {
            'name': item['name'],
            'artist': item['artists'][0]['name'],
            'year': item['album']['release_date'][:4],
            'spotify_link': item['external_urls']['spotify']
        }
        songs.append(song)
    
    return songs

# Initialize the webcam
cap = cv2.VideoCapture(1)  # Try index 1 first
if not cap.isOpened():
    cap = cv2.VideoCapture(0)  # If index 1 fails, try index 0

if not cap.isOpened():
    raise IOError("WebCam Error: Cannot open webcam")

# Initialize emotion countdown variables
emotions_to_detect = ["happy", "sad", "surprise", "neutral", "anger"]
countdown_duration = 3  # Countdown duration in seconds
start_time = None

st.title("K-Emotion Web App")

# Create placeholders for the video frames and countdown
frame_placeholder = st.empty()
countdown_placeholder = st.empty()

# Initialize session state for user input and songs
if 'user_input' not in st.session_state:
    st.session_state.user_input = None
if 'songs' not in st.session_state:
    st.session_state.songs = []
if 'emotion' not in st.session_state:
    st.session_state.emotion = None

# Add buttons for rerun and exit
col1, col2 = st.columns(2)
with col1:
    if st.button('Rerun'):
        st.rerun()
with col2:
    if st.button('Exit'):
        st.stop()

while True:
    ret, frame = cap.read()  # Capture the frame

    if not ret:
        st.error("Error reading frame from webcam")
        break

    try:
        result = DeepFace.analyze(frame, actions=['emotion'])
        dominant_emotion = result[0]['dominant_emotion']
    except ValueError:
        dominant_emotion = "No face detected"

    # Check if the detected emotion matches any of the desired emotions
    if dominant_emotion.lower() in emotions_to_detect:
        if start_time is None:
            start_time = time.time()
        else:
            elapsed_time = time.time() - start_time
            remaining_time = max(countdown_duration - elapsed_time, 0)
            countdown_placeholder.write(f"Countdown: {int(remaining_time)}s")
            if remaining_time <= 0:
                # Take a screenshot and print the emotion
                timestamp_str = time.strftime("%Y-%m-%d_%H-%M-%S")
                img_name = f"{dominant_emotion}_{timestamp_str}.png"
                cv2.imwrite(img_name, frame)
                st.success(f"Detected emotion: {dominant_emotion} (Saved as {img_name})")
                
                # Use the detected emotion to get song recommendations
                st.session_state.emotion = dominant_emotion.lower()
                st.session_state.user_input = None
                st.session_state.songs = []

                if st.session_state.emotion == 'happy':
                    st.write("You seem happy, here are some songs you can dance to.")
                    phrases = ['Feel Good Naija', 'Happy Naija Vibes']
                    st.session_state.songs = get_songs_by_phrases(phrases)
                elif st.session_state.emotion == 'sad':
                    st.write("You look sad, sorry you feel that way. Below is a list of sad songs.")
                    phrases = ['Sad and Calm Naija Songs', 'Naija Heartbreak']
                    st.session_state.songs = get_songs_by_phrases(phrases)
                elif st.session_state.emotion == 'neutral':
                    st.write("Emotion: Neutral. Here are the top 10 9ja songs today on Spotify.")
                    phrases = ['Hot Hits Naija']
                    st.session_state.songs = get_songs_by_phrases(phrases)
                elif st.session_state.emotion == 'angry':
                    st.write("Anger isn't all bad, how you express it is what makes it good or bad. Let's get furious together though😂.")
                    phrases = ['Gangsta rap', '90s rap']
                    st.session_state.songs = get_songs_by_phrases(phrases)
                elif st.session_state.emotion == 'surprise':
                    st.write("Surprise! Don't be, people dey code. Now let's listen to some old school jams.")
                    phrases = ['Old school naija hits']
                    st.session_state.songs = get_songs_by_phrases(phrases)
                else:
                    st.write("Invalid emotion. Please enter one of the following: neutral, happy, sad, angry, surprise.")
                    st.session_state.songs = []

                # Create a DataFrame
                df = pd.DataFrame(st.session_state.songs)
                
                # Merge all columns into one and add Spotify link
                df['Songs'] = df.apply(lambda row: f'<a href="{row["spotify_link"]}" target="_blank">{row["name"]} by {row["artist"]} ({row["year"]})</a>', axis=1)
                
                # Display the DataFrame in Streamlit
                st.markdown("<h3 style='text-align: center;'>Song List</h3>", unsafe_allow_html=True)
                st.write(df[['Songs']].to_html(escape=False, index=False), unsafe_allow_html=True)
                
                break
    else:
        start_time = None
        countdown_placeholder.write("")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(gray, 1.1, 4)

    # Plot the rectangle on the face and display messages
    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
    else:
        cv2.putText(frame, "Keep face still", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Display the frame in the Streamlit app
    frame_placeholder.image(frame_rgb, caption="Emotion Detection", use_column_width=True)

    if cv2.waitKey(2) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
