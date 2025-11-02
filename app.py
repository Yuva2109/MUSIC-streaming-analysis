import time
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pickle
import pandas as pd

# =============== CONFIGURATION ===============
CLIENT_ID = "6d7396640c854e5e9ac849dd33403e2f"
CLIENT_SECRET = "1c6b0452d69549609ee8d4781b2fa55f"
REDIRECT_URI = "https://music-streaming-analysis.onrender.com/callback"
SCOPE = "user-read-recently-played playlist-modify-public playlist-modify-private"

# =============== AUTHENTICATION ===============
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    redirect_uri=REDIRECT_URI,
    scope=SCOPE,
    cache_path=".spotifycache"
))

# =============== LOAD TRAINED MODEL ===============
try:
    with open("mood_model.pkl", "rb") as f:
        model, le = pickle.load(f)
    use_model = True
    print("✅ Loaded trained mood model.")
except Exception as e:
    print("⚠️ No model found or failed to load:", e)
    use_model = False

# =============== HELPER FUNCTIONS ===============
def get_or_create_playlist(sp, user_id, playlist_name):
    """Check if playlist exists; if not, create one."""
    playlists = sp.current_user_playlists(limit=50)
    for playlist in playlists["items"]:
        if playlist["name"].lower() == playlist_name.lower():
            return playlist["id"]
    playlist = sp.user_playlist_create(user_id, playlist_name, public=True)
    return playlist["id"]

def predict_mood(name):
    """Fallback simple mood prediction if ML model not used."""
    n = name.lower()
    if "sad" in n or "slow" in n:
        return "Sad"
    elif "party" in n or "dance" in n:
        return "Happy/Dance"
    elif "love" in n or "romantic" in n:
        return "Calm/Acoustic"
    else:
        return "Neutral"

def analyze_and_update_playlists():
    """Fetch recent songs, predict mood, and add to playlists."""
    user_id = sp.current_user()["id"]
    results = sp.current_user_recently_played(limit=20)

    if not results["items"]:
        print("No recent songs found.")
        return

    tracks = []
    for item in results["items"]:
        track = item["track"]
        name = track["name"]
        artist = track["artists"][0]["name"]
        track_id = track["uri"]
        tracks.append({
            "name": name,
            "artist": artist,
            "uri": track_id
        })

    df = pd.DataFrame(tracks)

    # Use ML model if available
    if use_model:
        # Dummy placeholders for Spotify data
        df["valence"] = 0.5
        df["energy"] = 0.5
        df["danceability"] = 0.5
        df["acousticness"] = 0.5
        df["speechiness"] = 0.5

        features = df[["valence", "energy", "danceability", "acousticness", "speechiness"]]
        predicted = model.predict(features)
        df["mood"] = le.inverse_transform(predicted)
    else:
        df["mood"] = df["name"].apply(predict_mood)

    print("\n🎵 Recent Songs and Moods:")
    print(df[["name", "artist", "mood"]])

    # Create/Update playlists for each mood
    for mood, group in df.groupby("mood"):
        playlist_name = f"{mood} Playlist 🎧"
        playlist_id = get_or_create_playlist(sp, user_id, playlist_name)
        sp.playlist_add_items(playlist_id, list(group["uri"]))
        print(f"✅ Added {len(group)} songs to {playlist_name}")

    print("\n✅ All songs processed successfully!\n")

# =============== LOOP TO RUN EVERY 1 HOUR ===============
if __name__ == "__main__":
    while True:
        print("🔁 Running mood analysis and playlist update...")
        analyze_and_update_playlists()
        print("😴 Sleeping for 1 hour...\n")
        time.sleep(3600)

