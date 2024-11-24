import spotipy
from spotipy.oauth2 import SpotifyOAuth
import os
import yaml


def main():
    # Ensure you have your Spotify credentials set up in environment variables:
    # SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
    # If they are already set, you can omit the client_id, client_secret, and redirect_uri parameters.
    sp = spotipy.Spotify(
        auth_manager=SpotifyOAuth(
            scope="playlist-read-private",
            client_id=os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        )
    )

    # Track details to search for
    track_name = "Billie Jean"
    artist_name = "Michael Jackson"

    # Get current user's playlists
    playlists = sp.current_user_playlists()
    print(playlists)
    song_found = False

    while playlists:
        for playlist in playlists["items"]:
            playlist_name = playlist["name"]
            playlist_id = playlist["id"]
            # Get tracks in the playlist
            results = sp.playlist_items(playlist_id)
            tracks = results["items"]
            while results["next"]:
                results = sp.next(results)
                tracks.extend(results["items"])
            # Check if the track is in the playlist
            for item in tracks:
                track = item["track"]
                if track and track["name"].lower() == track_name.lower():
                    # Check if the artist matches
                    for artist in track["artists"]:
                        if artist["name"].lower() == artist_name.lower():
                            print(
                                f'Found "{track_name}" by {artist_name} in playlist "{playlist_name}"'
                            )
                            song_found = True
        if playlists["next"]:
            playlists = sp.next(playlists)
        else:
            playlists = None

    if not song_found:
        print(
            f'"{track_name}" by {artist_name} was not found in your playlists.'
        )


if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    os.environ["SPOTIPY_CLIENT_ID"] = config["spotipy_client_id"]
    os.environ["SPOTIPY_CLIENT_SECRET"] = config["spotipy_client_secret"]
    os.environ["SPOTIPY_REDIRECT_URI"] = config["spotipy_redirect_uri"]
    main()
