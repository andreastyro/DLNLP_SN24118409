import requests
import re
import csv
import time
import os
from bs4 import BeautifulSoup
import lyricsgenius
import pandas as pd
from tqdm import tqdm

# You'll need to sign up for a Genius API client at https://genius.com/api-clients and get a client access token
# Replace this with your own token after registering
GENIUS_ACCESS_TOKEN = "B-m2BmOCqaI9KDJ2bx06EreK-6sRR-IeLFHjkogv5KJauHck7_4alB5pKGzgx1fF"

# Initialize the Genius API client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=3)
genius.verbose = False  # Turn off status messages
genius.remove_section_headers = True  # Remove section headers (e.g. [Chorus]) from lyrics
genius.skip_non_songs = True  # Skip non-songs (e.g. track lists)

# List of rappers to collect lyrics for
rappers = [
    "Drake"
]

def clean_lyrics(lyrics):
    """Clean up the lyrics text."""
    # Remove Genius-specific artifacts
    lyrics = re.sub(r'\d+Embed', '', lyrics)
    # Remove empty lines and trailing/leading whitespace
    lines = [line.strip() for line in lyrics.split('\n') if line.strip()]
    return lines

def get_artist_songs(artist_name, max_songs=75):
    """Get a list of songs by the artist."""
    try:
        artist = genius.search_artist(artist_name, max_songs=max_songs, sort="popularity")
        print(f"Found {len(artist.songs)} songs for {artist_name}")
        return artist.songs
    except Exception as e:
        print(f"Error fetching songs for {artist_name}: {e}")
        return []

def process_lyrics_to_csv(output_file="drake_lyrics.csv"):
    """Process lyrics from all artists and save to CSV."""
    rows = []
    
    for rapper in tqdm(rappers, desc="Processing artists"):
        print(f"\nFetching songs for {rapper}...")
        songs = get_artist_songs(rapper)
        
        for song in tqdm(songs, desc=f"Processing {rapper}'s songs"):
            try:
                # Skip songs that aren't primarily by this artist
                if rapper.lower() not in song.artist.lower():
                    continue
                
                # Get and clean lyrics
                lyrics_lines = clean_lyrics(song.lyrics)
                
                # Create pairs of current and next lines
                for i in range(len(lyrics_lines) - 1):
                    current_line = lyrics_lines[i]
                    next_line = lyrics_lines[i+1]
                    
                    # Skip section headers or empty lines
                    if (current_line.startswith('[') and current_line.endswith(']')) or not current_line:
                        continue
                    if (next_line.startswith('[') and next_line.endswith(']')) or not next_line:
                        continue
                    
                    rows.append({
                        "Rapper Name": rapper,
                        "Song Title": song.title,
                        "Lyric (current line)": current_line,
                        "Next lyric (next line)": next_line
                    })
                
                # Be respectful of the API rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing {song.title}: {e}")
                continue
    
    # Save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)
    print(f"\nDataset saved to {output_file}")
    print(f"Total entries: {len(df)}")

if __name__ == "__main__":
    print("Starting to collect rapper lyrics...")
    process_lyrics_to_csv()
    print("Done!") 