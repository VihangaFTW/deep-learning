import csv


def process_csv_file(csv_path: str = "spotify_lyrics.csv") -> list[str]:
    """
    Process the Spotify lyrics CSV file and extract all lyrics text.

    Args:
        csv_path: Path to the CSV file. Defaults to "spotify_lyrics.csv".

    Returns:
        List of lyrics text strings, one per song.
    """
    lyrics_list = []

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            # Extract the lyrics text from the 'text' column.
            lyrics = row.get("text", "").strip().lower()
            if lyrics:  # Only add non-empty lyrics.
                lyrics_list.append(lyrics)

    return lyrics_list


if __name__ == "__main__":
    lyrics = process_csv_file()
    print(f"Number of entries: {len(lyrics)}")
    print(f"sample lyrics: \n{lyrics[500]}")
