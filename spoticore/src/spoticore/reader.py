import csv
import string

from constants import LYRICS_COLUMN, DEFAULT_CSV_PATH


def read_all_lyrics(csv_path: str = DEFAULT_CSV_PATH) -> list[str]:
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
            lyrics = row.get(LYRICS_COLUMN, "").strip().lower()
            if lyrics:  # Only add non-empty lyrics.
                lyrics_list.append(lyrics)

    return lyrics_list


def read_all_unique_words(csv_path: str = DEFAULT_CSV_PATH) -> list[str]:
    """
    Process the Spotify lyrics CSV file and extract all unique individual words.

    Punctuation is removed from words before adding them to the vocabulary.

    Args:
        csv_path: Path to the CSV file. Defaults to "spotify_lyrics.csv".

    Returns:
        List of unique words from all lyrics, with punctuation removed.
    """
    words_list = []
    # Create a translation table that replaces punctuation with spaces.
    translator = str.maketrans({punct: " " for punct in string.punctuation})

    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            lyrics = row.get(LYRICS_COLUMN, "").strip().lower()
            if lyrics:
                # Replace punctuation with spaces before splitting.
                sanitized_lyrics = lyrics.translate(translator)
                # Filter out empty strings created by consecutive spaces.
                words_list.extend([word for word in sanitized_lyrics.split() if word])

    return list(set(words_list))


if __name__ == "__main__":
    # lyrics = read_all_lyrics()
    # print(f"Number of entries: {len(lyrics)}")
    # print(f"sample lyrics: \n{lyrics[500]}")

    words = read_all_unique_words()
    print(f"Number of words: {len(words)}")
    print(f"sample words: \n{words}")
    print("?" in words)
