"""
Download a YouTube video's transcript and save it to a text file with timestamps.

Setup (one time):
    pip install youtube-transcript-api

Usage:
    python yt_transcript.py <youtube-link>
    python yt_transcript.py <youtube-link> --lang ta en

Output:
    <video_id>_transcript.txt in the current folder, one line per caption:
    [00:01:23] text spoken at that moment
"""

import argparse
import re
import sys

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)


def extract_video_id(url: str) -> str:
    """Pull the 11-character video ID out of any common YouTube link format."""
    patterns = [
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
        r"/live/([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    raise ValueError(f"Could not find a video ID in: {url}")


def format_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def fetch_transcript(video_id: str, languages: list[str]):
    """
    Try the preferred languages first. If none match, fall back to
    whatever transcript the video has (manual or auto-generated).
    """
    api = YouTubeTranscriptApi()
    try:
        return api.fetch(video_id, languages=languages)
    except NoTranscriptFound:
        available = api.list(video_id)
        first = next(iter(available), None)
        if first is None:
            raise
        print(f"Preferred languages not found. Using: {first.language} ({first.language_code})")
        return first.fetch()


def save_transcript(transcript, output_path: str) -> int:
    """Write each caption line as '[HH:MM:SS] text'. Returns the line count."""
    count = 0
    with open(output_path, "w", encoding="utf-8") as file:
        for snippet in transcript:
            text = snippet.text.replace("\n", " ").strip()
            if not text:
                continue
            file.write(f"[{format_timestamp(snippet.start)}] {text}\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a YouTube transcript with timestamps.")
    parser.add_argument("url", help="YouTube video link or video ID")
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["en", "ta"],
        help="Preferred language codes in order (default: en ta)",
    )
    parser.add_argument("--out", help="Output file name (default: <video_id>_transcript.txt)")
    args = parser.parse_args()

    try:
        video_id = extract_video_id(args.url)
        transcript = fetch_transcript(video_id, args.lang)
    except ValueError as error:
        sys.exit(str(error))
    except TranscriptsDisabled:
        sys.exit("This video has no captions. Use a speech-to-text tool (e.g. Buzz) instead.")
    except VideoUnavailable:
        sys.exit("Video is unavailable (private, deleted, or region-blocked).")
    except NoTranscriptFound:
        sys.exit("No transcript found for this video in any language.")

    output_path = args.out or f"{video_id}_transcript.txt"
    lines = save_transcript(transcript, output_path)
    print(f"Saved {lines} lines to {output_path}")


if __name__ == "__main__":
    main()