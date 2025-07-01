import argparse
import subprocess
from pathlib import Path
import sys
import json


def run(cmd: str) -> None:
    """Run a shell command and raise if it fails."""
    print(f"[ffmpeg] {cmd}")
    subprocess.check_call(cmd, shell=True)


def has_audio(video_path: Path) -> bool:
    """Return True if *video_path* contains at least one audio stream."""
    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(video_path),
        ]
        out = subprocess.check_output(probe_cmd, text=True)
        data = json.loads(out)
        return bool(data.get("streams"))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return False


def make_loop(video_path: Path, keep_reversed: bool = False) -> Path:
    """Create a seamless loop version of *video_path*.

    The function creates a reversed version with ffmpeg and then concatenates
    the original with the reversed clip. A new file called
    `<original_stem>_loop<ext>` is returned.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    stem = video_path.stem
    ext = video_path.suffix  # includes leading dot

    reversed_path = video_path.with_name(f"{stem}_reversed{ext}")
    loop_path = video_path.with_name(f"{stem}_loop{ext}")

    audio_present = has_audio(video_path)

    # 1. Create reversed video
    if audio_present:
        reverse_cmd = (
            f"ffmpeg -y -i \"{video_path}\" "
            f"-vf reverse -af areverse "
            f"-c:v libx264 -preset veryfast -crf 18 "
            f"\"{reversed_path}\""
        )
    else:
        reverse_cmd = (
            f"ffmpeg -y -i \"{video_path}\" "
            f"-vf reverse "
            f"-c:v libx264 -preset veryfast -crf 18 "
            f"-an "  # ensure no implicit audio stream
            f"\"{reversed_path}\""
        )

    run(reverse_cmd)

    # 2. Concatenate original + reversed
    if audio_present:
        concat_cmd = (
            f"ffmpeg -y -i \"{video_path}\" -i \"{reversed_path}\" "
            f"-filter_complex \"[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[v][a]\" "
            f"-map \"[v]\" -map \"[a]\" "
            f"-c:v libx264 -preset veryfast -crf 18 "
            f"\"{loop_path}\""
        )
    else:
        concat_cmd = (
            f"ffmpeg -y -i \"{video_path}\" -i \"{reversed_path}\" "
            f"-filter_complex \"[0:v][1:v]concat=n=2:v=1:a=0[v]\" "
            f"-map \"[v]\" "
            f"-c:v libx264 -preset veryfast -crf 18 "
            f"\"{loop_path}\""
        )

    run(concat_cmd)

    # Optionally delete the reversed intermediate file
    if not keep_reversed:
        try:
            reversed_path.unlink()
        except FileNotFoundError:
            pass

    return loop_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a seamless loop for given videos by reversing and concatenating them."
    )
    parser.add_argument(
        "videos",
        nargs="+",
        help="One or more video files to process.",
    )
    parser.add_argument(
        "--keep-reversed",
        action="store_true",
        help="Keep the intermediate reversed file instead of deleting it.",
    )
    args = parser.parse_args()

    for vid in args.videos:
        try:
            out_path = make_loop(Path(vid), keep_reversed=args.keep_reversed)
            print(f"Created looped video: {out_path}")
        except Exception as exc:
            print(f"Failed to process {vid}: {exc}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main() 