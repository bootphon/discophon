import argparse
from pathlib import Path

from discophon.core.languages import commonvoice_languages

from .core import download_benchmark, prepare_downloaded_benchmark

if __name__ == "__main__":
    cv_codes = [lang.iso_639_3 for lang in commonvoice_languages()]
    parser = argparse.ArgumentParser(description="Prepare Phoneme Discovery benchmark", prog="discophon.prepare")
    subparsers = parser.add_subparsers(dest="command", required=True, help="command to run")
    parser_download = subparsers.add_parser(
        "download", description="Download benchmark data", help="download benchmark data"
    )
    parser_download.add_argument("data", help="path to data directory", type=Path)
    parser_audio = subparsers.add_parser("audio", description="Prepare audio files", help="prepare audio files")
    parser_audio.add_argument("data", help="path to data directory", type=Path)
    parser_audio.add_argument("code", help="CommonVoice language ISO 639-3 code", type=str, choices=cv_codes)
    args = parser.parse_args()
    match args.command:
        case "download":
            download_benchmark(args.data)
        case "audio":
            prepare_downloaded_benchmark(args.data, args.code)
        case _:
            parser.error("Invalid command")
