import argparse
from pathlib import Path

from discophon.core import COMMONVOICE_TO_ISO6393

from .core import download_benchmark, prepare_downloaded_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Phoneme Discovery benchmark", prog="discophon.prepare")
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="command to run",
    )
    parser_download = subparsers.add_parser(
        "download",
        description="Download benchmark data",
        help="download benchmark data",
    )
    parser_download.add_argument("data", help="path to data directory", type=Path)
    parser_audio = subparsers.add_parser("audio", description="Prepare audio files", help="prepare audio files")
    parser_audio.add_argument("data", help="path to data directory", type=Path)
    parser_audio.add_argument("code", help="CommonVoice language code", type=str, choices=list(COMMONVOICE_TO_ISO6393))
    args = parser.parse_args()
    match args.command:
        case "download":
            download_benchmark(args.data)
        case "audio":
            prepare_downloaded_benchmark(args.data, args.code)
        case _:
            parser.error("Invalid command")
