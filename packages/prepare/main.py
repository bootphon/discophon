from pathlib import Path
from prepare.prepare_multilingual import create_split

# Your paths
root = Path("/store/data/raw_data/commonvoice/cv17/")
textgrid = Path("/store/data/raw_data/commonvoice/cv_textgrid")
output = Path("cv17_ja")
# language_file = Path("/scratch2/mkhentout/multiling/scripts/angelo_multiligPrepare/languages.txt")
language_file = Path("./languages.txt")

# Create output directory if it doesn't exist
output.mkdir(parents=True, exist_ok=True)

# Run the processing
create_split(
    root=root,
    textgrid=textgrid,
    output=output,
    split="test",  # Change to "dev" or "test" as needed
    language_file=language_file,
    target_size=0,  # Target minutes of audio (adjust as needed, 0 for all data)
    min_length=2.0,  # Minimum audio length in seconds
    max_length=15.0,  # Maximum audio length in seconds
    align_suffix=".TextGrid",
)

print("Processing complete!")
print(f"Output files in: {output}")
