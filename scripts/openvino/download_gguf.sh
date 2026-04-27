#!/bin/bash

# Check if file argument is provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <ggufs_file>"
  exit 1
fi

GGUFS_FILE=$1
OUTPUT_DIR="models/gguf"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Read URLs from file
while IFS= read -r line; do
  # Skip empty lines and comments
  [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
  
  # Extract URL (trim whitespace)
  URL=$(echo "$line" | xargs)
  
  # Skip if URL is empty
  [[ -z "$URL" ]] && continue
  
  # Extract filename from URL (part after last /)
  FILENAME=$(basename "$URL")
  
  echo "Downloading $FILENAME..."
  wget "$URL" -O "$OUTPUT_DIR/$FILENAME"
done < "$GGUFS_FILE"
