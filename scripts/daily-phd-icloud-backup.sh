#!/bin/bash
set -euo pipefail

SOURCE="/Users/timfugl/Desktop/PhD_Ocean_Color_Chemistry"
DESTINATION="/Users/timfugl/Library/Mobile Documents/com~apple~CloudDocs/Backup PhD"

if [[ "$#" -gt 1 || ( "$#" -eq 1 && "$1" != "--dry-run" ) ]]; then
  printf 'Usage: %s [--dry-run]\n' "$0" >&2
  exit 2
fi

if [[ ! -d "$SOURCE" ]]; then
  printf 'Source folder is unavailable: %s\n' "$SOURCE" >&2
  exit 1
fi

mkdir -p "$DESTINATION"
SOURCE_REAL="$(cd "$SOURCE" && pwd -P)"
DESTINATION_REAL="$(cd "$DESTINATION" && pwd -P)"

case "$DESTINATION_REAL" in
  "$SOURCE_REAL"|"$SOURCE_REAL"/*)
    printf 'Refusing to copy into the source folder tree: %s\n' "$DESTINATION_REAL" >&2
    exit 1
    ;;
esac
case "$SOURCE_REAL" in
  "$DESTINATION_REAL"|"$DESTINATION_REAL"/*)
    printf 'Refusing to copy a source nested under the destination: %s\n' "$SOURCE_REAL" >&2
    exit 1
    ;;
esac

if [[ "${1:-}" == "--dry-run" ]]; then
  /usr/bin/rsync -a --delete --dry-run --stats -- "$SOURCE_REAL/" "$DESTINATION_REAL/"
  exit 0
fi

printf '[%s] Starting mirror from "%s" to "%s"\n' \
  "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$SOURCE_REAL" "$DESTINATION_REAL"
/usr/bin/rsync -a --delete -- "$SOURCE_REAL/" "$DESTINATION_REAL/"
printf '[%s] Backup copy completed; iCloud may still be syncing files\n' \
  "$(date '+%Y-%m-%d %H:%M:%S %Z')"
