#!/bin/bash
set -euo pipefail

source_root=/media/huklab/Data/NPX/Ryansorting/Luke/Luke0804_two_axis_pilot_imec1/quarantine/recordings
archive_root=/media/huklab/Expansion3/DARTsort_archive/luke0804_imec1/two_axis_pilot_complete_older_recordings_20260924
names=(core_depth_strip good_control good_pre_shared neutral_template)

mkdir -p "$archive_root"

for name in "${names[@]}"; do
    test -d "$source_root/$name"
    rsync -rt --partial --info=progress2 "$source_root/$name" "$archive_root/"
done

verification=$(mktemp /tmp/luke0804-archive-verification.XXXXXX)
trap 'rm -f "$verification"' EXIT
for name in "${names[@]}"; do
    rsync -rctn --delete --itemize-changes "$source_root/$name/" "$archive_root/$name/" >> "$verification"
done
test ! -s "$verification"

(
    cd "$archive_root"
    find "${names[@]}" -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS
)

rm -rf -- \
    "$source_root/core_depth_strip" \
    "$source_root/good_control" \
    "$source_root/good_pre_shared" \
    "$source_root/neutral_template"

for name in "${names[@]}"; do
    test ! -e "$source_root/$name"
    test -d "$archive_root/$name"
done

df -B1 /media/huklab/Data /media/huklab/Expansion3
