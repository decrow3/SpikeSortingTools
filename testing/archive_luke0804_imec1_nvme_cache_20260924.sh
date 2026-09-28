#!/bin/bash
set -euo pipefail

src=/home/huklab/DARTsort_runs/luke0804-imec1-full-ibllikecmr-float32-nvme-v1
name=luke0804-imec1-full-ibllikecmr-float32-nvme-v1
archive_parent=/media/huklab/Expansion3/DARTsort_archive/luke0804_imec1
partial="$archive_parent/$name.partial"
dest="$archive_parent/$name"
pending="$src.relocation-pending"
source_manifest="$archive_parent/$name.source.sha256"
dest_manifest="$archive_parent/$name.destination.sha256"

exec 9>/tmp/luke0804-imec1-nvme-cache-archive.lock
flock -n 9 || { echo "archive lock is already held" >&2; exit 1; }

mkdir -p "$archive_parent"
test "$(findmnt -T "$archive_parent" -no SOURCE)" = /dev/sdi2
available=$(df -B1 --output=avail "$archive_parent" | tail -1)
test "$available" -ge 600000000000

if test -L "$src"; then
    test "$(readlink -f "$src")" = "$dest"
    test -f "$dest/ARCHIVE_RECEIPT.txt"
    if test -d "$pending"; then
        rm -rf "$pending"
    fi
    echo "archive already complete"
    exit 0
fi

source_tree=$src
if test -d "$pending"; then
    test ! -e "$src"
    source_tree=$pending
fi
test -d "$source_tree"
test -f "$source_tree/traces_cached_seg0.raw"
test "$(stat -c %s "$source_tree/traces_cached_seg0.raw")" = 481360672008

if test ! -d "$dest"; then
    mkdir -p "$partial"
    rsync -rt --no-perms --no-owner --no-group --modify-window=1 \
        --partial --inplace --info=progress2 "$source_tree/" "$partial/"
    sync "$partial"
    mv "$partial" "$dest"
fi

test -f "$dest/traces_cached_seg0.raw"
test "$(stat -c %s "$dest/traces_cached_seg0.raw")" = 481360672008

(cd "$source_tree" && find . -type f -print0 | sort -z | xargs -0 sha256sum) > "$source_manifest.tmp"
mv "$source_manifest.tmp" "$source_manifest"
(cd "$dest" && find . -type f ! -name ARCHIVE_RECEIPT.txt -print0 | sort -z | xargs -0 sha256sum) > "$dest_manifest.tmp"
mv "$dest_manifest.tmp" "$dest_manifest"
cmp "$source_manifest" "$dest_manifest"

printf 'status=verified\nsource=%s\ndestination=%s\nbytes=%s\nverified_utc=%s\nsource_manifest=%s\ndestination_manifest=%s\n' \
    "$src" "$dest" "$(du -s -B1 "$dest" | cut -f1)" "$(date -u +%FT%TZ)" \
    "$source_manifest" "$dest_manifest" > "$dest/ARCHIVE_RECEIPT.txt.tmp"
mv "$dest/ARCHIVE_RECEIPT.txt.tmp" "$dest/ARCHIVE_RECEIPT.txt"

if test "$source_tree" = "$src"; then
    mv "$src" "$pending"
fi
test ! -e "$src"
ln -s "$dest" "$src"
test "$(readlink -f "$src")" = "$dest"
rm -rf "$pending"

df -h / "$archive_parent"
echo "archive, verification, relocation, and compatibility symlink complete"
