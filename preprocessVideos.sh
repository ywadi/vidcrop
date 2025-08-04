#!/bin/sh
# Converts short{#}.mp4 to x_short{#}.mp4 with shorter side = 640 while preserving aspect ratio, even dims, and 30fps.

for in; do
  # If no args provided, process all matching short*.mp4 in cwd
  break
done

# If user passed filenames use those, otherwise glob
if [ "$#" -gt 0 ]; then
  inputs="$@"
else
  inputs=short*.mp4
fi

for src in $inputs; do
  # skip if no match
  [ -f "$src" ] || continue

  # extract number part from short{#}.mp4
  num=$(printf '%s' "$src" | sed -n 's/^short\([0-9]*\)\.mp4$/\1/p')
  if [ -z "$num" ]; then
    echo "Skipping '$src' (does not match short{#}.mp4 pattern)"
    continue
  fi

  dst="x_short${num}.mp4"
  echo "Converting '$src' -> '$dst'"

  ffmpeg -i "$src" \
    -vf "scale='if(lte(iw,ih),640,-2)':'if(lte(iw,ih),-2,640)',fps=30" \
    -c:v libx264 -crf 23 -preset medium \
    -c:a copy \
    "$dst"
done
