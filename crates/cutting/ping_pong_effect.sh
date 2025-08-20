#!/bin/bash

# Ping-pong effect script
# Creates a ping-pong effect between 2.42s and 4.07s while keeping audio normal

INPUT_FILE="/Users/hectorcrean/rust/video-kit/crates/cutting/output/Scene_4.1.1.mp4"
OUTPUT_FILE="/Users/hectorcrean/rust/video-kit/crates/cutting/output/Scene_4.1.1_pingpong.mp4"

# Time points in seconds
START_TIME=2.42
END_TIME=4.07
PING_PONG_DURATION=$(echo "$END_TIME - $START_TIME" | bc -l)

echo "Creating ping-pong effect..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Ping-pong range: ${START_TIME}s to ${END_TIME}s"
echo "Duration: ${PING_PONG_DURATION}s"

# Create the ping-pong effect using FFmpeg
# This creates a complex filter that:
# 1. Extracts the video segment from 2.42s to 4.07s
# 2. Creates a ping-pong effect by reversing and concatenating
# 3. Keeps the original audio playing normally
# 4. Extends the clip to finish at 4.07s

ffmpeg -i "$INPUT_FILE" \
  -filter_complex "
    [0:v]trim=start=${START_TIME}:end=${END_TIME},setpts=PTS-STARTPTS[video_segment];
    [video_segment]reverse[video_reverse];
    [video_segment][video_reverse]concat=n=2:v=1:a=0[ping_pong_video];
    [0:a]atrim=start=0:duration=${END_TIME},asetpts=PTS-STARTPTS[audio_normal]
  " \
  -map "[ping_pong_video]" \
  -map "[audio_normal]" \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k \
  -y "$OUTPUT_FILE"

echo "Ping-pong effect created successfully!"
echo "Output file: $OUTPUT_FILE" 