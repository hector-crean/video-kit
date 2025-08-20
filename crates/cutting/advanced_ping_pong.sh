#!/bin/bash

# Advanced Ping-pong effect script
# Creates a ping-pong effect with customizable parameters

# Configuration
INPUT_FILE="/Users/hectorcrean/rust/video-kit/crates/cutting/output/Scene_4.1.1.mp4"
OUTPUT_FILE="/Users/hectorcrean/rust/video-kit/crates/cutting/output/Scene_4.1.1_advanced_pingpong.mp4"

# Time points in seconds
START_TIME=2.42
END_TIME=4.07
PING_PONG_DURATION=$(echo "$END_TIME - $START_TIME" | bc -l)

# Number of ping-pong cycles (default: 1)
CYCLES=1

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -i, --input FILE     Input video file (default: $INPUT_FILE)"
    echo "  -o, --output FILE    Output video file (default: $OUTPUT_FILE)"
    echo "  -s, --start TIME     Start time in seconds (default: $START_TIME)"
    echo "  -e, --end TIME       End time in seconds (default: $END_TIME)"
    echo "  -c, --cycles N       Number of ping-pong cycles (default: $CYCLES)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -s 2.42 -e 4.07 -c 3"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -s|--start)
            START_TIME="$2"
            shift 2
            ;;
        -e|--end)
            END_TIME="$2"
            shift 2
            ;;
        -c|--cycles)
            CYCLES="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file '$INPUT_FILE' does not exist!"
    exit 1
fi

# Calculate duration
PING_PONG_DURATION=$(echo "$END_TIME - $START_TIME" | bc -l)

echo "Creating advanced ping-pong effect..."
echo "Input: $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
echo "Ping-pong range: ${START_TIME}s to ${END_TIME}s"
echo "Duration per cycle: ${PING_PONG_DURATION}s"
echo "Number of cycles: $CYCLES"

# Create the advanced ping-pong effect
# This creates multiple ping-pong cycles and extends the audio accordingly
ffmpeg -i "$INPUT_FILE" \
  -filter_complex "
    [0:v]trim=start=${START_TIME}:end=${END_TIME},setpts=PTS-STARTPTS[video_segment];
    [video_segment]reverse[video_reverse];
    [video_segment][video_reverse]concat=n=2:v=1:a=0[ping_pong_base];
    [ping_pong_base]loop=loop=-1:size=$(echo "2 * $CYCLES" | bc)[ping_pong_video];
    [0:a]atrim=start=0:duration=${END_TIME},asetpts=PTS-STARTPTS[audio_normal]
  " \
  -map "[ping_pong_video]" \
  -map "[audio_normal]" \
  -c:v libx264 -preset fast -crf 23 \
  -c:a aac -b:a 128k \
  -y "$OUTPUT_FILE"

echo "Advanced ping-pong effect created successfully!"
echo "Output file: $OUTPUT_FILE"
echo "Total duration: $(echo "$END_TIME * $CYCLES" | bc)s" 