#!/usr/bin/env python3
"""
Generic SAM2 video inference script with type annotations.
Can be used with any video file by specifying the path.
"""

import argparse
from pathlib import Path
from typing import List, Optional

from video_prediction import SAM2VideoProcessor


def main() -> None:
    """Main function to run SAM2 inference on any video."""
    parser = argparse.ArgumentParser(description="Run SAM2 inference on a video file")
    parser.add_argument(
        "video_path", 
        type=str, 
        help="Path to the video file"
    )
    parser.add_argument(
        "--frame-idx", 
        type=int, 
        default=None,
        help="Frame index to add prompts (default: middle frame)"
    )
    parser.add_argument(
        "--points", 
        type=str, 
        default=None,
        help="Point coordinates as 'x1,y1 x2,y2' (default: center of frame)"
    )
    parser.add_argument(
        "--box", 
        type=str, 
        default=None,
        help="Bounding box as 'x1,y1,x2,y2'"
    )
    parser.add_argument(
        "--max-frames", 
        type=int, 
        default=100,
        help="Maximum number of frames to process (default: 100)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="results",
        help="Output directory name (default: results)"
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        default=True,
        help="Save individual frames to directory (default: True)"
    )
    parser.add_argument(
        "--no-originals",
        action="store_true",
        help="Don't save original frames"
    )
    parser.add_argument(
        "--no-overlays", 
        action="store_true",
        help="Don't save overlay frames"
    )
    parser.add_argument(
        "--no-binary-masks",
        action="store_true", 
        help="Don't save binary mask frames"
    )
    parser.add_argument(
        "--alpha-video",
        action="store_true",
        help="Generate WebM video with masks in alpha channel"
    )
    parser.add_argument(
        "--use-ffmpeg",
        action="store_true", 
        help="Use ffmpeg for better alpha channel support (requires ffmpeg installed)"
    )
    parser.add_argument(
        "--unique-ids",
        action="store_true",
        help="Assign unique IDs to each prompt and encode them as unique alpha values"
    )
    parser.add_argument(
        "--colored-masks",
        action="store_true",
        help="Generate video with uniquely colored masks using prime number combinations"
    )
    parser.add_argument(
        "--web-optimized",
        action="store_true",
        help="Generate web-optimized MP4 with H.264 encoding (requires ffmpeg for best results)"
    )
    parser.add_argument(
        "--quality",
        type=str,
        choices=["high", "medium", "low"],
        default="medium",
        help="Video quality preset (default: medium)"
    )
    parser.add_argument(
        "--direct-encoded",
        action="store_true",
        help="Generate video with direct RGB-to-ID encoding (lossless, perfect decoding)"
    )
    
    args = parser.parse_args()
    
    # Validate video file
    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"🎬 Processing video: {video_path}")
    
    # Initialize processor
    processor = SAM2VideoProcessor("facebook/sam2-hiera-large")
    
    try:
        # Parse prompts
        if args.box:
            # Box prompt
            box_coords = [int(x.strip()) for x in args.box.split(',')]
            if len(box_coords) != 4:
                print("❌ Box format should be 'x1,y1,x2,y2'")
                return
            
            frames, masks, temp_dir = processor.predict_video_with_box(
                video_path=video_path,
                frame_idx=args.frame_idx or len(processor.extract_frames_from_video(video_path, 1)[0]) // 2,
                box=box_coords,
                max_frames=args.max_frames
            )
            
            # Visualize results
            prompt_frame_idx = args.frame_idx or len(frames) // 2
            output_path = f"{args.output}_box_results.png"
            processor.visualize_results(
                frames, 
                masks, 
                prompt_frame_idx,
                prompt_points=None,
                output_path=output_path
            )
            
        else:
            # Point prompt (default)
            if args.points:
                # Parse custom points
                point_pairs = args.points.split()
                points = []
                for pair in point_pairs:
                    x, y = pair.split(',')
                    points.append([int(x.strip()), int(y.strip())])
            else:
                # Use center point by default
                frames_sample, temp_sample_dir, (width, height) = processor.extract_frames_from_video(video_path, 1)
                points = [[width // 2, height // 2]]
                processor.cleanup_temp_dir(temp_sample_dir)
            
            frame_idx = args.frame_idx
            if frame_idx is None:
                # Get total frames to calculate middle
                frames_sample, temp_sample, _ = processor.extract_frames_from_video(video_path, args.max_frames)
                frame_idx = len(frames_sample) // 2
                processor.cleanup_temp_dir(temp_sample)
            
            print(f"📍 Using points: {points} on frame {frame_idx}")
            
            frames, masks, temp_dir = processor.predict_video_with_points(
                video_path=video_path,
                frame_idx=frame_idx,
                points=points,
                max_frames=args.max_frames
            )
            
            # Visualize results
            output_path = f"{args.output}_points_results.png"
            processor.visualize_results(
                frames, 
                masks, 
                frame_idx,
                prompt_points=points,
                output_path=output_path
            )
        
        # Handle colored masks video generation
        if args.colored_masks:
            colored_video_path = f"{args.output}_colored_masks.mp4"
            processor.save_masks_as_colored_video(
                frames,
                masks,
                colored_video_path,
                fps=30
            )
        
        # Handle web-optimized video generation
        if args.web_optimized:
            web_video_path = f"{args.output}_web_optimized.mp4"
            processor.save_masks_as_web_optimized_video(
                frames,
                masks,
                web_video_path,
                fps=30,
                quality=args.quality
            )
        
        # Handle direct encoded video generation
        if args.direct_encoded:
            direct_video_path = f"{args.output}_direct_encoded.webm"
            output_path, prime_mapping = processor.save_masks_as_direct_encoded_video(
                frames,
                masks,
                direct_video_path,
                fps=30,
                use_ffmpeg=True
            )
            print(f"🎯 Direct encoding mapping saved for decoding: {prime_mapping}")
        
        # Handle unique IDs alpha video generation (deprecated approach)
        if args.unique_ids:
            print("⚠️ Unique IDs alpha encoding is deprecated. Use --colored-masks instead for better mask visualization.")
            # Only proceed if the required variable exists (from multi-object prediction)
            if hasattr(processor, 'predict_video_with_unique_ids'):
                print("ℹ️ For unique IDs, use multi-object prediction methods instead")
        
        # Save standard masked video (skip if using colored masks)
        if not args.colored_masks and not args.unique_ids:
            video_output_path = f"{args.output}_masked_video.mp4"
            processor.save_masks_as_video(frames, masks, video_output_path)
        
        # Save alpha channel videos if requested (skip if using colored masks)
        if args.alpha_video and not args.colored_masks and not args.unique_ids:
            if args.use_ffmpeg:
                # Use ffmpeg for high-quality alpha channel support
                alpha_output_path = f"{args.output}_alpha.webm"
                processor.save_masks_as_alpha_video_ffmpeg(
                    frames, 
                    masks, 
                    alpha_output_path,
                    fps=30
                )
            else:
                # Use OpenCV for basic alpha channel support
                alpha_output_path = f"{args.output}_alpha.webm"
                processor.save_masks_as_alpha_video(
                    frames, 
                    masks, 
                    alpha_output_path,
                    fps=30
                )
        
        # Save individual mask frames to directory (if requested)
        if args.save_frames:
            frames_output_dir = f"{args.output}_frames"
            processor.save_mask_frames_to_directory(
                frames, 
                masks, 
                output_dir=frames_output_dir,
                save_originals=not args.no_originals,
                save_overlays=not args.no_overlays, 
                save_binary_masks=not args.no_binary_masks
            )
        
        # Clean up
        processor.cleanup_temp_dir(temp_dir)
        
        print(f"\n🎉 Inference completed successfully!")
        print(f"✅ Processed {len(frames)} frames")
        print(f"✅ Generated masks for {len(masks)} frames")
        print(f"✅ Visualization saved: {output_path}")
        
        if args.colored_masks:
            print(f"✅ Colored mask video saved: {colored_video_path}")
        elif not args.unique_ids:
            print(f"✅ Masked video saved: {video_output_path}")
            if args.alpha_video:
                print(f"✅ Alpha channel video saved: {alpha_output_path}")
        
        if args.web_optimized:
            print(f"✅ Web-optimized video saved: {web_video_path}")
        
        if args.direct_encoded:
            print(f"✅ Direct-encoded video saved: {direct_video_path}")
            print(f"🔢 RGB decoding enabled with prime mapping")
        
        if args.save_frames:
            print(f"✅ Individual frames saved: {frames_output_dir}")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # No arguments provided, run Hero video example
        print("No arguments provided")
    else:
        # Run with command line arguments
        main() 