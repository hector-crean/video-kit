#!/usr/bin/env python3
"""
Example script demonstrating unique ID assignment for multiple prompts.
Each mask gets a unique ID and corresponding alpha value.
"""

from pathlib import Path
from video_prediction import SAM2VideoProcessor


def main():
    """
    Example of multi-object segmentation with unique IDs and alpha values.
    """
    print("🎭 SAM2 Unique IDs Example")
    print("=" * 50)
    
    # Initialize the processor
    processor = SAM2VideoProcessor("facebook/sam2-hiera-large")
    
    # Example video path - replace with your own
    video_path = "videos/scene_03_01.mp4"  # Change this to your video
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print("Please update the video_path variable to point to your video file")
        return
    
    print(f"🎬 Processing video: {video_path}")
    
    # Define multiple prompts - each will get a unique ID and alpha value
    prompts = [
        {
            "frame_idx": 50,  # Middle-ish frame
            "points": [[400, 300]],  # Point in upper-left area
            "labels": [1]  # Foreground
        },
        {
            "frame_idx": 50,  # Same frame
            "points": [[800, 300]],  # Point in upper-right area  
            "labels": [1]  # Foreground
        },
        {
            "frame_idx": 50,  # Same frame
            "points": [[600, 600]],  # Point in lower area
            "labels": [1]  # Foreground
        },
        # {
        #     "frame_idx": 50,  # Same frame
        #     "box": [100, 100, 300, 250]  # Bounding box prompt
        # }
    ]
    
    try:
        # Run multi-object segmentation with unique IDs
        frames, masks, object_id_to_alpha, temp_dir = processor.predict_video_with_unique_ids(
            video_path=video_path,
            prompts=prompts,
            max_frames=30  # Limit frames for faster processing
        )
        
        print(f"\n🎉 Segmentation Results:")
        print(f"📊 Found {len(object_id_to_alpha)} unique objects")
        print(f"🎬 Processed {len(frames)} frames")
        
        # Display object information
        print(f"\n🎨 Object Alpha Mapping:")
        for obj_id, alpha_value in object_id_to_alpha.items():
            print(f"  Object {obj_id:2d}: α = {alpha_value:.3f}")
        
        # Save visualization
        visualization_path = "unique_ids_example_results.png"
        middle_frame = 50 if 50 < len(frames) else len(frames) // 2
        processor.visualize_results(
            frames,
            masks,
            middle_frame,
            prompt_points=None,
            output_path=visualization_path
        )
        
        # Save alpha channel video with unique values per object
        alpha_video_path = "unique_ids_example_alpha.webm"
        processor.save_unique_id_alpha_video(
            frames,
            masks,
            object_id_to_alpha,
            alpha_video_path,
            fps=30
        )
        
        # Save individual frames for analysis
        frames_dir = "unique_ids_example_frames"
        processor.save_mask_frames_to_directory(
            frames,
            masks,
            frames_dir,
            save_originals=True,
            save_overlays=True,
            save_binary_masks=True
        )
        
        # Clean up temporary files
        processor.cleanup_temp_dir(temp_dir)
        
        print(f"\n✅ Example completed successfully!")
        print(f"📸 Visualization: {visualization_path}")
        print(f"🎥 Alpha video: {alpha_video_path}")
        print(f"📁 Frame analysis: {frames_dir}/")
        
        print(f"\n💡 Alpha Channel Benefits:")
        print(f"- Object 1 (α={object_id_to_alpha[1]:.3f}): Can be isolated in post-processing")
        print(f"- Object 2 (α={object_id_to_alpha[2]:.3f}): Different alpha for easy identification")
        print(f"- Object 3 (α={object_id_to_alpha[3]:.3f}): Unique alpha value for compositing")
        if len(object_id_to_alpha) > 3:
            print(f"- Object 4 (α={object_id_to_alpha[4]:.3f}): Box prompt with distinct alpha")
        
        print(f"\n🎬 Usage in Video Editing:")
        print(f"- Import the WebM file into your video editor")
        print(f"- Use alpha channel to isolate specific objects")
        print(f"- Each object can be controlled independently")
        print(f"- Perfect for compositing and visual effects")
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 