#!/usr/bin/env python3
"""
Example script demonstrating the new colored masks functionality with SAM2.
This script shows how to generate videos with uniquely colored masks that handle overlapping regions.
"""

from pathlib import Path
from video_prediction import SAM2VideoProcessor


def run_colored_masks_example():
    """
    Run an example with colored masks using multiple point prompts.
    """
    
    # Initialize processor
    processor = SAM2VideoProcessor("facebook/sam2-hiera-large")
    
    # Use one of the available videos
    video_path = Path("videos/scene_03_02.mp4")
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        print("Available videos:")
        videos_dir = Path("videos")
        if videos_dir.exists():
            for video in videos_dir.glob("*.mp4"):
                print(f"  - {video.name}")
        return
    
    print(f"🎬 Running colored masks example with: {video_path.name}")
    
    # Define multiple prompts for different objects
    prompts = [
        {
            "prompt_type": "point",
            "points": [[1264, 786]],  # First object
            "labels": [1],
            "frame_idx": 0,
            "obj_id": 1
        },
         {
            "prompt_type": "point",
            "points": [[566, 784]],  # First object
            "labels": [1],
            "frame_idx": 0,
            "obj_id": 2
        },
         {
            "prompt_type": "point",
            "points": [[417, 419]],  # First object
            "labels": [1],
            "frame_idx": 0,
            "obj_id": 3
        },
         {
            "prompt_type": "point",
            "points": [[1062, 376]],  # First object
            "labels": [1],
            "frame_idx": 0,
            "obj_id": 4
        },
         {
            "prompt_type": "point",
            "points": [[1410, 315]],  # First object
            "labels": [1],
            "frame_idx": 0,
            "obj_id": 5
        },
        # {
        #     "prompt_type": "box", 
        #     "box": [1123, 680, 1457, 950],  # Box coordinates [x1, y1, x2, y2]
        #     "labels": [1],
        #     "frame_idx": 0,
        #     "obj_id": 2
        # }
        # {
        #     "prompt_type": "point", 
        #     "points": [[600, 400]],  # Second object
        #     "labels": [1],
        #     "frame_idx": 0,
        #     "obj_id": 2
        # },
        # {
        #     "prompt_type": "point",
        #     "points": [[200, 250]],  # Third object  
        #     "labels": [1],
        #     "frame_idx": 0,
        #     "obj_id": 3
        # }
    ]
    
    try:
        # For demonstration, we'll run individual predictions for each prompt type
        # In a real scenario, you might want to use the multi-object prediction method
        
        frames = None
        all_masks = {}
        temp_dirs = []
        
        for i, prompt in enumerate(prompts):
            print(f"🎯 Processing object {prompt['obj_id']}: {prompt['prompt_type'].upper()}")
            
            if prompt['prompt_type'] == 'point':
                print(f"   Point coordinates: {prompt['points'][0]}")
                frames_i, masks_i, temp_dir_i = processor.predict_video_with_points(
                    video_path=video_path,
                    frame_idx=prompt['frame_idx'],
                    points=prompt['points'],
                    max_frames=60  # Limit frames for faster example
                )
            elif prompt['prompt_type'] == 'box':
                print(f"   Box coordinates: {prompt['box']} (top-left: {prompt['box'][:2]}, bottom-right: {prompt['box'][2:]})")
                frames_i, masks_i, temp_dir_i = processor.predict_video_with_box(
                    video_path=video_path,
                    frame_idx=prompt['frame_idx'],
                    box=prompt['box'],
                    max_frames=60  # Limit frames for faster example
                )
            else:
                print(f"❌ Unknown prompt type: {prompt['prompt_type']}")
                continue
            
            # Store frames from first prediction
            if frames is None:
                frames = frames_i
            
            # Merge masks - reassign object IDs
            for frame_idx, frame_masks in masks_i.items():
                if frame_idx not in all_masks:
                    all_masks[frame_idx] = {}
                
                # Reassign the object ID
                for orig_obj_id, mask in frame_masks.items():
                    all_masks[frame_idx][prompt['obj_id']] = mask
            
            temp_dirs.append(temp_dir_i)
        
        print(f"📊 Combined {len(prompts)} objects across {len(frames)} frames")
        
        # Generate colored mask video
        output_path = "colored_masks_example.mp4"
        processor.save_masks_as_colored_video(
            frames,
            all_masks,
            output_path,
            fps=30
        )
        
        
        # Also save individual frames for inspection
        frames_output_dir = "colored_masks_example_frames"
        processor.save_mask_frames_to_directory(
            frames,
            all_masks,
            output_dir=frames_output_dir,
            save_originals=True,
            save_overlays=True,
            save_binary_masks=True
        )
        
        # Visualize results - collect all prompt points for visualization
        all_prompt_points = []
        for prompt in prompts:
            if prompt['prompt_type'] == 'point':
                all_prompt_points.extend(prompt['points'])
            elif prompt['prompt_type'] == 'box':
                # Add box center point for visualization
                box = prompt['box']
                center_x = (box[0] + box[2]) // 2
                center_y = (box[1] + box[3]) // 2
                all_prompt_points.append([center_x, center_y])
        
        visualization_path = "colored_masks_example_results.png"
        processor.visualize_results(
            frames,
            all_masks,
            prompt_frame_idx=0,
            prompt_points=all_prompt_points,
            output_path=visualization_path
        )
        
        # Clean up temporary directories
        for temp_dir in temp_dirs:
            processor.cleanup_temp_dir(temp_dir)
        
        print(f"\n🎉 Colored masks example completed successfully!")
        print(f"✅ Colored mask video: {output_path}")
        print(f"✅ Individual frames: {frames_output_dir}")
        print(f"✅ Visualization: {visualization_path}")
        
        print(f"\n🎨 Color Strategy:")
        print(f"• Each mask gets a unique color based on prime number combinations")
        print(f"• Overlapping regions get blended colors")
        print(f"• Prime factorization ensures unique identification of overlaps")
        print(f"• Background areas remain black (transparent)")
        
        print(f"\n📦 Prompt Types Used:")
        for prompt in prompts:
            if prompt['prompt_type'] == 'point':
                print(f"• Object {prompt['obj_id']}: Point prompt at {prompt['points'][0]}")
            elif prompt['prompt_type'] == 'box':
                box = prompt['box']
                print(f"• Object {prompt['obj_id']}: Box prompt [{box[0]}, {box[1]}, {box[2]}, {box[3]}]")
                print(f"  └─ Size: {box[2]-box[0]}×{box[3]-box[1]} pixels")
        
    except Exception as e:
        print(f"❌ Error during colored masks example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_colored_masks_example() 