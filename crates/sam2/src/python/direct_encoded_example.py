#!/usr/bin/env python3
"""
Direct RGB-to-ID Encoding Example for SAM2 Video Prediction

This example demonstrates the revolutionary direct RGB encoding where:
1. Each RGB pixel encodes exactly which objects are present
2. Perfect mathematical guarantee of RGB-to-ID decoding via prime factorization
3. Lossless video encoding for data integrity
4. Web-compatible output format

Key Features:
- RGB values directly map to object combinations
- Object 1 → Prime 2 → RGB(2,0,0) 
- Object 2 → Prime 3 → RGB(3,0,0)
- Objects 1+2 → 2×3=6 → RGB(6,0,0)
- Perfect decoding: RGB(6,0,0) → [1,2] via prime factorization

Mathematical Foundation:
- Each object gets a unique prime number
- Overlapping pixels: multiply primes
- Decoding: prime factorization reveals all contributing objects
- 24-bit RGB space allows ~10-12 objects max
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

from video_prediction import SAM2VideoProcessor


def demonstrate_encoding_math():
    """Show the mathematical foundation of the direct encoding."""
    print("🔢 Direct RGB-to-ID Encoding Mathematics")
    print("=" * 50)
    
    # Create processor instance for the methods
    processor = SAM2VideoProcessor("facebook/sam2-hiera-large")
    
    # Demo with 3 objects
    obj_to_prime = processor.generate_object_prime_mapping(3)
    print(f"Prime assignments: {obj_to_prime}")
    
    # Show individual object encodings
    print("\n🎯 Individual Object Encodings:")
    for obj_id, prime in obj_to_prime.items():
        rgb = processor.integer_to_rgb(prime)
        print(f"  Object {obj_id} → Prime {prime} → RGB{rgb}")
    
    # Show combination encodings
    print("\n🔗 Object Combination Encodings:")
    
    # Two objects
    obj1, obj2 = 1, 2
    combo_value = obj_to_prime[obj1] * obj_to_prime[obj2]
    combo_rgb = processor.integer_to_rgb(combo_value)
    decoded = processor.decode_rgb_to_object_ids(*combo_rgb, obj_to_prime)
    print(f"  Objects {obj1}+{obj2} → Value {combo_value} → RGB{combo_rgb} → Decoded: {decoded}")
    
    # Three objects
    obj1, obj2, obj3 = 1, 2, 3
    combo_value = obj_to_prime[obj1] * obj_to_prime[obj2] * obj_to_prime[obj3]
    combo_rgb = processor.integer_to_rgb(combo_value)
    decoded = processor.decode_rgb_to_object_ids(*combo_rgb, obj_to_prime)
    print(f"  Objects {obj1}+{obj2}+{obj3} → Value {combo_value} → RGB{combo_rgb} → Decoded: {decoded}")
    
    # Show mathematical limits
    print(f"\n📊 24-bit RGB Limits:")
    print(f"  Maximum value: 16,777,215 (2^24 - 1)")
    print(f"  3-object product: {combo_value}")
    print(f"  Remaining capacity: {16777215 - combo_value:,}")
    
    print("\n✅ Perfect mathematical guarantee: Every RGB value uniquely decodes to object set!")


def run_direct_encoding_example(video_path: str, output_name: str = "direct_demo"):
    """Run the direct encoding example on a video."""
    
    print("🎬 Direct RGB-to-ID Encoding Video Example")
    print("=" * 50)
    
    # Initialize processor
    processor = SAM2VideoProcessor("facebook/sam2-hiera-large")
    
    # Use multiple points for multiple objects
    sample_frames, temp_dir, (width, height) = processor.extract_frames_from_video(video_path, 5)
    processor.cleanup_temp_dir(temp_dir)
    
    # Create multiple prompts for interesting overlaps
    points = [
        [width // 3, height // 3],      # Top-left region
        [2 * width // 3, height // 3],  # Top-right region
        [width // 2, 2 * height // 3]   # Bottom-center region
    ]
    
    print(f"📍 Using {len(points)} points for multi-object segmentation:")
    for i, point in enumerate(points, 1):
        print(f"  Object {i}: {point}")
    
    # Run prediction with multiple objects
    frames, all_masks, temp_dir = processor.predict_video_with_multiple_objects(
        video_path=video_path,
        frame_idx=len(sample_frames) // 2,
        all_points=points,
        max_frames=60  # Limit for demo
    )
    
    print(f"\n🎯 Predictions completed: {len(frames)} frames, {len(all_masks)} frame results")
    
    # Create direct encoded video
    output_path = f"{output_name}_direct_encoded.webm"
    final_path, prime_mapping = processor.save_masks_as_direct_encoded_video(
        frames,
        all_masks,
        output_path,
        fps=30,
        use_ffmpeg=True
    )
    
    # Create visualization for comparison
    viz_path = f"{output_name}_visualization.png"
    processor.visualize_results(
        frames,
        all_masks,
        len(frames) // 2,
        prompt_points=points,
        output_path=viz_path
    )
    
    # Create regular colored video for comparison
    colored_path = f"{output_name}_colored_comparison.mp4"
    processor.save_masks_as_colored_video(
        frames,
        all_masks,
        colored_path,
        fps=30
    )
    
    # Demo decoding a few pixels
    print(f"\n🔍 Decoding Example Pixels:")
    demonstrate_pixel_decoding(frames[0], all_masks[0], prime_mapping, processor)
    
    # Cleanup
    processor.cleanup_temp_dir(temp_dir)
    
    print(f"\n🎉 Direct Encoding Demo Complete!")
    print(f"✅ Direct encoded video: {final_path}")
    print(f"✅ Visualization: {viz_path}")
    print(f"✅ Colored comparison: {colored_path}")
    print(f"🔢 Prime mapping: {prime_mapping}")
    
    return final_path, prime_mapping


def demonstrate_pixel_decoding(
    frame: np.ndarray, 
    masks: Dict[int, object], 
    prime_mapping: Dict[int, int],
    processor: SAM2VideoProcessor
):
    """Show how to decode specific pixels from the direct encoded video."""
    
    height, width = frame.shape[:2]
    
    # Test a few strategic pixels
    test_pixels = [
        (width // 4, height // 4),      # Likely object 1
        (3 * width // 4, height // 4),  # Likely object 2  
        (width // 2, height // 2),      # Possible overlap
        (width // 2, 3 * height // 4),  # Likely object 3
    ]
    
    for x, y in test_pixels:
        # Find which objects are at this pixel
        objects_at_pixel = []
        for obj_id, mask in masks.items():
            # Convert mask to binary
            mask_tensor = mask[0]
            if hasattr(mask_tensor, "cpu"):
                mask_binary = mask_tensor.cpu().numpy() > 0.5
            else:
                mask_binary = mask_tensor > 0.5
            
            if mask_binary[y, x]:  # Note: y,x for numpy indexing
                objects_at_pixel.append(obj_id)
        
        # Encode to RGB
        if objects_at_pixel:
            r, g, b = processor.encode_objects_to_rgb(objects_at_pixel, prime_mapping)
            # Decode back
            decoded = processor.decode_rgb_to_object_ids(r, g, b, prime_mapping)
            print(f"  Pixel ({x},{y}): Objects {objects_at_pixel} → RGB{(r,g,b)} → Decoded: {decoded}")
        else:
            print(f"  Pixel ({x},{y}): Background → RGB(0,0,0) → Decoded: []")


def main():
    """Main function for the direct encoding example."""
    parser = argparse.ArgumentParser(description="Direct RGB-to-ID Encoding Example")
    parser.add_argument(
        "video_path",
        nargs="?",
        help="Path to video file (optional, will use demo if not provided)"
    )
    parser.add_argument(
        "--output",
        default="direct_demo",
        help="Output name prefix (default: direct_demo)"
    )
    parser.add_argument(
        "--math-only",
        action="store_true",
        help="Only show mathematical demonstration, don't process video"
    )
    
    args = parser.parse_args()
    
    # Always show the math
    demonstrate_encoding_math()
    
    if args.math_only:
        print("\n✅ Mathematical demonstration complete!")
        return
    
    # Process video if provided
    if args.video_path:
        video_path = Path(args.video_path)
        if not video_path.exists():
            print(f"❌ Video file not found: {video_path}")
            return
        
        run_direct_encoding_example(str(video_path), args.output)
    else:
        # Check for demo videos
        demo_videos = list(Path("videos").glob("*.mp4")) if Path("videos").exists() else []
        if demo_videos:
            print(f"\n🎬 Using demo video: {demo_videos[0]}")
            run_direct_encoding_example(str(demo_videos[0]), args.output)
        else:
            print("\n💡 Provide a video path to see the full demonstration:")
            print("   python direct_encoded_example.py your_video.mp4")
            print("   Or place videos in a 'videos/' directory")


if __name__ == "__main__":
    main() 