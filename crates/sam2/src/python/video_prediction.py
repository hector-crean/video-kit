import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image
import os
import tempfile
import shutil

from sam2.sam2_video_predictor import SAM2VideoPredictor


class SAM2VideoProcessor:
    """
    A wrapper class for SAM2 video prediction with utilities for video processing.
    """

    def __init__(
        self, model_id: str = "facebook/sam2-hiera-large", device: str = "auto"
    ) -> None:
        """
        Initialize the SAM2 video processor.

        Args:
            model_id: The model ID to load from Hugging Face
            device: Device to use ('auto', 'cuda', 'mps', 'cpu')
        """
        # Auto-detect device if needed - Mac-friendly approach
        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.device = device
        print(f"Using device: {self.device}")

        # Load the predictor
        self.predictor = SAM2VideoPredictor.from_pretrained(model_id, device=device)

    def extract_frames_from_video(
        self, video_path: Union[str, Path], max_frames: Optional[int] = None
    ) -> Tuple[List[np.ndarray], str, Tuple[int, int]]:
        """
        Extract frames from video file and save as JPEG files in temp directory.

        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract (None for all)

        Returns:
            Tuple of (frame_list, temp_frames_dir, (width, height))
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create temporary directory for frames
        temp_dir = tempfile.mkdtemp()
        frames_dir = Path(temp_dir) / "frames"
        frames_dir.mkdir(exist_ok=True, parents=True)

        cap = cv2.VideoCapture(str(video_path))

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video properties:")
        print(f"  File: {video_path.name}")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Total frames: {total_frames}")

        # Limit frames if specified
        if max_frames and total_frames > max_frames:
            print(f"  Limiting to {max_frames} frames")
            total_frames = max_frames

        frames = []
        frame_count = 0

        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as JPEG (SAM2 compatible format)
            cv2.imwrite(str(frames_dir / f"{frame_count:05d}.jpg"), frame)

            # Also store RGB version for visualization
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            frame_count += 1

            if frame_count % 50 == 0:
                print(f"  Extracted {frame_count}/{total_frames} frames")

        cap.release()
        print(f"Extracted {frame_count} frames to temporary directory")

        return frames, str(frames_dir), (width, height)

    def load_video_frames(self, video_path: Union[str, Path]) -> List[np.ndarray]:
        """
        Load video frames from a video file.

        Args:
            video_path: Path to the video file

        Returns:
            List of video frames as numpy arrays
        """
        frames, _, _ = self.extract_frames_from_video(video_path)
        return frames

    def load_frames_from_directory(
        self, frames_dir: Union[str, Path]
    ) -> List[np.ndarray]:
        """
        Load video frames from a directory of images.

        Args:
            frames_dir: Path to directory containing frame images

        Returns:
            List of video frames as numpy arrays
        """
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        # Get all image files and sort them
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        image_files = [
            f for f in frames_dir.iterdir() if f.suffix.lower() in image_extensions
        ]
        image_files.sort()

        frames = []
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            frames.append(np.array(img))

        print(f"Loaded {len(frames)} frames from {frames_dir}")
        return frames

    def predict_video_with_points(
        self,
        video_path: Union[str, Path],
        frame_idx: int,
        points: List[List[int]],
        labels: Optional[List[int]] = None,
        obj_id: int = 1,
        max_frames: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[np.ndarray], Dict[int, Dict[int, torch.Tensor]], str]:
        """
        Run SAM2 inference on a video with point prompts.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to add prompts to
            points: List of [x, y] coordinates for point prompts
            labels: List of labels (1 for foreground, 0 for background)
            obj_id: Object ID for tracking
            max_frames: Maximum frames to process (None for all)
            dtype: Data type for inference

        Returns:
            Tuple of (frames, masks_dict, temp_dir_path)
        """
        # Extract frames
        frames, frames_dir, (width, height) = self.extract_frames_from_video(
            video_path, max_frames
        )

        # Default labels if not provided
        if labels is None:
            labels = [1] * len(points)

        # Validate frame index
        if frame_idx >= len(frames):
            frame_idx = len(frames) // 2
            print(f"Adjusted frame index to middle frame: {frame_idx}")

        print(f"Using prompt frame {frame_idx} with points {points}")

        # Set up inference
        dtype = dtype or torch.bfloat16
        if self.device == "cuda":
            autocast_context = torch.autocast("cuda", dtype=dtype)
        else:
            autocast_context = torch.autocast("cpu", dtype=dtype)

        all_masks: Dict[int, Dict[int, torch.Tensor]] = {}

        with torch.inference_mode(), autocast_context:
            # Initialize inference state
            inference_state = self.predictor.init_state(frames_dir)

            # Add point prompts
            result_frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=np.array(points),
                labels=np.array(labels),
            )

            print(f"Initial prediction on frame {result_frame_idx}")
            print(f"Object IDs: {object_ids}")
            print(f"Mask shapes: {[mask.shape for mask in masks]}")

            # Store initial results
            all_masks[result_frame_idx] = {
                obj_id: mask for obj_id, mask in zip(object_ids, masks)
            }

            # Propagate through video
            print("Propagating through video...")
            for (
                prop_frame_idx,
                prop_object_ids,
                prop_masks,
            ) in self.predictor.propagate_in_video(inference_state):
                all_masks[prop_frame_idx] = {
                    obj_id: mask for obj_id, mask in zip(prop_object_ids, prop_masks)
                }
                if prop_frame_idx % 20 == 0:
                    print(f"  Processed frame {prop_frame_idx}")

        return frames, all_masks, frames_dir

    def predict_video_with_box(
        self,
        video_path: Union[str, Path],
        frame_idx: int,
        box: List[int],
        obj_id: int = 1,
        max_frames: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[np.ndarray], Dict[int, Dict[int, torch.Tensor]], str]:
        """
        Run SAM2 inference on a video with box prompt.

        Args:
            video_path: Path to video file
            frame_idx: Frame index to add prompt to
            box: Bounding box as [x1, y1, x2, y2]
            obj_id: Object ID for tracking
            max_frames: Maximum frames to process (None for all)
            dtype: Data type for inference

        Returns:
            Tuple of (frames, masks_dict, temp_dir_path)
        """
        # Extract frames
        frames, frames_dir, (width, height) = self.extract_frames_from_video(
            video_path, max_frames
        )

        # Validate frame index
        if frame_idx >= len(frames):
            frame_idx = len(frames) // 2
            print(f"Adjusted frame index to middle frame: {frame_idx}")

        print(f"Using prompt frame {frame_idx} with box {box}")

        # Set up inference
        dtype = dtype or torch.bfloat16
        if self.device == "cuda":
            autocast_context = torch.autocast("cuda", dtype=dtype)
        else:
            autocast_context = torch.autocast("cpu", dtype=dtype)

        all_masks: Dict[int, Dict[int, torch.Tensor]] = {}

        with torch.inference_mode(), autocast_context:
            # Initialize inference state
            inference_state = self.predictor.init_state(frames_dir)

            # Add box prompt
            result_frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                box=np.array(box),
            )

            print(f"Initial prediction on frame {result_frame_idx}")
            print(f"Object IDs: {object_ids}")
            print(f"Mask shapes: {[mask.shape for mask in masks]}")

            # Store initial results
            all_masks[result_frame_idx] = {
                obj_id: mask for obj_id, mask in zip(object_ids, masks)
            }

            # Propagate through video
            print("Propagating through video...")
            for (
                prop_frame_idx,
                prop_object_ids,
                prop_masks,
            ) in self.predictor.propagate_in_video(inference_state):
                all_masks[prop_frame_idx] = {
                    obj_id: mask for obj_id, mask in zip(prop_object_ids, prop_masks)
                }
                if prop_frame_idx % 20 == 0:
                    print(f"  Processed frame {prop_frame_idx}")

        return frames, all_masks, frames_dir

    def predict_video(
        self,
        video_frames: List[np.ndarray],
        prompts: List[Dict],
        dtype: torch.dtype = None,
    ) -> Dict:
        """
        Perform video prediction using SAM2.

        Args:
            video_frames: List of video frames
            prompts: List of prompt dictionaries with frame_idx, points, labels, etc.
            dtype: Data type for inference

        Returns:
            Dictionary containing prediction results
        """
        results = {"masks": {}, "object_ids": {}, "frame_indices": []}

        # Set appropriate dtype and autocast based on device
        if dtype is None:
            dtype = torch.bfloat16  # Use bfloat16 for all autocast operations

        if self.device == "cuda":
            autocast_context = torch.autocast("cuda", dtype=dtype)
        else:
            # Use CPU autocast for both MPS and CPU devices
            autocast_context = torch.autocast("cpu", dtype=dtype)

        with torch.inference_mode(), autocast_context:
            # Initialize state with video frames
            state = self.predictor.init_state(video_frames)

            # Process each prompt
            for prompt in prompts:
                frame_idx = prompt["frame_idx"]

                if "points" in prompt:
                    # Point prompts
                    points = np.array(prompt["points"])
                    labels = np.array(prompt.get("labels", [1] * len(points)))

                    frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=prompt.get("obj_id", 1),
                        points=points,
                        labels=labels,
                    )

                elif "box" in prompt:
                    # Box prompt
                    box = np.array(prompt["box"])

                    frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                        inference_state=state,
                        frame_idx=frame_idx,
                        obj_id=prompt.get("obj_id", 1),
                        box=box,
                    )

                # Store initial results
                if frame_idx not in results["masks"]:
                    results["masks"][frame_idx] = {}
                    results["object_ids"][frame_idx] = object_ids

                for obj_id, mask in zip(object_ids, masks):
                    results["masks"][frame_idx][obj_id] = mask

            # Propagate through the video
            print("Propagating through video...")
            for frame_idx, object_ids, masks in self.predictor.propagate_in_video(
                state
            ):
                results["frame_indices"].append(frame_idx)

                if frame_idx not in results["masks"]:
                    results["masks"][frame_idx] = {}
                    results["object_ids"][frame_idx] = object_ids

                for obj_id, mask in zip(object_ids, masks):
                    results["masks"][frame_idx][obj_id] = mask

        return results

    def visualize_results(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        prompt_frame_idx: int,
        prompt_points: Optional[List[List[int]]] = None,
        output_path: str = "sam2_results.png",
        show_frames: Optional[List[int]] = None,
    ) -> None:
        """
        Visualize and save the prediction results.

        Args:
            frames: Original video frames
            results: Prediction results masks dictionary
            prompt_frame_idx: Frame index where prompts were added
            prompt_points: Points used for prompting (for visualization)
            output_path: Path to save visualization
            show_frames: Specific frame indices to visualize (None for automatic selection)
        """
        if show_frames is None:
            # Show frames around the prompt frame
            num_frames = len(frames)
            show_frames = [
                max(0, prompt_frame_idx - 50),
                max(0, prompt_frame_idx - 40),
                max(0, prompt_frame_idx - 30),
                max(0, prompt_frame_idx - 20),
                max(0, prompt_frame_idx - 10),
                prompt_frame_idx,
                min(num_frames - 1, prompt_frame_idx + 10),
                min(num_frames - 1, prompt_frame_idx + 20),
                min(num_frames - 1, prompt_frame_idx + 30),
                min(num_frames - 1, prompt_frame_idx + 40),
                min(num_frames - 1, prompt_frame_idx + 50),
            ]
            # Remove duplicates while preserving order
            show_frames = list(dict.fromkeys(show_frames))

        fig, axes = plt.subplots(2, len(show_frames), figsize=(4 * len(show_frames), 8))
        if len(show_frames) == 1:
            axes = axes.reshape(2, 1)

        for i, frame_idx in enumerate(show_frames):
            if frame_idx >= len(frames):
                continue

            frame = frames[frame_idx]

            # Original frame
            axes[0, i].imshow(frame)
            axes[0, i].set_title(f"Original Frame {frame_idx}")
            axes[0, i].axis("off")

            # Add prompt points marker if this is the prompt frame
            if frame_idx == prompt_frame_idx and prompt_points:
                for point in prompt_points:
                    axes[0, i].plot(point[0], point[1], "r*", markersize=12)
                axes[0, i].set_title(f"Frame {frame_idx} (Prompt)")

            # Masked frame
            if frame_idx in results:
                masked_frame = frame.copy()
                for obj_id, mask in results[frame_idx].items():
                    # Move tensor to CPU if needed
                    mask_tensor = mask[0]
                    if hasattr(mask_tensor, "cpu"):
                        mask_binary = mask_tensor.cpu().numpy() > 0.5
                    else:
                        mask_binary = mask_tensor > 0.5

                    # Create colored overlay
                    masked_frame[mask_binary] = [255, 255, 0]  # Yellow mask

                axes[1, i].imshow(masked_frame)
                axes[1, i].set_title(f"Masked Frame {frame_idx}")
            else:
                axes[1, i].imshow(frame)
                axes[1, i].set_title(f"No Mask {frame_idx}")

            axes[1, i].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Visualization saved as '{output_path}'")

    def save_masks_as_video(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Union[str, Path] = "masked_video.mp4",
        fps: int = 30,
    ) -> None:
        """
        Save the masked video as an MP4 file.

        Args:
            frames: Original video frames
            results: Prediction results
            output_path: Output video path
            fps: Frames per second
        """
        output_path = Path(output_path)
        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        for frame_idx, frame in enumerate(frames):
            if frame_idx in results:
                masked_frame = frame.copy()
                for obj_id, mask in results[frame_idx].items():
                    # Move tensor to CPU and apply mask
                    mask_tensor = mask[0]
                    if hasattr(mask_tensor, "cpu"):
                        mask_binary = mask_tensor.cpu().numpy() > 0.5
                    else:
                        mask_binary = mask_tensor > 0.5

                    # Apply mask overlay
                    mask_colored = np.zeros_like(frame)
                    mask_colored[mask_binary] = [255, 0, 0]  # Red mask
                    masked_frame = cv2.addWeighted(
                        masked_frame, 0.7, mask_colored, 0.3, 0
                    )

                # Convert RGB to BGR for OpenCV
                masked_frame_bgr = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
                out.write(masked_frame_bgr)
            else:
                # No mask for this frame, use original
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)

        out.release()
        print(f"Masked video saved to {output_path}")

    def save_mask_frames_to_directory(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_dir: Union[str, Path] = "mask_frames",
        save_originals: bool = True,
        save_overlays: bool = True,
        save_binary_masks: bool = True,
    ) -> None:
        """
        Save individual mask frames to a directory.
        
        Args:
            frames: Original video frames
            results: Prediction results masks dictionary
            output_dir: Directory to save frames
            save_originals: Whether to save original frames
            save_overlays: Whether to save frames with mask overlays
            save_binary_masks: Whether to save binary mask images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        if save_originals:
            originals_dir = output_dir / "originals"
            originals_dir.mkdir(exist_ok=True)
        
        if save_overlays:
            overlays_dir = output_dir / "overlays"
            overlays_dir.mkdir(exist_ok=True)
        
        if save_binary_masks:
            masks_dir = output_dir / "binary_masks"
            masks_dir.mkdir(exist_ok=True)
        
        print(f"Saving mask frames to {output_dir}")
        
        for frame_idx, frame in enumerate(frames):
            # Save original frame
            if save_originals:
                original_path = originals_dir / f"frame_{frame_idx:05d}.png"
                plt.imsave(original_path, frame)
            
            # Create and save masked frame
            if frame_idx in results:
                if save_overlays or save_binary_masks:
                    # Create overlay version
                    overlay_frame = frame.copy()
                    combined_mask = np.zeros(frame.shape[:2], dtype=bool)
                    
                    for obj_id, mask in results[frame_idx].items():
                        # Move tensor to CPU if needed
                        mask_tensor = mask[0]
                        if hasattr(mask_tensor, "cpu"):
                            mask_binary = mask_tensor.cpu().numpy() > 0.5
                        else:
                            mask_binary = mask_tensor > 0.5
                        
                        combined_mask |= mask_binary
                        
                        # Apply colored overlay for this object
                        if save_overlays:
                            # Use different colors for different objects
                            colors = [
                                [255, 255, 0],  # Yellow
                                [255, 0, 255],  # Magenta  
                                [0, 255, 255],  # Cyan
                                [255, 128, 0],  # Orange
                                [128, 255, 0],  # Lime
                            ]
                            color = colors[obj_id % len(colors)]
                            overlay_frame[mask_binary] = (
                                overlay_frame[mask_binary] * 0.6 + np.array(color) * 0.4
                            )
                    
                    if save_overlays:
                        overlay_path = overlays_dir / f"frame_{frame_idx:05d}.png"
                        plt.imsave(overlay_path, overlay_frame.astype(np.uint8))
                    
                    if save_binary_masks:
                        mask_path = masks_dir / f"frame_{frame_idx:05d}.png"
                        plt.imsave(mask_path, combined_mask, cmap='gray')
            
            else:
                # No mask for this frame
                if save_overlays:
                    overlay_path = overlays_dir / f"frame_{frame_idx:05d}.png"
                    plt.imsave(overlay_path, frame)
                
                if save_binary_masks:
                    # Save empty mask
                    empty_mask = np.zeros(frame.shape[:2], dtype=bool)
                    mask_path = masks_dir / f"frame_{frame_idx:05d}.png"
                    plt.imsave(mask_path, empty_mask, cmap='gray')
            
            if frame_idx % 20 == 0:
                print(f"  Saved frame {frame_idx}/{len(frames)}")
        
        print(f"✅ Saved {len(frames)} frames to {output_dir}")
        if save_originals:
            print(f"  📁 Original frames: {originals_dir}")
        if save_overlays:
            print(f"  📁 Overlay frames: {overlays_dir}")  
        if save_binary_masks:
            print(f"  📁 Binary masks: {masks_dir}")
    
    def cleanup_temp_dir(self, temp_dir_path: str) -> None:
        """Clean up temporary directory."""
        if os.path.exists(temp_dir_path):
            shutil.rmtree(os.path.dirname(temp_dir_path))
            print(f"Cleaned up temporary directory")

    def generate_unique_mask_colors(self, num_objects: int) -> Dict[int, Tuple[int, int, int]]:
        """
        Generate unique RGB colors for masks using prime number combinations.
        
        Args:
            num_objects: Number of unique objects to generate colors for
            
        Returns:
            Dictionary mapping object IDs to RGB tuples
        """
        # First few prime numbers for generating unique colors
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
        
        colors = {}
        for obj_id in range(1, num_objects + 1):
            # Use three consecutive primes for RGB channels
            base_idx = (obj_id - 1) % (len(primes) - 2)
            r_prime = primes[base_idx]
            g_prime = primes[base_idx + 1] 
            b_prime = primes[base_idx + 2]
            
            # Scale primes to RGB range (0-255)
            # Use modular arithmetic to ensure good color distribution
            r = (r_prime * 17) % 256
            g = (g_prime * 23) % 256
            b = (b_prime * 29) % 256
            
            # Ensure colors are bright enough to be visible
            r = max(r, 64)
            g = max(g, 64)
            b = max(b, 64)
            
            colors[obj_id] = (r, g, b)
        
        return colors

    def handle_overlapping_masks(
        self, 
        masks_dict: Dict[int, np.ndarray],
        colors_dict: Dict[int, Tuple[int, int, int]]
    ) -> np.ndarray:
        """
        Handle overlapping masks by using prime number multiplication.
        
        Args:
            masks_dict: Dictionary of object_id -> binary mask
            colors_dict: Dictionary of object_id -> RGB color
            
        Returns:
            RGB image with unique colors for overlapping regions
        """
        if not masks_dict:
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # Get dimensions from first mask
        first_mask = next(iter(masks_dict.values()))
        height, width = first_mask.shape
        
        # Create prime value array for each pixel
        prime_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        prime_map = np.ones((height, width), dtype=np.int64)
        
        # Multiply prime values for each overlapping mask
        for obj_id, mask in masks_dict.items():
            if obj_id <= len(prime_values):
                prime = prime_values[obj_id - 1]
                prime_map[mask] *= prime
        
        # Create output image
        output = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Assign colors based on prime factorization
        unique_primes = np.unique(prime_map)
        for prime_product in unique_primes:
            if prime_product == 1:  # No mask
                continue
                
            # Find which objects contribute to this prime product
            contributing_objects = []
            temp_product = prime_product
            
            for obj_id in range(1, len(prime_values) + 1):
                if obj_id in masks_dict:
                    prime = prime_values[obj_id - 1]
                    if temp_product % prime == 0:
                        contributing_objects.append(obj_id)
                        temp_product //= prime
            
            # Calculate blended color for overlapping regions
            if contributing_objects:
                pixels_mask = (prime_map == prime_product)
                
                if len(contributing_objects) == 1:
                    # Single object - use its color
                    obj_id = contributing_objects[0]
                    output[pixels_mask] = colors_dict[obj_id]
                else:
                    # Multiple objects - blend their colors
                    blended_color = np.zeros(3, dtype=np.float32)
                    for obj_id in contributing_objects:
                        blended_color += np.array(colors_dict[obj_id], dtype=np.float32)
                    blended_color /= len(contributing_objects)
                    output[pixels_mask] = blended_color.astype(np.uint8)
        
        return output

    def save_masks_as_colored_video(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Union[str, Path] = "colored_masks_video.mp4",
        fps: int = 30,
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),  # RGBA
        quality: str = "high"  # "high", "medium", "low"
    ) -> None:
        """
        Save video with uniquely colored masks, handling overlapping regions.
        Uses web-friendly codecs for better compression and compatibility.
        
        Args:
            frames: Original video frames
            results: Prediction results containing masks
            output_path: Output video path
            fps: Frames per second
            background_color: RGBA color for areas without masks (default: transparent black)
            quality: Video quality preset ("high", "medium", "low")
        """
        output_path = Path(output_path)
        
        # Ensure output has appropriate extension
        if output_path.suffix.lower() not in ['.mp4', '.webm']:
            output_path = output_path.with_suffix('.mp4')
        
        height, width = frames[0].shape[:2]
        
        # Collect all unique object IDs
        all_obj_ids = set()
        for frame_results in results.values():
            all_obj_ids.update(frame_results.keys())
        
        # Generate unique colors for each object
        colors_dict = self.generate_unique_mask_colors(len(all_obj_ids))
        
        print(f"Generated unique colors for {len(all_obj_ids)} objects:")
        for obj_id in sorted(all_obj_ids):
            color = colors_dict[obj_id]
            print(f"  Object {obj_id}: RGB{color}")
        
        # Setup web-friendly video codec
        success = False
        codecs_to_try = []
        
        if output_path.suffix.lower() == '.webm':
            # WebM codecs in order of preference
            codecs_to_try = [
                ('VP90', 'VP9 (best quality)'),  # VP9 - modern, excellent compression
                ('VP80', 'VP8 (good compatibility)'),  # VP8 - older but widely supported
            ]
        else:  # MP4
            # MP4 codecs in order of preference  
            codecs_to_try = [
                ('H264', 'H.264/AVC (best web compatibility)'),  # H.264 - universal support
                ('avc1', 'H.264/AVC (alternative)'),  # Alternative H.264 fourcc
                ('X264', 'x264 encoder'),  # x264 implementation
                ('mp4v', 'MPEG-4 (fallback)')  # Fallback to old codec
            ]
        
        out = None
        used_codec = None
        
        for fourcc_str, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                if out.isOpened():
                    print(f"✅ Using codec: {codec_name}")
                    used_codec = codec_name
                    success = True
                    break
                else:
                    out.release()
                    
            except Exception as e:
                print(f"⚠️ Codec {codec_name} failed: {e}")
                if out:
                    out.release()
                continue
        
        if not success or not out.isOpened():
            print("❌ Could not initialize video writer with any codec")
            return
        
        print(f"Creating web-optimized colored mask video: {output_path}")
        print(f"📊 Format: {output_path.suffix.upper()}, Codec: {used_codec}")
        print(f"📐 Resolution: {width}×{height}, FPS: {fps}")
        
        frames_written = 0
        for frame_idx, frame in enumerate(frames):
            if frame_idx in results:
                # Convert masks to binary format
                masks_dict = {}
                for obj_id, mask in results[frame_idx].items():
                    # Move tensor to CPU and convert to binary
                    mask_tensor = mask[0]
                    if hasattr(mask_tensor, "cpu"):
                        mask_binary = mask_tensor.cpu().numpy() > 0.5
                    else:
                        mask_binary = mask_tensor > 0.5
                    masks_dict[obj_id] = mask_binary
                
                # Generate colored mask frame
                colored_frame = self.handle_overlapping_masks(masks_dict, colors_dict)
                
                # Apply background color to areas without masks
                if background_color != (0, 0, 0, 0):
                    # Create mask for areas without any object
                    combined_mask = np.zeros((height, width), dtype=bool)
                    for mask in masks_dict.values():
                        combined_mask |= mask
                    
                    # Apply background color to unmasked areas
                    colored_frame[~combined_mask] = background_color[:3]
                
                # Convert RGB to BGR for OpenCV
                colored_frame_bgr = cv2.cvtColor(colored_frame, cv2.COLOR_RGB2BGR)
                out.write(colored_frame_bgr)
                frames_written += 1
                
            else:
                # No mask for this frame - create frame with background color
                if background_color == (0, 0, 0, 0):
                    # Transparent black
                    empty_frame = np.zeros((height, width, 3), dtype=np.uint8)
                else:
                    # Fill with background color
                    empty_frame = np.full((height, width, 3), background_color[:3], dtype=np.uint8)
                
                empty_frame_bgr = cv2.cvtColor(empty_frame, cv2.COLOR_RGB2BGR)
                out.write(empty_frame_bgr)
                frames_written += 1
            
            if frame_idx % 20 == 0:
                print(f"  Processed frame {frame_idx}/{len(frames)}")
        
        out.release()
        
        # Get file size for reporting
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ Web-optimized colored mask video saved to {output_path}")
        print(f"📁 File size: {file_size_mb:.2f} MB ({frames_written} frames)")
        print(f"🌐 Web compatibility: {'Excellent' if 'H.264' in used_codec or 'VP' in used_codec else 'Good'}")

    def rgb_to_integer(self, r: int, g: int, b: int) -> int:
        """Convert RGB values to single 24-bit integer."""
        return (r << 16) | (g << 8) | b  # r*65536 + g*256 + b

    def integer_to_rgb(self, value: int) -> Tuple[int, int, int]:
        """Convert 24-bit integer back to RGB values."""
        r = (value >> 16) & 0xFF  # value // 65536
        g = (value >> 8) & 0xFF   # (value % 65536) // 256  
        b = value & 0xFF          # value % 256
        return (r, g, b)

    def generate_object_prime_mapping(self, num_objects: int) -> Dict[int, int]:
        """
        Generate object ID to prime number mapping for direct RGB encoding.
        
        Args:
            num_objects: Number of unique objects to encode
            
        Returns:
            Dictionary mapping object IDs to prime numbers
            
        Raises:
            ValueError: If too many objects for 24-bit encoding
        """
        # Prime numbers for direct encoding
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        
        if num_objects > len(primes):
            raise ValueError(f"Too many objects ({num_objects}). Maximum supported: {len(primes)}")
        
        # Check if worst-case overlap fits in 24-bit space
        # Worst case: all objects overlap in one pixel
        max_product = 1
        for i in range(num_objects):
            max_product *= primes[i]
            if max_product > 16777215:  # 2^24 - 1
                max_safe_objects = i
                safe_product = max_product // primes[i]
                raise ValueError(
                    f"Too many objects for 24-bit encoding.\n"
                    f"Maximum safe objects: {max_safe_objects} (max value: {safe_product})\n"
                    f"Requested: {num_objects} objects (would need: {max_product})"
                )
        
        # Create mapping
        obj_to_prime = {}
        for obj_id in range(1, num_objects + 1):
            obj_to_prime[obj_id] = primes[obj_id - 1]
        
        return obj_to_prime

    def encode_objects_to_rgb(self, object_ids: List[int], obj_to_prime: Dict[int, int]) -> Tuple[int, int, int]:
        """
        Encode list of object IDs to RGB values using prime multiplication.
        
        Args:
            object_ids: List of object IDs present at this pixel
            obj_to_prime: Object ID to prime number mapping
            
        Returns:
            RGB tuple representing the encoded object combination
        """
        if not object_ids:
            return (0, 0, 0)  # Background pixel
        
        # Multiply primes for all objects
        encoded_value = 1
        for obj_id in object_ids:
            if obj_id in obj_to_prime:
                encoded_value *= obj_to_prime[obj_id]
        
        return self.integer_to_rgb(encoded_value)

    def decode_rgb_to_object_ids(self, r: int, g: int, b: int, obj_to_prime: Dict[int, int]) -> List[int]:
        """
        Decode RGB values back to object IDs using prime factorization.
        
        Args:
            r, g, b: RGB channel values  
            obj_to_prime: Object ID to prime mapping
            
        Returns:
            List of object IDs present at this pixel
        """
        encoded_value = self.rgb_to_integer(r, g, b)
        
        if encoded_value == 0:
            return []  # Background pixel
        
        # Prime factorization to find contributing objects
        contributing_objects = []
        
        for obj_id, prime in obj_to_prime.items():
            if encoded_value % prime == 0:
                contributing_objects.append(obj_id)
                # Remove this prime factor (handle multiple instances)
                while encoded_value % prime == 0:
                    encoded_value //= prime
        
        return sorted(contributing_objects)

    def save_masks_as_direct_encoded_video(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Union[str, Path] = "direct_encoded_masks.webm",
        fps: int = 30,
        use_ffmpeg: bool = True
    ) -> Tuple[str, Dict[int, int]]:
        """
        Save video with direct RGB-encoded object IDs using lossless web-compatible codec.
        Each RGB pixel directly encodes which objects are present via prime factorization.
        
        Args:
            frames: Original video frames
            results: Prediction results containing masks
            output_path: Output video path
            fps: Frames per second
            use_ffmpeg: Use ffmpeg for lossless encoding (recommended)
            
        Returns:
            Tuple of (output_path, object_to_prime_mapping)
        """
        output_path = Path(output_path)
        
        # Force WebM for best lossless web compatibility
        if output_path.suffix.lower() not in ['.webm', '.mkv']:
            output_path = output_path.with_suffix('.webm')
        
        height, width = frames[0].shape[:2]
        
        # Collect all unique object IDs
        all_obj_ids = set()
        for frame_results in results.values():
            all_obj_ids.update(frame_results.keys())
        
        all_obj_ids = sorted(all_obj_ids)
        
        # Generate prime mapping
        try:
            obj_to_prime = self.generate_object_prime_mapping(len(all_obj_ids))
        except ValueError as e:
            print(f"❌ {e}")
            return str(output_path), {}
        
        print(f"🔢 Direct RGB encoding setup:")
        print(f"📊 Objects: {len(all_obj_ids)}, Resolution: {width}×{height}")
        print(f"🧮 Prime assignments:")
        for obj_id in all_obj_ids:
            prime = obj_to_prime[obj_id]
            example_rgb = self.integer_to_rgb(prime)
            print(f"  Object {obj_id} → Prime {prime} → RGB{example_rgb}")
        
        # Show example combinations
        print(f"📈 Example overlaps:")
        if len(all_obj_ids) >= 2:
            obj1, obj2 = all_obj_ids[0], all_obj_ids[1]
            combo_value = obj_to_prime[obj1] * obj_to_prime[obj2]
            combo_rgb = self.integer_to_rgb(combo_value)
            print(f"  Objects {obj1}+{obj2} → Value {combo_value} → RGB{combo_rgb}")
            
            # Verify decoding works
            decoded = self.decode_rgb_to_object_ids(*combo_rgb, obj_to_prime)
            print(f"  Decode test: RGB{combo_rgb} → Objects {decoded} ✅")
        
        if use_ffmpeg and self._check_ffmpeg_available():
            return self._save_direct_encoded_with_ffmpeg(
                frames, results, output_path, fps, obj_to_prime
            )
        else:
            return self._save_direct_encoded_with_opencv(
                frames, results, output_path, fps, obj_to_prime
            )

    def _save_direct_encoded_with_ffmpeg(
        self,
        frames: List[np.ndarray], 
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Path,
        fps: int,
        obj_to_prime: Dict[int, int]
    ) -> Tuple[str, Dict[int, int]]:
        """Save direct encoded video using ffmpeg with lossless VP9."""
        import subprocess
        import tempfile
        
        height, width = frames[0].shape[:2]
        
        print(f"🎬 Creating lossless direct-encoded video with ffmpeg")
        print(f"📊 Codec: VP9 Lossless, Format: WebM")
        
        # Create temporary directory for PNG frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Generate encoded frames
            for frame_idx, frame in enumerate(frames):
                # Create encoded frame (start with black background)
                encoded_frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                if frame_idx in results:
                    # Create object presence map for each pixel
                    for y in range(height):
                        for x in range(width):
                            # Find which objects are present at this pixel
                            objects_at_pixel = []
                            
                            for obj_id, mask in results[frame_idx].items():
                                # Convert mask to binary
                                mask_tensor = mask[0]
                                if hasattr(mask_tensor, "cpu"):
                                    mask_binary = mask_tensor.cpu().numpy() > 0.5
                                else:
                                    mask_binary = mask_tensor > 0.5
                                
                                if mask_binary[y, x]:
                                    objects_at_pixel.append(obj_id)
                            
                            # Encode objects to RGB
                            if objects_at_pixel:
                                r, g, b = self.encode_objects_to_rgb(objects_at_pixel, obj_to_prime)
                                encoded_frame[y, x] = [r, g, b]
                
                # Save frame as PNG (lossless)
                frame_path = temp_dir_path / f"frame_{frame_idx:06d}.png"
                from PIL import Image
                img = Image.fromarray(encoded_frame, 'RGB')
                img.save(frame_path, compress_level=0)  # No compression for speed
                
                if frame_idx % 20 == 0:
                    print(f"  Generated frame {frame_idx}/{len(frames)}")
            
            # FFmpeg command for lossless VP9
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-framerate', str(fps),
                '-i', str(temp_dir_path / 'frame_%06d.png'),
                '-c:v', 'libvpx-vp9',     # VP9 codec
                '-lossless', '1',          # Lossless mode (critical!)
                '-pix_fmt', 'yuv444p',     # Lossless pixel format
                '-crf', '0',               # Best quality
                '-b:v', '0',               # No bitrate limit
                '-speed', '0',             # Best compression (slower)
                str(output_path)
            ]
            
            try:
                print(f"🚀 Encoding lossless VP9...")
                result = subprocess.run(ffmpeg_cmd, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=600)  # 10 minute timeout
                
                if result.returncode == 0:
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"✅ Lossless direct-encoded video saved: {output_path}")
                    print(f"📁 File size: {file_size_mb:.2f} MB")
                    print(f"🌐 Web compatibility: Excellent (VP9 lossless)")
                    print(f"🔒 Data integrity: Perfect (lossless codec)")
                    print(f"🎯 RGB decoding: Enabled")
                    return str(output_path), obj_to_prime
                else:
                    print(f"❌ ffmpeg failed: {result.stderr}")
                    return self._save_direct_encoded_with_opencv(
                        frames, results, output_path, fps, obj_to_prime
                    )
                    
            except subprocess.TimeoutExpired:
                print("⏰ ffmpeg timed out, falling back to OpenCV")
                return self._save_direct_encoded_with_opencv(
                    frames, results, output_path, fps, obj_to_prime
                )

    def _save_direct_encoded_with_opencv(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]], 
        output_path: Path,
        fps: int,
        obj_to_prime: Dict[int, int]
    ) -> Tuple[str, Dict[int, int]]:
        """Fallback: Save direct encoded video using OpenCV (less optimal)."""
        
        height, width = frames[0].shape[:2]
        
        print(f"📹 Using OpenCV fallback (limited lossless options)")
        print(f"⚠️ Warning: OpenCV may not preserve exact RGB values!")
        
        # Try lossless codecs in order of preference
        codecs_to_try = [
            ('FFV1', 'FFV1 (lossless)'),
            ('H264', 'H.264 (high quality)'),
            ('mp4v', 'MPEG-4 (fallback)')
        ]
        
        out = None
        used_codec = None
        
        for fourcc_str, codec_name in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
                
                if out.isOpened():
                    print(f"✅ Using codec: {codec_name}")
                    used_codec = codec_name
                    break
                else:
                    out.release()
                    
            except Exception as e:
                if out:
                    out.release()
                continue
        
        if not out or not out.isOpened():
            print("❌ Could not initialize any video codec")
            return str(output_path), obj_to_prime
        
        frames_written = 0
        for frame_idx, frame in enumerate(frames):
            # Create encoded frame
            encoded_frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            if frame_idx in results:
                # Encode each pixel
                for y in range(height):
                    for x in range(width):
                        objects_at_pixel = []
                        
                        for obj_id, mask in results[frame_idx].items():
                            mask_tensor = mask[0]
                            if hasattr(mask_tensor, "cpu"):
                                mask_binary = mask_tensor.cpu().numpy() > 0.5
                            else:
                                mask_binary = mask_tensor > 0.5
                            
                            if mask_binary[y, x]:
                                objects_at_pixel.append(obj_id)
                        
                        if objects_at_pixel:
                            r, g, b = self.encode_objects_to_rgb(objects_at_pixel, obj_to_prime)
                            encoded_frame[y, x] = [r, g, b]
            
            # Convert RGB to BGR for OpenCV
            encoded_frame_bgr = cv2.cvtColor(encoded_frame, cv2.COLOR_RGB2BGR)
            out.write(encoded_frame_bgr)
            frames_written += 1
            
            if frame_idx % 20 == 0:
                print(f"  Encoded frame {frame_idx}/{len(frames)}")
        
        out.release()
        
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"✅ Direct-encoded video saved: {output_path}")
        print(f"📁 File size: {file_size_mb:.2f} MB ({frames_written} frames)")
        print(f"⚠️ Data integrity: May be compromised (lossy codec)")
        
        return str(output_path), obj_to_prime

    def save_masks_as_alpha_video(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Union[str, Path] = "masked_video_alpha.webm",
        fps: int = 30,
        codec: str = "VP90"
    ) -> None:
        """
        Save video with masks encoded in the alpha channel as WebM with VP9.
        
        Args:
            frames: Original video frames
            results: Prediction results
            output_path: Output video path (should end with .webm)
            fps: Frames per second
            codec: Video codec (VP90 for VP9 with alpha support)
        """
        output_path = Path(output_path)
        
        # Ensure output has .webm extension
        if output_path.suffix.lower() != '.webm':
            output_path = output_path.with_suffix('.webm')
        
        height, width = frames[0].shape[:2]
        
        # VP90 is VP9 with alpha channel support
        fourcc = cv2.VideoWriter_fourcc(*codec)
        
        # Create VideoWriter with alpha channel support
        out = cv2.VideoWriter(
            str(output_path), 
            fourcc, 
            fps, 
            (width, height),
            isColor=True
        )
        
        if not out.isOpened():
            print(f"❌ Failed to open video writer for {output_path}")
            print("Trying alternative codec...")
            # Try alternative codec
            fourcc = cv2.VideoWriter_fourcc(*'VP80')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height), isColor=True)
            
            if not out.isOpened():
                print("❌ Could not create WebM video with alpha channel")
                return
        
        print(f"Creating alpha channel video: {output_path}")
        
        for frame_idx, frame in enumerate(frames):
            # Create RGBA frame
            rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_frame[:, :, :3] = frame  # RGB channels
            
            if frame_idx in results:
                # Combine all masks for this frame
                combined_mask = np.zeros((height, width), dtype=np.float32)
                
                for obj_id, mask in results[frame_idx].items():
                    # Move tensor to CPU and normalize
                    mask_tensor = mask[0]
                    if hasattr(mask_tensor, 'cpu'):
                        mask_array = mask_tensor.cpu().numpy()
                    else:
                        mask_array = mask_tensor.numpy()
                    
                    # Normalize mask to 0-1 range
                    mask_normalized = np.clip(mask_array, 0, 1)
                    combined_mask = np.maximum(combined_mask, mask_normalized)
                
                # Set alpha channel: areas with masks get the mask values, background stays opaque
                # This preserves the original content in areas without masks
                rgba_frame[:, :, 3] = np.where(combined_mask > 0, 
                                               (combined_mask * 255).astype(np.uint8), 
                                               255)
            else:
                # No mask, set alpha to fully opaque to preserve background
                rgba_frame[:, :, 3] = 255
            
            # Convert RGBA to BGRA for OpenCV
            bgra_frame = cv2.cvtColor(rgba_frame, cv2.COLOR_RGBA2BGRA)
            
            # Write frame (OpenCV doesn't directly support RGBA writing, so we use a workaround)
            # For true alpha channel support, we need to use ffmpeg
            out.write(bgra_frame[:, :, :3])  # Write BGR channels only for now
            
            if frame_idx % 20 == 0:
                print(f"  Processed frame {frame_idx}/{len(frames)}")
        
        out.release()
        print(f"✅ Alpha channel video saved to {output_path}")
        print("⚠️  Note: OpenCV has limited alpha channel support. For full alpha transparency, use ffmpeg.")
    
    def save_masks_as_alpha_video_ffmpeg(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Union[str, Path] = "masked_video_alpha.webm",
        fps: int = 30,
        temp_dir: Optional[str] = None
    ) -> None:
        """
        Save video with masks in alpha channel using ffmpeg for true alpha support.
        
        Args:
            frames: Original video frames
            results: Prediction results
            output_path: Output video path
            fps: Frames per second
            temp_dir: Temporary directory for frame files
        """
        import subprocess
        import tempfile as tf
        
        output_path = Path(output_path)
        
        # Create temporary directory for RGBA frames
        if temp_dir is None:
            temp_frame_dir = tf.mkdtemp()
        else:
            temp_frame_dir = temp_dir
        
        rgba_frames_dir = Path(temp_frame_dir) / "rgba_frames"
        rgba_frames_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"Creating RGBA frames in {rgba_frames_dir}")
        
        height, width = frames[0].shape[:2]
        
        # Create RGBA frames
        for frame_idx, frame in enumerate(frames):
            # Create RGBA frame
            rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_frame[:, :, :3] = frame  # RGB channels
            
            if frame_idx in results:
                # Start with full opacity (background areas remain opaque)
                alpha_channel = np.ones((height, width), dtype=np.float32)
                
                # Combine all masks for this frame
                combined_mask = np.zeros((height, width), dtype=np.float32)
                
                for obj_id, mask in results[frame_idx].items():
                    # Move tensor to CPU and normalize
                    mask_tensor = mask[0]
                    if hasattr(mask_tensor, 'cpu'):
                        mask_array = mask_tensor.cpu().numpy()
                    else:
                        mask_array = mask_tensor.numpy()
                    
                    # Normalize mask to 0-1 range
                    mask_normalized = np.clip(mask_array, 0, 1)
                    combined_mask = np.maximum(combined_mask, mask_normalized)
                
                # Set alpha channel: areas with masks get the mask values, background stays opaque
                # This preserves the original content in areas without masks
                rgba_frame[:, :, 3] = (combined_mask * 255).astype(np.uint8)
                # For areas without masks (combined_mask = 0), set them to fully opaque
                rgba_frame[:, :, 3] = np.where(combined_mask > 0, 
                                               (combined_mask * 255).astype(np.uint8), 
                                               255)
            else:
                # No mask, set alpha to fully opaque for background
                rgba_frame[:, :, 3] = 255
            
            # Save as PNG (supports RGBA)
            frame_path = rgba_frames_dir / f"frame_{frame_idx:05d}.png"
            
            # Use PIL to save RGBA
            from PIL import Image
            rgba_pil = Image.fromarray(rgba_frame, 'RGBA')
            rgba_pil.save(frame_path)
            
            if frame_idx % 20 == 0:
                print(f"  Created RGBA frame {frame_idx}/{len(frames)}")
        
        # Use ffmpeg to create WebM with VP9 and alpha channel
        try:
            print(f"Encoding WebM with VP9 and alpha channel...")
            
            # FFmpeg command for VP9 with alpha
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-framerate', str(fps),
                '-i', str(rgba_frames_dir / 'frame_%05d.png'),
                '-c:v', 'libvpx-vp9',  # VP9 codec
                '-pix_fmt', 'yuva420p',  # Pixel format with alpha
                '-auto-alt-ref', '0',  # Disable alt-ref frames for better alpha support
                '-lag-in-frames', '0',  # Reduce latency
                '-crf', '30',  # Quality setting (lower = better quality)
                '-b:v', '0',  # Use CRF rate control
                str(output_path)
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Alpha channel WebM saved to {output_path}")
                
                # Also create a second copy with different settings if requested
                output_path_alt = output_path.with_name(f"{output_path.stem}_alt{output_path.suffix}")
                
                cmd_alt = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', str(rgba_frames_dir / 'frame_%05d.png'),
                    '-c:v', 'libvpx-vp9',
                    '-pix_fmt', 'yuva420p',
                    '-crf', '20',  # Higher quality
                    '-b:v', '0',
                    str(output_path_alt)
                ]
                
                result_alt = subprocess.run(cmd_alt, capture_output=True, text=True)
                if result_alt.returncode == 0:
                    print(f"✅ High quality alpha WebM saved to {output_path_alt}")
                else:
                    print(f"⚠️  Alternative encoding failed: {result_alt.stderr}")
            else:
                print(f"❌ FFmpeg failed: {result.stderr}")
                print("Make sure ffmpeg is installed and libvpx-vp9 codec is available")
                
        except FileNotFoundError:
            print("❌ FFmpeg not found. Please install ffmpeg for alpha channel video support.")
            print("Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")
        
        # Clean up temporary frames if we created the temp dir
        if temp_dir is None:
            import shutil
            shutil.rmtree(temp_frame_dir)
            print("Cleaned up temporary RGBA frames")

    def predict_video_with_unique_ids(
        self,
        video_path: Union[str, Path],
        prompts: List[Dict],
        max_frames: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[np.ndarray], Dict[int, Dict[int, torch.Tensor]], Dict[int, float], str]:
        """
        Run SAM2 inference on a video with multiple prompts, assigning unique IDs and alpha values.

        Args:
            video_path: Path to video file
            prompts: List of prompt dictionaries, each containing:
                - frame_idx: Frame index to add prompts to
                - points: List of [x, y] coordinates (optional)
                - labels: List of labels for points (optional)
                - box: Bounding box as [x1, y1, x2, y2] (optional)
                - obj_id: Object ID (will be overridden with unique sequential IDs)
            max_frames: Maximum frames to process (None for all)
            dtype: Data type for inference

        Returns:
            Tuple of (frames, masks_dict, object_id_to_alpha_dict, temp_dir_path)
        """
        # Extract frames
        frames, frames_dir, (width, height) = self.extract_frames_from_video(
            video_path, max_frames
        )

        print(f"🎭 Running multi-object segmentation with unique IDs")
        print(f"📐 Video resolution: {width}x{height}")
        print(f"🎯 Processing {len(prompts)} prompts")

        # Set up inference
        dtype = dtype or torch.bfloat16
        if self.device == "cuda":
            autocast_context = torch.autocast("cuda", dtype=dtype)
        else:
            autocast_context = torch.autocast("cpu", dtype=dtype)

        all_masks: Dict[int, Dict[int, torch.Tensor]] = {}
        object_id_to_alpha: Dict[int, float] = {}

        with torch.inference_mode(), autocast_context:
            # Initialize inference state
            inference_state = self.predictor.init_state(frames_dir)

            # Process each prompt with a unique sequential ID
            for prompt_idx, prompt in enumerate(prompts):
                unique_obj_id = prompt_idx + 1  # Start from 1
                frame_idx = prompt["frame_idx"]

                # Validate frame index
                if frame_idx >= len(frames):
                    frame_idx = len(frames) // 2
                    print(f"  ⚠️ Adjusted frame index to middle frame: {frame_idx}")

                # Calculate unique alpha value (distributed across 0.1 to 1.0)
                if len(prompts) == 1:
                    alpha_value = 1.0  # Single object gets full opacity
                else:
                    alpha_value = 0.1 + (0.9 * prompt_idx / (len(prompts) - 1))
                
                object_id_to_alpha[unique_obj_id] = alpha_value

                try:
                    if "points" in prompt:
                        # Point prompts
                        points = np.array(prompt["points"])
                        labels = np.array(prompt.get("labels", [1] * len(points)))

                        result_frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=unique_obj_id,
                            points=points,
                            labels=labels,
                        )

                        print(f"  ✅ Object {unique_obj_id}: Point prompts {points.tolist()}, α = {alpha_value:.3f}")

                    elif "box" in prompt:
                        # Box prompt
                        box = np.array(prompt["box"])

                        result_frame_idx, object_ids, masks = self.predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=frame_idx,
                            obj_id=unique_obj_id,
                            box=box,
                        )

                        print(f"  ✅ Object {unique_obj_id}: Box prompt {box.tolist()}, α = {alpha_value:.3f}")

                    else:
                        print(f"  ⚠️ Prompt {prompt_idx} missing 'points' or 'box'")
                        continue

                    # Store initial results
                    if result_frame_idx not in all_masks:
                        all_masks[result_frame_idx] = {}

                    for obj_id, mask in zip(object_ids, masks):
                        if obj_id == unique_obj_id:  # Only store our specific object
                            all_masks[result_frame_idx][obj_id] = mask

                except Exception as e:
                    print(f"  ❌ Failed to process prompt {prompt_idx}: {e}")
                    continue

            # Propagate through video
            if all_masks:
                print("🔄 Propagating segmentation through video...")
                for (prop_frame_idx, prop_object_ids, prop_masks) in self.predictor.propagate_in_video(inference_state):
                    if prop_frame_idx not in all_masks:
                        all_masks[prop_frame_idx] = {}

                    for obj_id, mask in zip(prop_object_ids, prop_masks):
                        if obj_id in object_id_to_alpha:  # Only keep objects we defined
                            all_masks[prop_frame_idx][obj_id] = mask

                    if prop_frame_idx % 20 == 0:
                        print(f"  📹 Propagated to frame {prop_frame_idx}")

        print(f"✨ Multi-object segmentation complete!")
        print(f"📊 Objects found: {len(object_id_to_alpha)}")
        print(f"🎨 Alpha value mapping:")
        for obj_id, alpha in object_id_to_alpha.items():
            print(f"  Object {obj_id}: α = {alpha:.3f}")

        return frames, all_masks, object_id_to_alpha, frames_dir

    def save_unique_id_alpha_video(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        object_id_to_alpha: Dict[int, float],
        output_path: Union[str, Path] = "automatic_segmentation_alpha.webm",
        fps: int = 30,
        temp_dir: Optional[str] = None
    ) -> None:
        """
        Save video where each object has a unique alpha value based on its ID.
        Background areas (without masks) preserve original RGB values with full opacity (alpha = 1.0).
        
        Args:
            frames: Original video frames
            results: Prediction results
            object_id_to_alpha: Mapping from object ID to alpha value
            output_path: Output video path
            fps: Frames per second
            temp_dir: Temporary directory for frame files
        """
        import subprocess
        import tempfile as tf
        
        output_path = Path(output_path)
        
        # Create temporary directory for RGBA frames
        if temp_dir is None:
            temp_frame_dir = tf.mkdtemp()
        else:
            temp_frame_dir = temp_dir
        
        rgba_frames_dir = Path(temp_frame_dir) / "rgba_frames"
        rgba_frames_dir.mkdir(exist_ok=True, parents=True)
        
        print(f"🎨 Creating RGBA frames with unique alpha values...")
        print(f"📁 Frames directory: {rgba_frames_dir}")
        
        height, width = frames[0].shape[:2]
        
        # Create RGBA frames with unique alpha values per object
        for frame_idx, frame in enumerate(frames):
            # Create RGBA frame - preserve original RGB values completely
            rgba_frame = np.zeros((height, width, 4), dtype=np.uint8)
            rgba_frame[:, :, :3] = frame  # Copy original RGB channels exactly
            rgba_frame[:, :, 3] = 255  # Background starts with full opacity (alpha = 1.0)
            
            if frame_idx in results:
                # Create a composite alpha channel based on object priorities
                alpha_channel = np.ones((height, width), dtype=np.float32)  # Start with 1.0 (background)
                
                # Sort objects by ID to ensure consistent layering
                sorted_objects = sorted(results[frame_idx].items())
                
                for obj_id, mask in sorted_objects:
                    if obj_id in object_id_to_alpha:
                        # Move tensor to CPU and normalize
                        mask_tensor = mask[0]
                        if hasattr(mask_tensor, 'cpu'):
                            mask_array = mask_tensor.cpu().numpy()
                        else:
                            mask_array = mask_tensor.numpy()
                        
                        # Normalize mask to 0-1 range
                        mask_normalized = np.clip(mask_array, 0, 1)
                        
                        # Get alpha value for this object
                        alpha_value = object_id_to_alpha[obj_id]
                        
                        # Apply alpha value where mask is active
                        # Background (non-mask areas) remain at 1.0, preserving original RGB
                        alpha_channel = np.where(mask_normalized > 0.5, alpha_value, alpha_channel)
                
                # Set alpha channel (RGB channels remain unchanged from original frame)
                rgba_frame[:, :, 3] = (alpha_channel * 255).astype(np.uint8)
            
            # Save as PNG (supports RGBA)
            frame_path = rgba_frames_dir / f"frame_{frame_idx:05d}.png"
            
            # Use PIL to save RGBA
            from PIL import Image
            rgba_pil = Image.fromarray(rgba_frame, 'RGBA')
            rgba_pil.save(frame_path)
            
            if frame_idx % 20 == 0:
                print(f"  🖼️ Created RGBA frame {frame_idx}/{len(frames)}")
        
        # Use ffmpeg to create WebM with VP9 and alpha channel
        try:
            print(f"🎬 Encoding WebM with VP9 and unique alpha values...")
            
            # FFmpeg command for VP9 with alpha
            cmd = [
                'ffmpeg', '-y',  # -y to overwrite output file
                '-framerate', str(fps),
                '-i', str(rgba_frames_dir / 'frame_%05d.png'),
                '-c:v', 'libvpx-vp9',  # VP9 codec
                '-pix_fmt', 'yuva420p',  # Pixel format with alpha
                '-auto-alt-ref', '0',  # Disable alt-ref frames for better alpha support
                '-lag-in-frames', '0',  # Reduce latency
                '-crf', '25',  # Quality setting (lower = better quality)
                '-b:v', '0',  # Use CRF rate control
                str(output_path)
            ]
            
            # Run ffmpeg
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Automatic segmentation alpha video saved to {output_path}")
                print(f"🎨 Each object has a unique alpha value from {min(object_id_to_alpha.values()):.3f} to {max(object_id_to_alpha.values()):.3f}")
            else:
                print(f"❌ FFmpeg failed: {result.stderr}")
                print("Make sure ffmpeg is installed and libvpx-vp9 codec is available")
                
        except FileNotFoundError:
            print("❌ FFmpeg not found. Please install ffmpeg for alpha channel video support.")
            print("Install with: brew install ffmpeg (macOS) or apt install ffmpeg (Ubuntu)")
        
        # Clean up temporary frames if we created the temp dir
        if temp_dir is None:
            import shutil
            shutil.rmtree(temp_frame_dir)
            print("🧹 Cleaned up temporary RGBA frames")

    def save_masks_as_web_optimized_video(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Union[str, Path] = "colored_masks_web.mp4",
        fps: int = 30,
        background_color: Tuple[int, int, int, int] = (0, 0, 0, 0),
        quality: str = "high",  # "high", "medium", "low"
        use_ffmpeg: bool = True
    ) -> None:
        """
        Save web-optimized video with H.264 encoding for maximum compatibility.
        Falls back to OpenCV if ffmpeg is not available.
        
        Args:
            frames: Original video frames
            results: Prediction results containing masks
            output_path: Output video path
            fps: Frames per second
            background_color: RGBA color for areas without masks
            quality: Video quality preset
            use_ffmpeg: Use ffmpeg for encoding (better compression and web compatibility)
        """
        output_path = Path(output_path)
        
        # For web optimization, prefer MP4 with H.264
        if output_path.suffix.lower() not in ['.mp4', '.webm']:
            output_path = output_path.with_suffix('.mp4')
        
        if use_ffmpeg and self._check_ffmpeg_available():
            self._save_with_ffmpeg(frames, results, output_path, fps, background_color, quality)
        else:
            print("📹 Using OpenCV fallback (ffmpeg not available or disabled)")
            self.save_masks_as_colored_video(frames, results, output_path, fps, background_color, quality)

    def _check_ffmpeg_available(self) -> bool:
        """Check if ffmpeg is available on the system."""
        try:
            import subprocess
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    def _save_with_ffmpeg(
        self,
        frames: List[np.ndarray],
        results: Dict[int, Dict[int, torch.Tensor]],
        output_path: Path,
        fps: int,
        background_color: Tuple[int, int, int, int],
        quality: str
    ) -> None:
        """Save video using ffmpeg for optimal web compression."""
        import subprocess
        import tempfile
        
        height, width = frames[0].shape[:2]
        
        # Collect all unique object IDs and generate colors
        all_obj_ids = set()
        for frame_results in results.values():
            all_obj_ids.update(frame_results.keys())
        colors_dict = self.generate_unique_mask_colors(len(all_obj_ids))
        
        print(f"🎬 Creating web-optimized video with ffmpeg")
        print(f"📊 Format: {output_path.suffix.upper()}, Codec: H.264")
        print(f"📐 Resolution: {width}×{height}, FPS: {fps}, Quality: {quality}")
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            
            # Generate and save frames
            for frame_idx, frame in enumerate(frames):
                if frame_idx in results:
                    # Convert masks to binary format and create colored frame
                    masks_dict = {}
                    for obj_id, mask in results[frame_idx].items():
                        mask_tensor = mask[0]
                        if hasattr(mask_tensor, "cpu"):
                            mask_binary = mask_tensor.cpu().numpy() > 0.5
                        else:
                            mask_binary = mask_tensor > 0.5
                        masks_dict[obj_id] = mask_binary
                    
                    colored_frame = self.handle_overlapping_masks(masks_dict, colors_dict)
                    
                    # Apply background color
                    if background_color != (0, 0, 0, 0):
                        combined_mask = np.zeros((height, width), dtype=bool)
                        for mask in masks_dict.values():
                            combined_mask |= mask
                        colored_frame[~combined_mask] = background_color[:3]
                else:
                    # No mask frame
                    if background_color == (0, 0, 0, 0):
                        colored_frame = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        colored_frame = np.full((height, width, 3), background_color[:3], dtype=np.uint8)
                
                # Save frame as PNG (lossless for ffmpeg input)
                frame_path = temp_dir_path / f"frame_{frame_idx:06d}.png"
                
                # Use PIL to save RGBA
                from PIL import Image
                rgba_pil = Image.fromarray(colored_frame, 'RGB')
                rgba_pil.save(frame_path)
                
                if frame_idx % 50 == 0:
                    print(f"  Generated frame {frame_idx}/{len(frames)}")
            
            # Set quality parameters based on preset
            quality_params = {
                "high": ["-crf", "18", "-preset", "slow"],      # High quality, slower encoding
                "medium": ["-crf", "23", "-preset", "medium"],  # Balanced
                "low": ["-crf", "28", "-preset", "fast"]        # Lower quality, faster encoding
            }
            
            # Build ffmpeg command for web-optimized H.264
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # Overwrite output
                '-framerate', str(fps),
                '-i', str(temp_dir_path / 'frame_%06d.png'),
                '-c:v', 'libx264',  # H.264 codec
                '-pix_fmt', 'yuv420p',  # Web-compatible pixel format
                *quality_params.get(quality, quality_params["medium"]),
                '-movflags', '+faststart',  # Optimize for web streaming
                '-tune', 'stillimage',  # Optimize for generated content
                str(output_path)
            ]
            
            try:
                print(f"🚀 Encoding with ffmpeg...")
                result = subprocess.run(ffmpeg_cmd, 
                                      capture_output=True, 
                                      text=True, 
                                      timeout=300)  # 5 minute timeout
                
                if result.returncode == 0:
                    file_size_mb = output_path.stat().st_size / (1024 * 1024)
                    print(f"✅ Web-optimized video saved to {output_path}")
                    print(f"📁 File size: {file_size_mb:.2f} MB")
                    print(f"🌐 Web compatibility: Excellent (H.264 + faststart)")
                    print(f"📱 Mobile friendly: Yes")
                    print(f"🎯 Streaming optimized: Yes")
                else:
                    print(f"❌ ffmpeg failed: {result.stderr}")
                    print("📹 Falling back to OpenCV...")
                    self.save_masks_as_colored_video(frames, results, output_path, fps, background_color, quality)
                    
            except subprocess.TimeoutExpired:
                print("⏰ ffmpeg encoding timed out, falling back to OpenCV")
                self.save_masks_as_colored_video(frames, results, output_path, fps, background_color, quality)
            except Exception as e:
                print(f"❌ ffmpeg error: {e}")
                print("📹 Falling back to OpenCV...")
                self.save_masks_as_colored_video(frames, results, output_path, fps, background_color, quality)


def main() -> None:
    """
    Example usage of the SAM2VideoProcessor
    """
    # Initialize the processor
    processor = SAM2VideoProcessor("facebook/sam2-hiera-large")

    # Example usage with your own video
    video_path = "path/to/your/video.mp4"  # Replace with your video path

    print("To use the SAM2VideoProcessor:")
    print("1. For point prompts:")
    print(f"   frames, masks, temp_dir = processor.predict_video_with_points(")
    print(f"       video_path='{video_path}',")
    print(f"       frame_idx=50,  # Middle frame")
    print(f"       points=[[960, 540]],  # Center point")
    print(f"       max_frames=100")
    print(f"   )")
    print(f"   processor.visualize_results(frames, masks, 50, [[960, 540]])")
    print(f"   processor.cleanup_temp_dir(temp_dir)")

    print("\n2. For box prompts:")
    print(f"   frames, masks, temp_dir = processor.predict_video_with_box(")
    print(f"       video_path='{video_path}',")
    print(f"       frame_idx=50,")
    print(f"       box=[100, 100, 500, 400],  # [x1, y1, x2, y2]")
    print(f"       max_frames=100")
    print(f"   )")


if __name__ == "__main__":
    main()
