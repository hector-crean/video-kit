import torch
import traceback

print('=== PyTorch Environment ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')

print('\n=== Testing SAM2 Import ===')
try:
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    print('✓ SAM2 import successful')
except Exception as e:
    print(f'✗ SAM2 import failed: {e}')
    traceback.print_exc()
    exit(1)

print('\n=== Testing SAM2 Model Loading with Explicit Device ===')
try:
    # Force CPU device for Mac compatibility
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    predictor = SAM2VideoPredictor.from_pretrained('facebook/sam2-hiera-large', device=device)
    print('✓ SAM2 predictor created successfully')
except Exception as e:
    print(f'✗ SAM2 predictor creation failed: {e}')
    traceback.print_exc()
    
    # Try with CPU as fallback
    print('\n=== Trying with CPU device ===')
    try:
        predictor = SAM2VideoPredictor.from_pretrained('facebook/sam2-hiera-large', device='cpu')
        print('✓ SAM2 predictor created successfully with CPU')
    except Exception as e2:
        print(f'✗ CPU fallback also failed: {e2}')
        traceback.print_exc()
        exit(1)

print('\n=== Testing Device Movement ===')
try:
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Target device: {device}')
    predictor = predictor.to(device)
    print(f'✓ SAM2 moved to {device} successfully')
except Exception as e:
    print(f'✗ Device movement failed: {e}')
    traceback.print_exc()
    exit(1)

print('\n=== All Tests Passed! ===')
print('SAM2 is working correctly on your Mac!') 