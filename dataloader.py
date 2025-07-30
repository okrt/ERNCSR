import os, random, math
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import io
import pillow_avif
class FastImageEnhancerDataset(Dataset):
    """
    Fast dataloader optimized for Windows compatibility and performance
    """
    def __init__(self, root_dir, patch_size, upscale_factor, random_zoom=True, seed=22, rgb=True, blur=True, repeat=100, 
                 permanent_cache_size=200, compression_prob=0.8, noise_prob=0.0, compression_ranges=None, noise_range=(0, 10),
                 smart_caching=True):
        if seed is not None:
            self.fix_seed(seed)
        
        self.root_dir = root_dir
        self.pre_processed = True
        self.hq_dir = os.path.join(root_dir, 'hq')
        if os.path.exists(os.path.join(root_dir, 'lq')):
            self.lq_dir = os.path.join(root_dir, 'lq')
        else:
            self.lq_dir = os.path.join(root_dir, 'hq')
            self.pre_processed = False
            
        # Get file lists
        self.hq_files = sorted([f for f in os.listdir(self.hq_dir) 
                                if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        self.lq_files = sorted([f for f in os.listdir(self.lq_dir)
                                if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
        
        self.patch_size = patch_size
        self.upscale_factor = upscale_factor
        self.random_zoom = random_zoom
        self.rgb = rgb
        self.blur = blur
        self.repeat = repeat
        self.compression_prob = compression_prob
        self.noise_prob = noise_prob
        self.noise_range = noise_range
        
        # Set default compression ranges if not provided
        if compression_ranges is None:
            self.compression_ranges = {
                'jpeg': (40, 80),  # JPEG quality range
                'webp': (40, 80),  # WebP quality range
                'avif': (40, 80)   # AVIF quality range
            }
        else:
            self.compression_ranges = compression_ranges
        
        # Pre-compute blur kernel
        if self.blur:
            self.blur_kernel = cv2.getGaussianKernel(5, 1)
            self.blur_kernel = self.blur_kernel @ self.blur_kernel.T
        
        # Cache configuration
        self.permanent_cache_size = min(permanent_cache_size, len(self.hq_files))
        self.image_cache = {}
        
        # Worker-specific cache settings
        self.worker_id = 0
        self.num_workers = 1
        self.worker_cache_size = self.permanent_cache_size
        self.smart_caching = smart_caching
        self.worker_cache_indices = None

    def fix_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_image(self, path):
        """Load image with caching"""
        if path in self.image_cache:
            # Return a copy to avoid modifying cached data
            return self.image_cache[path].copy()
        
        # Load with OpenCV (faster than PIL)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            # Fallback to PIL if cv2 fails
            img = np.array(Image.open(path).convert('RGB'))[:, :, ::-1]  # RGB to BGR
        
        # Smart caching: only cache if this image is assigned to this worker
        if self.smart_caching and self.worker_cache_indices is not None:
            # Extract filename from path
            filename = os.path.basename(path)
            if filename in [self.hq_files[i] for i in self.worker_cache_indices]:
                self.image_cache[path] = img.copy()
        elif len(self.image_cache) < self.worker_cache_size:
            # Simple caching: cache until limit reached
            self.image_cache[path] = img.copy()
        
        return img

    def apply_jpeg_compression(self, img_bgr, quality):
        """Apply JPEG compression to image"""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        img_bgr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return img_bgr
    
    def apply_webp_compression(self, img_bgr, quality):
        """Apply WebP compression to image"""
        encode_param = [int(cv2.IMWRITE_WEBP_QUALITY), quality]
        _, encimg = cv2.imencode('.webp', img_bgr, encode_param)
        img_bgr = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
        return img_bgr
    
    def apply_avif_compression(self, img_bgr, quality):
        """Apply AVIF compression to image using PIL"""
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Save to buffer with AVIF compression
        buffer = io.BytesIO()
        try:
            img_pil.save(buffer, format='AVIF', quality=quality)
            buffer.seek(0)
            # Read back and convert to BGR
            img_pil = Image.open(buffer)
            img_rgb = np.array(img_pil)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            # If AVIF is not supported, fallback to WebP
            img_bgr = self.apply_webp_compression(img_bgr, quality)
        
        return img_bgr
    
    def apply_gaussian_noise(self, img_bgr, sigma):
        """Apply Gaussian noise to image"""
        noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
        noisy_img = img_bgr.astype(np.float32) + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return noisy_img

    def process_image_pair(self, hq_bgr, lq_bgr):
        """Fast augmentation pipeline with synchronized crops"""
        if self.patch_size is None:
            return hq_bgr, lq_bgr
        
        # Get image dimensions
        h_lq, w_lq = lq_bgr.shape[:2]
        h_hq, w_hq = hq_bgr.shape[:2]
        
        # Verify that images are properly scaled
        # Allow small differences due to integer division
        expected_h_lq = h_hq // self.upscale_factor
        expected_w_lq = w_hq // self.upscale_factor
        
        if abs(h_lq - expected_h_lq) > 1 or abs(w_lq - expected_w_lq) > 1:
            print(f"Warning: Size mismatch - LQ: {w_lq}x{h_lq}, HQ: {w_hq}x{h_hq}, scale: {self.upscale_factor}")
            # Resize LQ to ensure alignment
            lq_bgr = cv2.resize(lq_bgr, (expected_w_lq, expected_h_lq), interpolation=cv2.INTER_CUBIC)
            h_lq, w_lq = lq_bgr.shape[:2]
        
        # Calculate maximum valid crop size
        max_crop_size = min(h_lq, w_lq)
        
        # Random crop size (with zoom)
        selected_crop_size = self.patch_size
        if self.random_zoom and max_crop_size > self.patch_size and random.random() < 0.2:
            max_zoom = max_crop_size / self.patch_size
            zoom = 2 ** random.uniform(0, math.log2(max_zoom))
            selected_crop_size = int(self.patch_size * zoom)
            selected_crop_size = min(selected_crop_size, max_crop_size)
        
        # Calculate valid crop range
        max_top = h_lq - selected_crop_size
        max_left = w_lq - selected_crop_size
        
        # Random crop coordinates
        if max_top > 0 and max_left > 0:
            top = random.randint(0, max_top)
            left = random.randint(0, max_left)
        else:
            top = left = 0
        
        # Crop LQ image
        lq_crop = lq_bgr[top:top+selected_crop_size, left:left+selected_crop_size]
        
        # Crop HQ image with scaled coordinates
        top_hq = top * self.upscale_factor
        left_hq = left * self.upscale_factor
        size_hq = selected_crop_size * self.upscale_factor
        hq_crop = hq_bgr[top_hq:top_hq+size_hq, left_hq:left_hq+size_hq]
        
        # Verify the crops are properly aligned
        if hq_crop.shape[0] != lq_crop.shape[0] * self.upscale_factor or \
           hq_crop.shape[1] != lq_crop.shape[1] * self.upscale_factor:
            print(f"Crop size mismatch - LQ crop: {lq_crop.shape}, HQ crop: {hq_crop.shape}")
            # Force resize to ensure alignment
            expected_hq_h = lq_crop.shape[0] * self.upscale_factor
            expected_hq_w = lq_crop.shape[1] * self.upscale_factor
            hq_crop = cv2.resize(hq_crop, (expected_hq_w, expected_hq_h), interpolation=cv2.INTER_CUBIC)
        
        # Resize if zoomed
        if selected_crop_size != self.patch_size:
            lq_crop = cv2.resize(lq_crop, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
            hq_crop = cv2.resize(hq_crop, (self.patch_size * self.upscale_factor, 
                                          self.patch_size * self.upscale_factor), interpolation=cv2.INTER_CUBIC)
        
        # Apply same augmentations to both images
        # Generate all random decisions first
        do_hflip = random.random() > 0.5
        do_vflip = random.random() > 0.5
        do_rotate = random.random() > 0.5
        rotate_angle = random.choice([cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]) if do_rotate else None
        do_blur = self.blur and random.random() <= 0.2
        
        # Apply augmentations
        if do_hflip:
            hq_crop = cv2.flip(hq_crop, 1)
            lq_crop = cv2.flip(lq_crop, 1)
        
        if do_vflip:
            hq_crop = cv2.flip(hq_crop, 0)
            lq_crop = cv2.flip(lq_crop, 0)
        
        if do_rotate:
            hq_crop = cv2.rotate(hq_crop, rotate_angle)
            lq_crop = cv2.rotate(lq_crop, rotate_angle)
        
        # Blur augmentation (only on LQ)
        if do_blur:
            lq_crop = cv2.filter2D(lq_crop, -1, self.blur_kernel)
        # Gaussian noise augmentation (only on LQ)
        if random.random() < self.noise_prob:
            sigma = random.uniform(self.noise_range[0], self.noise_range[1])
            lq_crop = self.apply_gaussian_noise(lq_crop, sigma)
        
        # Compression augmentation (only on LQ)
        if random.random() < self.compression_prob:
            compression_type = random.choice(['jpeg', 'webp', 'avif'])
            quality_range = self.compression_ranges[compression_type]
            quality = random.randint(quality_range[0], quality_range[1])
            
            if compression_type == 'jpeg':
                lq_crop = self.apply_jpeg_compression(lq_crop, quality)
            elif compression_type == 'webp':
                lq_crop = self.apply_webp_compression(lq_crop, quality)
            elif compression_type == 'avif':
                lq_crop = self.apply_avif_compression(lq_crop, quality)
        
        
        
        return hq_crop, lq_crop

    def __getitem__(self, idx):
        t_idx = idx % len(self.hq_files)
        hq_path = os.path.join(self.hq_dir, self.hq_files[t_idx])
        lq_path = os.path.join(self.lq_dir, self.lq_files[t_idx])

        # Load HQ image first
        hq_bgr = self.get_image(hq_path)
        
        if self.pre_processed:
            # Load pre-processed LQ image
            lq_bgr = self.get_image(lq_path)
            
            # IMPORTANT: Ensure LQ and HQ are properly aligned
            # Pre-processed LQ should be exactly 1/scale of HQ size
            expected_lq_h = hq_bgr.shape[0] // self.upscale_factor
            expected_lq_w = hq_bgr.shape[1] // self.upscale_factor
            
            if lq_bgr.shape[0] != expected_lq_h or lq_bgr.shape[1] != expected_lq_w:
                # Resize LQ to match expected dimensions
                lq_bgr = cv2.resize(lq_bgr, (expected_lq_w, expected_lq_h), interpolation=cv2.INTER_CUBIC)
        else:
            # Create LQ from HQ to ensure perfect alignment
            if self.upscale_factor == 1:
                lq_bgr = hq_bgr.copy()
            else:
                new_h = hq_bgr.shape[0] // self.upscale_factor
                new_w = hq_bgr.shape[1] // self.upscale_factor
                lq_bgr = cv2.resize(hq_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        
        # Process and augment
        hq_bgr, lq_bgr = self.process_image_pair(hq_bgr, lq_bgr)
        
        # Convert to tensors
        if self.rgb:
            # BGR to RGB and normalize
            hq_rgb = cv2.cvtColor(hq_bgr, cv2.COLOR_BGR2RGB)
            lq_rgb = cv2.cvtColor(lq_bgr, cv2.COLOR_BGR2RGB)
            
            tensor_hq = torch.from_numpy(hq_rgb.transpose(2, 0, 1)).float() / 255.0
            tensor_lq = torch.from_numpy(lq_rgb.transpose(2, 0, 1)).float() / 255.0
        else:
            # Extract Y channel
            hq_ycrcb = cv2.cvtColor(hq_bgr, cv2.COLOR_BGR2YCrCb)
            lq_ycrcb = cv2.cvtColor(lq_bgr, cv2.COLOR_BGR2YCrCb)
            
            hq_y = hq_ycrcb[:, :, 0]
            lq_y = lq_ycrcb[:, :, 0]
            
            tensor_hq = torch.from_numpy(hq_y).unsqueeze(0).float() / 255.0
            tensor_lq = torch.from_numpy(lq_y).unsqueeze(0).float() / 255.0
        
        return tensor_lq, tensor_hq

    def __len__(self):
        return len(self.hq_files) * self.repeat


def worker_init_fn(worker_id):
    """Worker initialization with per-worker caching"""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is None:  # Single process data loading
        return
    
    dataset = worker_info.dataset
    dataset.worker_id = worker_info.id
    dataset.num_workers = worker_info.num_workers
    
    # Calculate subset of images to cache per worker
    # Each worker caches a different subset to maximize memory efficiency
    total_cache_size = dataset.permanent_cache_size
    per_worker_cache = max(1, total_cache_size // worker_info.num_workers)
    
    # Calculate start and end indices for this worker's cache
    start_idx = worker_info.id * per_worker_cache
    end_idx = min(start_idx + per_worker_cache, len(dataset.hq_files))
    
    # Set this worker's cache size
    dataset.worker_cache_size = end_idx - start_idx
    
    # Store indices for smart caching
    if dataset.smart_caching:
        dataset.worker_cache_indices = list(range(start_idx, end_idx))
    
    # Clear any existing cache
    dataset.image_cache.clear()
    
    # Set numpy random seed per worker
    # Use a smaller seed value that fits in 32-bit integer
    seed = int(worker_info.seed % 2**31)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


def create_fast_dataloader(hr_dir, batch_size, shuffle=False, scale=4, num_workers=2, rgb=True, 
                          compression_prob=0.3, noise_prob=0.3, compression_ranges=None, noise_range=(0, 30),
                          permanent_cache_size=400, smart_caching=True):
    """Create fast dataloader"""
    dataset = FastImageEnhancerDataset(
        root_dir=hr_dir, 
        upscale_factor=scale, 
        repeat=100, 
        patch_size=64,
        seed=None,
        rgb=rgb,
        permanent_cache_size=permanent_cache_size,
        compression_prob=compression_prob,
        noise_prob=noise_prob,
        compression_ranges=compression_ranges,
        noise_range=noise_range,
        smart_caching=smart_caching
    )
    
    # Adjust settings for Windows
    if os.name == 'nt':  # Windows
        num_workers = min(num_workers, 4)  # Limit workers on Windows
        prefetch_factor = 2
    else:
        prefetch_factor = 4
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        worker_init_fn=worker_init_fn
    )
    
    return dataloader

def benchmark_w_cache():
     # Benchmark
    num_workers=8
    print(f"\nBenchmarking with batch_size={batch_size}, num_workers={num_workers}")
    dataset = FastImageEnhancerDataset(
        root_dir="ds_test", 
        upscale_factor=4, 
        repeat=50, 
        patch_size=64,
        seed=22,
        rgb=rgb,
        permanent_cache_size=1200,
        compression_prob=0,
        noise_prob=0,
        smart_caching=True
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=num_workers, 
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=4,
        worker_init_fn=worker_init_fn
    )
    print(len(dataloader))
    start_time = time.time()
    for i, (lr_images, hr_images) in enumerate(dataloader):
 
        if (i%50==0):
            print(i,"/",len(dataloader))
    
    elapsed = time.time() - start_time
    print(f"\nCACHE ENABLED Loaded {len(dataloader)} batches in {elapsed:.2f} seconds")
    print(f"Average time per batch: {elapsed/len(dataloader):.3f} seconds")
    print(f"Images per second: {3250/elapsed:.1f}")

def benchmark_wo_cache():
     # Benchmark
    num_workers=8
    print(f"\nBenchmarking without CACHE with batch_size={batch_size}, num_workers={num_workers}")
    dataset = FastImageEnhancerDataset(
        root_dir="ds_test", 
        upscale_factor=4, 
        repeat=50, 
        patch_size=64,
        seed=22,
        rgb=rgb,
        permanent_cache_size=1,
        compression_prob=0,
        noise_prob=0,
        smart_caching=False
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=num_workers, 
        prefetch_factor=2,
        pin_memory=True,
        persistent_workers=4,
        worker_init_fn=worker_init_fn
    )
    print(len(dataloader))
    start_time = time.time()
    for i, (lr_images, hr_images) in enumerate(dataloader):
        if (i%50==0):
            print(i,"/",len(dataloader))
    
    elapsed = time.time() - start_time
    print(f"\nCACHE DISABLED Loaded {len(dataloader)} batches in {elapsed:.2f} seconds")
    print(f"Average time per batch: {elapsed/len(dataloader):.3f} seconds")
    print(f"Images per second: {3250/elapsed:.1f}")

if __name__ == "__main__":
    """Test the fast dataloader"""
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    import time
    import multiprocessing
    
    multiprocessing.freeze_support()
    
    rgb = True
    batch_size = 32
    num_workers = 4 if os.name != 'nt' else 2
    
    print("Creating fast dataloader...")
    print(f"Platform: {'Windows' if os.name == 'nt' else 'Unix-based'}")
    print(f"Workers: {num_workers}")
    
    dataloader = create_fast_dataloader("dataset", batch_size=batch_size, rgb=rgb, num_workers=num_workers)
    
    
    
    for i, (lr_images, hr_images) in enumerate(dataloader):
        if i >= 10:
            break
        if i == 0:
            # Display first batch - show individual pairs to verify alignment
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.suptitle('LR-HR Pairs (Top: LR, Bottom: HR)', fontsize=16)
            
            # Show first 4 image pairs
            for idx in range(min(4, lr_images.shape[0])):
                # LR image
                lr_img = lr_images[idx].permute(1, 2, 0).numpy()
                axes[0, idx].imshow(lr_img, cmap=None if rgb else 'gray')
                axes[0, idx].set_title(f'LR {idx+1}')
                axes[0, idx].axis('off')
                
                # HR image
                hr_img = hr_images[idx].permute(1, 2, 0).numpy()
                axes[1, idx].imshow(hr_img, cmap=None if rgb else 'gray')
                axes[1, idx].set_title(f'HR {idx+1}')
                axes[1, idx].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            print(f"LR shape: {lr_images[0].shape}, HR shape: {hr_images[0].shape}")
            print(f"Scale factor verified: {hr_images[0].shape[-1] // lr_images[0].shape[-1]}")
    print("Do you wanna run bench? (Y/(N))")
    a=input()
    if(a=="Y"):
        benchmark_w_cache()
        benchmark_wo_cache()
   