"""
Grad-CAM Explainability Module (XAI).

Generates visual attention heatmaps showing WHAT the YOLO model focused on
when making a detection. This provides interpretable evidence for AEB decisions:
"The system braked because it saw THIS."

Grad-CAM (Gradient-weighted Class Activation Mapping):
    1. Forward pass through the model
    2. Compute gradients of the target class w.r.t. feature maps
    3. Global-average-pool the gradients to get channel importance weights
    4. Weighted combination of feature maps → attention heatmap
    5. ReLU to keep only positive contributions
    6. Upsample and overlay on original frame

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization" (ICCV 2017)
"""

import cv2
import numpy as np
import os

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class GradCAM:
    """
    Generate visual explanations of YOLO detections using Grad-CAM.

    Hooks into the last convolutional layer of the YOLO backbone to
    capture activations and gradients for attention visualization.
    """

    def __init__(self, model_path="models/best.pt", device=None):
        """
        Load YOLO model and register hooks on the target layer.

        Args:
            model_path: Path to trained YOLO .pt model
            device: "cpu" or "cuda" (auto-detect if None)
        """
        self.activations = None
        self.gradients = None
        self.model = None
        self.hooks = []

        if not YOLO_AVAILABLE or not TORCH_AVAILABLE:
            print("[GradCAM] PyTorch or Ultralytics not available.")
            return

        if not os.path.exists(model_path):
            print(f"[GradCAM] Model not found at {model_path}")
            return

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        try:
            self.model = YOLO(model_path)
            # Access the underlying PyTorch model
            self._torch_model = self.model.model
            self._torch_model.to(self.device)
            self._torch_model.eval()

            # Register hooks on the last backbone conv layer
            self._register_hooks()
            print(f"[GradCAM] Model loaded, hooks registered on {self.device}")
        except Exception as e:
            print(f"[GradCAM] Failed to initialize: {e}")
            self.model = None

    def _register_hooks(self):
        """
        Register forward and backward hooks on the target layer.

        Target: the last convolutional layer in the YOLO backbone,
        which contains the most semantically rich feature maps.
        """
        if self._torch_model is None:
            return

        # Find the last Conv2d layer in the backbone
        target_layer = None
        for module in self._torch_model.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module

        if target_layer is None:
            print("[GradCAM] Warning: Could not find Conv2d layer for hooks.")
            return

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, frame, target_bbox=None):
        """
        Generate Grad-CAM attention heatmap for detections in a frame.

        Steps:
        1. Forward pass through model (with gradient tracking)
        2. Compute a target score (sum of detection confidences)
        3. Backward pass to get gradients at the hooked layer
        4. Weight activations by global-average-pooled gradients
        5. Apply ReLU (only keep positive contributions)
        6. Resize heatmap to frame dimensions
        7. Normalize to 0–255

        Args:
            frame: BGR frame (numpy H×W×3)
            target_bbox: Optional [x1, y1, x2, y2] to focus on specific detection

        Returns:
            heatmap: uint8 array (H×W), ready for colormap application
            Returns None if model is not available
        """
        if self.model is None or not TORCH_AVAILABLE:
            return None

        h, w = frame.shape[:2]

        # Prepare input tensor
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        img_tensor.requires_grad_(True)

        # Enable gradient computation
        self._torch_model.train()
        self.activations = None
        self.gradients = None

        try:
            # Forward pass
            output = self._torch_model(img_tensor)

            # Create a scalar target for backprop
            # Use the sum of all detection scores as the target
            if isinstance(output, (list, tuple)):
                # YOLO outputs can be complex — concatenate all predictions
                target = sum(o.sum() for o in output if isinstance(o, torch.Tensor))
            else:
                target = output.sum()

            # Backward pass
            self._torch_model.zero_grad()
            target.backward(retain_graph=False)

        except Exception as e:
            print(f"[GradCAM] Error during forward/backward pass: {e}")
            self._torch_model.eval()
            return None

        self._torch_model.eval()

        if self.activations is None or self.gradients is None:
            print("[GradCAM] No activations/gradients captured.")
            return None

        # Grad-CAM computation
        # Global average pool the gradients → channel importance weights
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weighted combination of activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)

        # ReLU — only positive contributions matter
        cam = F.relu(cam)

        # Resize to frame dimensions
        cam = F.interpolate(cam, size=(h, w), mode='bilinear', align_corners=False)

        # Normalize to 0–255
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = (cam * 255).astype(np.uint8)

        return cam

    def overlay_heatmap(self, frame, heatmap, alpha=0.4):
        """
        Blend attention heatmap over original frame.

        Red = high attention (model focused here)
        Blue = low attention (model ignored this area)

        Args:
            frame: BGR frame (original)
            heatmap: uint8 array from generate_heatmap()
            alpha: Blend ratio (0.0–1.0, higher = more heatmap)

        Returns:
            blended: BGR frame with heatmap overlay
        """
        if heatmap is None:
            return frame

        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Ensure same size
        if heatmap_colored.shape[:2] != frame.shape[:2]:
            heatmap_colored = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))

        return cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)

    def process_frame(self, frame, alpha=0.4):
        """
        Convenience: generate heatmap and overlay in one call.

        Args:
            frame: BGR frame
            alpha: Heatmap transparency

        Returns:
            tuple: (overlay_frame, raw_heatmap)
        """
        heatmap = self.generate_heatmap(frame)
        if heatmap is None:
            return frame, None
        overlay = self.overlay_heatmap(frame, heatmap, alpha)
        return overlay, heatmap

    def process_video(self, video_path, output_path, alpha=0.4):
        """
        Generate Grad-CAM overlay for an entire video.

        For each frame: detect → generate heatmap → overlay → write.

        Args:
            video_path: Input video file path
            output_path: Output video file path
            alpha: Heatmap transparency
        """
        if self.model is None:
            print("[GradCAM] No model loaded, cannot process video.")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[GradCAM] Cannot open video: {video_path}")
            return

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            overlay, _ = self.process_frame(frame, alpha)
            out.write(overlay)

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"[GradCAM] Processed {frame_count} frames...")

        cap.release()
        out.release()
        print(f"[GradCAM] Output saved to {output_path} ({frame_count} frames)")

    def save_heatmap(self, frame, save_path, alpha=0.4):
        """
        Save a single frame's Grad-CAM overlay to disk.

        Args:
            frame: BGR frame
            save_path: Output image path
            alpha: Heatmap transparency
        """
        overlay, heatmap = self.process_frame(frame, alpha)
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        cv2.imwrite(save_path, overlay)
        return overlay, heatmap

    def cleanup(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.cleanup()
