"""
Weather & Visibility Classification Module.

Classifies dashcam frames into weather categories: Clear, Rain, Fog, Night.
Supports two modes:
    1. ML-based: Pretrained ResNet-18 fine-tuned on weather dataset (if model available)
    2. Heuristic-based: Image statistics analysis (brightness, contrast, blur) as fallback

The classification result feeds into weather_to_physics.py to automatically
adjust road surface friction, reaction time penalties, and visibility range.

Classes:
    Clear — normal conditions (μ=0.85, full visibility)
    Rain  — wet road, reduced visibility (μ=0.55, 60m range)
    Fog   — severely reduced visibility (μ=0.55, 30m range)
    Night — low light, slightly reduced contrast (μ=0.85, 50m range)
"""

import cv2
import numpy as np

try:
    import torch
    import torchvision.transforms as transforms
    from torchvision import models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class WeatherClassifier:
    """
    Classify frame weather conditions.

    Dual-mode classifier:
    - If a trained model file is provided → uses ResNet-18 inference
    - Otherwise → falls back to heuristic classification (always available)

    Usage:
        classifier = WeatherClassifier()
        result = classifier.classify(frame)
        # result = {"weather": "Rain", "confidence": 0.85, "method": "heuristic"}
    """

    CLASSES = ["Clear", "Rain", "Fog", "Night"]

    def __init__(self, model_path=None, device="cpu"):
        """
        Initialize weather classifier.

        Args:
            model_path: Path to a trained ResNet-18 .pth file. If None or file
                        doesn't exist, uses heuristic classification.
            device: PyTorch device string ("cpu" or "cuda")
        """
        self.model = None
        self.device = device
        self.transform = None

        if model_path and TORCH_AVAILABLE:
            try:
                self._load_model(model_path)
            except (FileNotFoundError, RuntimeError) as e:
                print(f"[WeatherClassifier] Could not load model: {e}")
                print("[WeatherClassifier] Falling back to heuristic mode.")

    def _load_model(self, model_path):
        """Load pretrained ResNet-18 for weather classification."""
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = models.resnet18(weights=None)
        # Replace final FC layer for 4-class classification
        num_features = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_features, len(self.CLASSES))

        # Load trained weights
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def classify(self, frame):
        """
        Classify weather condition in a frame.

        Automatically chooses ML or heuristic mode based on model availability.

        Args:
            frame: BGR frame (numpy H×W×3)

        Returns:
            dict with:
                "weather": str — one of CLASSES
                "confidence": float — prediction confidence (0.0–1.0)
                "method": str — "ml" or "heuristic"
        """
        if self.model is not None and TORCH_AVAILABLE:
            return self._ml_classify(frame)
        else:
            result = self.heuristic_classify(frame)
            result["method"] = "heuristic"
            return result

    def _ml_classify(self, frame):
        """Run ResNet-18 inference on a frame."""
        # Convert BGR to RGB for torchvision
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        return {
            "weather": self.CLASSES[predicted.item()],
            "confidence": float(confidence.item()),
            "method": "ml",
        }

    @staticmethod
    def heuristic_classify(frame):
        """
        Fallback: Image statistics-based weather classification.

        Analysis pipeline:
        1. Mean brightness → Night detection (< 60)
        2. Laplacian variance + contrast → Fog detection (blurry + low contrast)
        3. Edge density + saturation → Rain detection
        4. Otherwise → Clear

        This provides reasonable classification without a trained model,
        suitable for development and testing.

        Args:
            frame: BGR frame (numpy H×W×3)

        Returns:
            dict with "weather" (str) and "confidence" (float)
        """
        if frame is None or frame.size == 0:
            return {"weather": "Clear", "confidence": 0.0}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # ── Metric 1: Mean brightness ──
        mean_brightness = float(np.mean(gray))

        # ── Metric 2: Contrast (standard deviation of intensity) ──
        contrast = float(np.std(gray))

        # ── Metric 3: Blur level (Laplacian variance — lower = blurrier) ──
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # ── Metric 4: Saturation analysis ──
        mean_saturation = float(np.mean(hsv[:, :, 1]))

        # ── Decision tree ──

        # Night: very dark frames
        if mean_brightness < 60:
            confidence = min(1.0, (60 - mean_brightness) / 40)
            return {"weather": "Night", "confidence": max(0.5, confidence)}

        # Fog: low contrast + blurry + washed-out colors
        if contrast < 35 and laplacian_var < 500 and mean_saturation < 50:
            confidence = min(1.0, (500 - laplacian_var) / 400)
            return {"weather": "Fog", "confidence": max(0.5, confidence)}

        # Rain: moderate brightness drop + some blur + higher saturation (reflections)
        if mean_brightness < 120 and laplacian_var < 1000 and mean_saturation > 30:
            # Additional check: look for vertical streak patterns (rain drops)
            # Using Sobel in vertical direction
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            vertical_energy = float(np.mean(np.abs(sobel_y)))

            if vertical_energy > 15:
                confidence = min(1.0, vertical_energy / 30)
                return {"weather": "Rain", "confidence": max(0.4, confidence)}

        # Default: Clear
        confidence = min(1.0, mean_brightness / 200)
        return {"weather": "Clear", "confidence": max(0.5, confidence)}
