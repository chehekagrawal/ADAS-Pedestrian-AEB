import argparse
import time
import cv2
from ultralytics import YOLO
from pathlib import Path


def run_inference(source, model_path, save_dir, show=False, classes=None):
    """
    Run YOLO inference on image / video / webcam.

    Args:
        source: image path | video path | webcam
        model_path: path to trained model
        save_dir: folder to save outputs
        show: display live window
        classes: list of class indices to filter
    """

    # ---- Safety checks ----
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # ---- Webcam ----
    if source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open webcam")

        output_path = save_dir / "webcam_output_multiclass.mp4"

    # ---- Video file ----
    elif Path(source).suffix.lower() in [".mp4", ".avi", ".mov"]:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found: {source}")

        output_path = save_dir / "video_output_multiclass.mp4"

    # ---- Image ----
    else:
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Image not found: {source}")

        start = time.time()
        results = model(img, classes=classes)
        end = time.time()

        annotated = results[0].plot()
        output_path = save_dir / f"{Path(source).stem}_multiclass.jpg"
        cv2.imwrite(str(output_path), annotated)

        print(f"Inference time: {end - start:.3f}s")
        print(f"Saved → {output_path}")
        return

    # ---- Video/Webcam processing ----
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20

    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    print("Running real-time detection... Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.time()
        results = model(frame, classes=classes)
        end = time.time()

        annotated = results[0].plot()
        fps_text = f"FPS: {1 / max(end - start, 1e-6):.1f}"

        cv2.putText(
            annotated,
            fps_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        out.write(annotated)

        if show:
            cv2.imshow("Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLO Multiclass Inference")
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Image path | video path | webcam",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolo_multiclass_best.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="results/inference",
        help="Output folder",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display live window",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="Filter by class index (e.g. 0 1 2 3)",
    )

    args = parser.parse_args()
    run_inference(args.source, args.model, args.save, args.show, args.classes)


if __name__ == "__main__":
    main()
