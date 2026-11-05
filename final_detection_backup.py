#!/usr/bin/env python3
"""
Final implementation: Real-time plastic detection with stable grid highlighting
This version incorporates all lessons learned from testing:
- Stable grids (no flickering/jumping)
- Reduced false positives
- Good performance
- Clear visual feedback
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import deque

# ========== CONFIGURATION ==========
# Model and classes
MODEL_PATH = "runs/detect/train-8/weights/best.onnx"
CLASS_NAMES = ['Plastic bottle', 'Plastic bag', 'Plastic wrapper', 'Plastic container', 'Other plastic']

# Detection settings - balanced for accuracy and false positive reduction
CONFIDENCE_THRESHOLD = 0.50  # Increased from default to reduce false positives
NMS_THRESHOLD = 0.45         # Non-maximum suppression threshold

# Stability settings - prevents grid flickering
STABLE_FRAMES_REQUIRED = 2   # Need this many consecutive detections to show grid
MAX_HISTORY_FRAMES = 5       # How many frames to keep in history

# Grid appearance - optimized for visibility
GRID_COLOR = (0, 255, 255)   # Bright cyan - excellent visibility
GRID_THICKNESS = 2
GRID_SPACING = 20            # Pixels between grid lines
GRID_ALPHA = 0.25            # Grid transparency (0.0-1.0)

# Performance settings
PROCESS_WIDTH = 420          # Process at this width (balance speed/detail)
PROCESS_HEIGHT = 320
TARGET_FPS = 30              # Desired webcam FPS

# False positive filtering
MIN_OBJECT_AREA_RATIO = 0.001   # Minimum object size (0.1% of frame)
MAX_OBJECT_AREA_RATIO = 0.6     # Maximum object size (60% of frame)
MIN_ASPECT_RATIO = 0.15         # Reject very tall/narrow objects (humans/poles)
MAX_ASPECT_RATIO = 5.0          # Reject very wide/tall objects
# ===================================

class DetectionStabilizer:
    """Ensures detections are stable before showing grid"""
    def __init__(self, required_frames=STABLE_FRAMES_REQUIRED, max_history=MAX_HISTORY_FRAMES):
        self.required_frames = required_frames
        self.max_history = max_history
        self.detection_history = deque(maxlen=max_history)  # Stores bool: detection in frame
        self.box_history = deque(maxlen=max_history)        # Stores boxes when detected
        self.stable_boxes = np.empty((0, 4))                # [x1, y1, x2, y2]
        self.stable_classes = np.empty((0,))                # class IDs

    def update(self, has_detection, boxes=None, classes=None):
        """Update history and compute stable detections"""
        self.detection_history.append(has_detection)
        if boxes is not None and len(boxes) > 0:
            self.box_history.append((boxes.copy(), classes.copy() if classes is not None else None))
        else:
            self.box_history.append((np.empty((0, 4)), np.empty((0,))))

        # Check if we have enough consecutive detections
        if len(self.detection_history) < self.required_frames:
            self.stable = False
            self.stable_boxes = np.empty((0, 4))
            self.stable_classes = np.empty((0,))
            return False, self.stable_boxes, self.stable_classes

        # Stable if last 'required_frames' entries are all True
        recent_detections = list(self.detection_history)[-self.required_frames:]
        self.stable = all(recent_detections)

        if self.stable and len(self.box_history) > 0:
            # Use the most recent detection set
            self.stable_boxes, self.stable_classes = self.box_history[-1]
        else:
            self.stable_boxes = np.empty((0, 4))
            self.stable_classes = np.empty((0,))

        return self.stable, self.stable_boxes, self.stable_classes

    def is_stable(self):
        return getattr(self, 'stable', False)

def is_valid_detection(box, frame_shape):
    """
    Validate if a detection is likely to be a real plastic object
    Returns True if valid, False if likely false positive
    """
    x1, y1, x2, y2 = box
    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    # Must have positive dimensions
    if width <= 0 or height <= 0:
        return False

    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_w * frame_h
    if frame_area == 0:
        return False

    # Area checks
    area = width * height
    area_ratio = area / frame_area

    if area_ratio < MIN_OBJECT_AREA_RATIO or area_ratio > MAX_OBJECT_AREA_RATIO:
        return False  # Too small or too large

    # Aspect ratio checks - helps distinguish humans from plastic
    if height > 0:
        aspect_ratio = width / height
        if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
            return False  # Too tall/narrow or too wide/tall

    return True

def draw_precise_grid(frame, boxes, class_ids=None):
    """
    Draw precise grid highlighting ONLY on validated stable detections
    """
    if boxes is None or len(boxes) == 0:
        return frame

    # Create overlay for transparency
    overlay = frame.copy()
    h, w = frame.shape[:2]
    valid_indices = []

    # Validate each detection
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)

        # Clamp to frame boundaries
        x1 = max(0, min(w-1, x1))
        x2 = max(x1+1, min(w-1, x2))  # Ensure at least 1px width
        y1 = max(0, min(h-1, y1))
        y2 = max(y1+1, min(h-1, y2))  # Ensure at least 1px height

        # Skip if validation fails
        if not is_valid_detection([x1, y1, x2, y2], frame.shape):
            continue

        # Class validation
        if class_ids is not None and i < len(class_ids):
            cls_id = int(class_ids[i])
            if cls_id >= len(CLASS_NAMES):
                continue

        valid_indices.append(i)

    if len(valid_indices) == 0:
        return frame

    # Extract valid detections
    valid_boxes = boxes[valid_indices]
    valid_classes = class_ids[valid_indices] if class_ids is not None else None

    # Draw grid for each valid detection
    for idx, box in enumerate(valid_boxes):
        x1, y1, x2, y2 = map(int, box)

        # Draw vertical grid lines
        x = x1 + GRID_SPACING
        while x < x2:
            cv2.line(overlay, (x, y1), (x, y2), GRID_COLOR, GRID_THICKNESS)
            x += GRID_SPACING

        # Draw horizontal grid lines
        y = y1 + GRID_SPACING
        while y < y2:
            cv2.line(overlay, (x1, y), (x2, y), GRID_COLOR, GRID_THICKNESS)
            y += GRID_SPACING

        # Draw border outline
        cv2.rectangle(overlay, (x1, y1), (x2, y2), GRID_COLOR, GRID_THICKNESS)

    # Apply the grid overlay with transparency
    cv2.addWeighted(overlay, GRID_ALPHA, frame, 1 - GRID_ALPHA, 0, frame)
    return frame

def main():
    print("Initializing Plastic Detection System...")
    print("=" * 50)

    # Load YOLO model
    print(f"Loading model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    # Initialize webcam with multiple backend attempts
    def init_webcam():
        backends = [
            (cv2.CAP_DSHOW, "DirectShow"),
            (cv2.CAP_MSMF, "Media Foundation"),
            (cv2.CAP_ANY, "Any Available")
        ]

        for backend_id, name in backends:
            print(f"Trying {name}...", end=" ")
            cam = cv2.VideoCapture(0, backend_id)
            if cam.isOpened():
                # Set properties
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, PROCESS_WIDTH)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, PROCESS_HEIGHT)
                cam.set(cv2.CAP_PROP_FPS, TARGET_FPS)

                # Test frame capture
                ret, test_frame = cam.read()
                if ret and test_frame is not None and test_frame.size > 0:
                    print("✓ Success")
                    return cam
                else:
                    print("✗ Opened but failed to read frame")
                    cam.release()
            else:
                print("✗ Failed to open")
                if cam.isOpened():
                    cam.release()

        raise RuntimeError("Could not initialize webcam with any backend")

    try:
        webcam = init_webcam()
    except RuntimeError as e:
        print(f"\n✗ Fatal error: {e}")
        print("\nTroubleshooting:")
        print("1. Close any applications using the webcam")
        print("2. Check Windows Camera privacy settings")
        print("3. Try a different USB port or restart PC")
        print("4. Update webcam drivers")
        return

    print(f"✓ Webcam initialized at {PROCESS_WIDTH}x{PROCESS_HEIGHT}")

    # Initialize systems
    stabilizer = DetectionStabilizer()
    frame_count = 0
    fps_history = deque(maxlen=30)
    detection_log = []
    last_log_time = 0

    print("\n" + "=" * 60)
    print("PLASTIC DETECTION WITH STABLE GRID HIGHLIGHTING")
    print("=" * 60)
    print("FEATURES:")
    print("  ✓ Stable grids (no flickering/jumping)")
    print("  ✓ False positive reduction (humans, etc.)")
    print("  ✓ Optimized performance")
    print("  ✓ Clear visual feedback")
    print()
    print("CONTROLS:")
    print("  'q' - Quit application")
    print("  's' - Save current frame")
    print("  'r' - Reset detection stability")
    print("  'l' - Show recent detection log")
    print("  'f' - Toggle FPS display detail")
    print("=" * 60)
    print("\nPOINT WEBCAM AT PLASTIC ITEMS TO SEE GRID HIGHLIGHTING\n")

    show_detailed_fps = False

    try:
        while True:
            frame_start = time.time()

            # Capture frame
            ret, frame = webcam.read()
            if not ret or frame is None:
                # Skip empty frames but don't spam warnings
                if frame_count % 100 == 0:
                    print("Warning: Empty frame received")
                time.sleep(0.005)
                continue

            frame_count += 1

            # Run detection
            results = model(frame, conf=CONFIDENCE_THRESHOLD)

            # Process detection results
            raw_boxes = np.empty((0, 4))
            raw_classes = np.empty((0,))
            raw_confidences = np.empty((0,))

            if len(results[0].boxes) > 0:
                # Extract data from YOLO results
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                # Apply confidence threshold
                conf_mask = confidences >= CONFIDENCE_THRESHOLD
                if np.any(conf_mask):
                    raw_boxes = boxes_xyxy[conf_mask]
                    raw_classes = class_ids[conf_mask]
                    raw_confidences = confidences[conf_mask]

            # Filter out likely false positives
            if len(raw_boxes) > 0:
                valid_indices = []
                for i, box in enumerate(raw_boxes):
                    if is_valid_detection(box, frame.shape):
                        valid_indices.append(i)

                if len(valid_indices) > 0:
                    filtered_boxes = raw_boxes[valid_indices]
                    filtered_classes = raw_classes[valid_indices]
                    filtered_confidences = raw_confidences[valid_indices]
                else:
                    filtered_boxes = np.empty((0, 4))
                    filtered_classes = np.empty((0,))
                    filtered_confidences = np.empty((0,))
            else:
                filtered_boxes = np.empty((0, 4))
                filtered_classes = np.empty((0,))
                filtered_confidences = np.empty((0,))

            # Update stabilizer with filtered detections
            has_detection = len(filtered_boxes) > 0
            is_stable, stable_boxes, stable_classes = stabilizer.update(
                has_detection,
                filtered_boxes if has_detection else None,
                filtered_classes if has_detection else None
            )

            # Log detections periodically (every 2 seconds)
            current_time = time.time()
            if current_time - last_log_time > 2.0 and has_detection:
                timestamp = time.strftime("%H:%M:%S")
                class_names = [CLASS_NAMES[min(int(cls), len(CLASS_NAMES)-1)] for cls in stable_classes]
                log_entry = f"[{timestamp}] Frame {frame_count}: {', '.join(class_names[:3])}"
                if len(stable_classes) > 3:
                    log_entry += f" (+{len(stable_classes)-3} more)"
                detection_log.append(log_entry)
                if len(detection_log) > 10:
                    detection_log.pop(0)
                last_log_time = current_time

                # Print to console every 2 seconds when detecting
                if has_detection:
                    print(log_entry)

            # Start with YOLO's visualization (shows what model detects)
            display_frame = results[0].plot()

            # OVERLAY our stable grid highlighting on stable detections only
            if is_stable and len(stable_boxes) > 0:
                display_frame = draw_precise_grid(display_frame, stable_boxes, stable_classes)

            # Calculate and track FPS
            frame_end = time.time()
            frame_process_time = frame_end - frame_start
            if frame_process_time > 0:
                current_fps = 1.0 / frame_process_time
                fps_history.append(current_fps)

            avg_fps = np.mean(fps_history) if len(fps_history) > 0 else 0

            # === DRAW INFORMATION OVERLAY ===
            info_y = 25
            line_height = 22

            # FPS display
            if show_detailed_fps and len(fps_history) >= 5:
                min_fps = np.min(fps_history)
                max_fps = np.max(fps_history)
                fps_text = f"FPS: {avg_fps:.1f} (min:{min_fps:.0f} max:{max_fps:.0f})"
            else:
                fps_text = f"FPS: {avg_fps:.1f}"
            cv2.putText(display_frame, fps_text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Detection count
            stable_count = len(stable_boxes)
            count_text = f"Stable Detections: {stable_count}"
            cv2.putText(display_frame, count_text, (10, info_y + line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Stability status
            status_text = f"Status: {'STABLE ✓' if is_stable else 'BUILDING... ○'}"
            status_color = (0, 255, 0) if is_stable else (0, 165, 255)
            cv2.putText(display_frame, status_text, (10, info_y + 2*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

            # Instructions (bottom)
            cv2.putText(display_frame, "'q'=quit 's'=save 'r'=reset 'l'=log 'f'=fps detail",
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show the final frame
            cv2.imshow('Plastic Detection - STABLE GRID HIGHLIGHTING', display_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nUser requested exit - shutting down...")
                break
            elif key == ord('s'):
                # Save frame
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}_{frame_count:04d}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"[SAVED] {filename}")
            elif key == ord('r'):
                # Reset stabilizer
                stabilizer = DetectionStabilizer()
                print("[RESET] Detection stability history cleared")
            elif key == ord('l'):
                # Show log
                print("\n--- RECENT DETECTION LOG ---")
                for entry in detection_log:
                    print(entry)
                if not detection_log:
                    print("[No detections logged yet]")
                print("--- END LOG ---\n")
            elif key == ord('f'):
                # Toggle FPS detail
                show_detailed_fps = not show_detailed_fps
                state = "DETAILED" if show_detailed_fps else "BASIC"
                print(f"[FPS] Display mode: {state}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean shutdown
        print("\nShutting down systems...")
        webcam.release()
        cv2.destroyAllWindows()

        # Final statistics
        print("\n" + "=" * 50)
        print("SESSION SUMMARY")
        print("=" * 50)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {np.mean(fps_history):.1f}" if len(fps_history) > 0 else "No FPS data")
        print(f"Detection log entries: {len(detection_log)}")
        if detection_log:
            print("Recent detections:")
            for entry in detection_log[-3:]:  # Show last 3
                print(f"  {entry}")
        print("\nThank you for using the Plastic Detection System!")
        print("=" * 50)

if __name__ == "__main__":
    main()