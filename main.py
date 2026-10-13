from pathlib import Path
import cv2
from ultralytics import YOLO

INPUT_PATH = Path("examples/input.jpg")
OUTPUT_PATH = Path("examples/output.jpg")

MODEL_PATH = Path("models/traffic_light_best.pt")
CONFIDENCE_THRESHOLD = 0.25

CLASS_COLORS = {
    "red" : (0, 0, 255),
    "yellow": (0, 255, 255),
    "green": (0, 255, 0),
}

def main():

    if not INPUT_PATH.is_file():
        print(f'Error: Could not find input file: {INPUT_PATH}')
        return

    if not MODEL_PATH.is_file():
        print(f"Error: Could not find model file: {MODEL_PATH}")
        return


    '''
    original
    model = YOLO("yolo11n.pt")
    '''

    # load trained model
    model = YOLO(str(MODEL_PATH))

    # run object detection on image
    result = model.predict(
        source=str(INPUT_PATH),
        conf=CONFIDENCE_THRESHOLD,
        imgsz=320,
        verbose=False
    )[0]

    # original image copy
    output_image = result.orig_img.copy()

    counts = {
        "red": 0,
        "yellow": 0,
        "green": 0,
    }

    traffic_light_count = 0

    # examine each detected object
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id].lower()
            confidence = float(box.conf[0])

            # Ignore any class that is not one of our target colours.
            if class_name not in CLASS_COLORS:
                continue

            counts[class_name] += 1

            # Extract bounding-box coordinates.
            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            box_color = CLASS_COLORS[class_name]

            # Draw a colour-specific bounding box.
            cv2.rectangle(
                output_image,
                (x1, y1),
                (x2, y2),
                box_color,
                2,
            )

            # Label the box with its predicted colour and confidence.
            label = f"{class_name.capitalize()}: {confidence:.2f}"

            # Measure the label so its background fits the text.
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )

            # Keep the label background within the image.
            label_top = max(0, y1 - text_height - baseline - 8)
            label_bottom = label_top + text_height + baseline + 8
            label_right = min(output_image.shape[1] - 1, x1 + text_width + 8)

            # Draw a solid background behind the label.
            cv2.rectangle(
                output_image,
                (x1, label_top),
                (label_right, label_bottom),
                box_color,
                -1,
            )

            # Draw readable text over the background.
            cv2.putText(
                output_image,
                label,
                (x1 + 4, label_bottom - baseline - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            print(
                f"Detected {class_name}: "
                f"confidence={confidence:.3f}, "
                f"box=({x1}, {y1}, {x2}, {y2})"
            )

        # Save the annotated image even when no objects were detected.
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    saved = cv2.imwrite(str(OUTPUT_PATH), output_image)

    if not saved:
        print(f"Error: Could not save output image to {OUTPUT_PATH}")
        return

    total = sum(counts.values())

    print("\nDetection summary:")
    print(f"Red lights:    {counts['red']}")
    print(f"Yellow lights: {counts['yellow']}")
    print(f"Green lights:  {counts['green']}")
    print(f"Total detected: {total}")

    if total == 0:
        print("No traffic lights detected.")

    print(f"\nAnnotated image saved to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
