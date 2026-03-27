import argparse


def detect_objects_yolo_stub(image_path: str) -> list[str]:
    # YOLO does object detection in one forward pass; this is a tiny placeholder.
    if "night" in image_path.lower() or "rain" in image_path.lower():
        return ["car", "traffic_light"]
    return ["car", "lane", "traffic_sign"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny CV starter")
    parser.add_argument("--image", default="sample.jpg", help="Path to image")
    args = parser.parse_args()

    detected = detect_objects_yolo_stub(args.image)
    print("YOLO starter (stub)")
    print(f"image: {args.image}")
    print(f"detected: {', '.join(detected)}")


if __name__ == "__main__":
    main()
