
import cv2
import json
import os
import itertools
from pathlib import Path


CASCADE_PATH = "models/haarcascade_frontalface_default.xml"

# Test/validation image directories
POSITIVE_DIR = "test_images/positive"
NEGATIVE_DIR = "test_images/negative"

# Output configuration
OUTPUT_DIR = "models"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "best_params.json")


def load_images(directory):
    """
    Load all supported images from a directory.

    Returns:
        List of grayscale images.
    """

    images = []

    if not os.path.exists(directory):
        print(f"Warning: Directory not found: {directory}")
        return images

    supported_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )

    for filename in os.listdir(directory):

        if not filename.lower().endswith(supported_extensions):
            continue

        path = os.path.join(directory, filename)

        image = cv2.imread(path)

        if image is None:
            print(f"Could not read: {path}")
            continue

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        images.append({
            "filename": filename,
            "image": gray
        })

    return images



def evaluate_parameters(
    cascade,
    positive_images,
    negative_images,
    scale_factor,
    min_neighbors,
    min_size,
    max_size
):
    """
    Evaluate one Haar Cascade parameter configuration.

    Positive image:
        At least one face should be detected.

    Negative image:
        No face should be detected.

    Returns:
        accuracy, precision, recall, f1
    """

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0


    for item in positive_images:

        image = item["image"]

        faces = cascade.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            maxSize=max_size
        )

        if len(faces) > 0:
            true_positive += 1
        else:
            false_negative += 1


    for item in negative_images:

        image = item["image"]

        faces = cascade.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            maxSize=max_size
        )

        if len(faces) > 0:
            false_positive += 1
        else:
            true_negative += 1


    total = (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )

    accuracy = (
        (true_positive + true_negative) / total
        if total > 0
        else 0
    )

    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0
    )

    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0
    )

    if precision + recall > 0:
        f1 = (
            2 * precision * recall
            / (precision + recall)
        )
    else:
        f1 = 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "tp": true_positive,
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative
    }


def main():

    print("=" * 60)
    print("OPTIMIZED HAAR CASCADE - PARAMETER FINE-TUNING")
    print("=" * 60)



    if not os.path.exists(CASCADE_PATH):

        print("\nERROR:")
        print(f"Haar Cascade file not found:")
        print(CASCADE_PATH)

        return

    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if cascade.empty():

        print("\nERROR:")
        print("Could not load Haar Cascade classifier.")

        return

    print("\nHaar Cascade loaded successfully.")



    print("\nLoading positive images...")
    positive_images = load_images(POSITIVE_DIR)

    print(f"Positive images: {len(positive_images)}")

    print("\nLoading negative images...")
    negative_images = load_images(NEGATIVE_DIR)

    print(f"Negative images: {len(negative_images)}")

    if len(positive_images) == 0 or len(negative_images) == 0:

        print("\nERROR:")
        print("Both positive and negative image directories")
        print("must contain images.")

        print("\nExpected structure:")
        print("""
test_images/
├── positive/
│   ├── face1.jpg
│   ├── face2.jpg
│   └── ...
│
└── negative/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
""")

        return


    scale_factors = [
        1.05,
        1.10,
        1.15,
        1.20,
        1.25
    ]

    min_neighbors_values = [
        3,
        4,
        5,
        6,
        7
    ]

    min_sizes = [
        (20, 20),
        (30, 30),
        (40, 40)
    ]

    max_sizes = [
        (300, 300),
        (500, 500),
        (1000, 1000)
    ]

    combinations = itertools.product(
        scale_factors,
        min_neighbors_values,
        min_sizes,
        max_sizes
    )


    best_result = None
    best_parameters = None

    total_combinations = (
        len(scale_factors)
        * len(min_neighbors_values)
        * len(min_sizes)
        * len(max_sizes)
    )

    print("\nStarting parameter fine-tuning...")
    print(f"Total configurations: {total_combinations}")
    print()

    for index, parameters in enumerate(combinations, start=1):

        (
            scale_factor,
            min_neighbors,
            min_size,
            max_size
        ) = parameters

        result = evaluate_parameters(
            cascade,
            positive_images,
            negative_images,
            scale_factor,
            min_neighbors,
            min_size,
            max_size
        )

        if (
            best_result is None
            or result["f1_score"] > best_result["f1_score"]
        ):

            best_result = result

            best_parameters = {
                "scaleFactor": scale_factor,
                "minNeighbors": min_neighbors,
                "minSize": list(min_size),
                "maxSize": list(max_size)
            }

            print(
                f"[{index}/{total_combinations}] "
                f"New best F1: "
                f"{result['f1_score']:.4f}"
            )


    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output = {
        "model": "OpenCV Haar Cascade",
        "classifier": CASCADE_PATH,
        "parameters": best_parameters,
        "metrics": best_result
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(output, file, indent=4)


    print("\n")
    print("=" * 60)
    print("FINE-TUNING COMPLETED")
    print("=" * 60)

    print("\nBest Parameters:")
    print(
        f"scaleFactor  : "
        f"{best_parameters['scaleFactor']}"
    )

    print(
        f"minNeighbors : "
        f"{best_parameters['minNeighbors']}"
    )

    print(
        f"minSize      : "
        f"{best_parameters['minSize']}"
    )

    print(
        f"maxSize      : "
        f"{best_parameters['maxSize']}"
    )

    print("\nPerformance:")

    print(
        f"Accuracy     : "
        f"{best_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision    : "
        f"{best_result['precision']:.4f}"
    )

    print(
        f"Recall       : "
        f"{best_result['recall']:.4f}"
    )

    print(
        f"F1-Score     : "
        f"{best_result['f1_score']:.4f}"
    )

    print("\nConfusion Matrix:")
    print(f"True Positive  : {best_result['tp']}")
    print(f"True Negative  : {best_result['tn']}")
    print(f"False Positive : {best_result['fp']}")
    print(f"False Negative : {best_result['fn']}")

    print("\nBest configuration saved to:")
    print(OUTPUT_FILE)

    print("=" * 60)


if __name__ == "__main__":
    main()