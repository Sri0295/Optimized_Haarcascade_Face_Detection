
import cv2
import json
import os
import argparse



CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
PARAMETERS_PATH = "models/best_params.json"

DEFAULT_OUTPUT = "results/fine_tuned_detection.jpg"


def load_parameters():

    if not os.path.exists(PARAMETERS_PATH):

        print("ERROR:")
        print(f"Parameter file not found:")
        print(PARAMETERS_PATH)

        print("\nRun train.py first.")

        return None

    with open(PARAMETERS_PATH, "r") as file:
        data = json.load(file)

    return data["parameters"]



def detect_faces(image, cascade, parameters):

    # Convert image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Improve contrast
    gray = cv2.equalizeHist(gray)

    # Haar Cascade detection
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=parameters["scaleFactor"],
        minNeighbors=parameters["minNeighbors"],
        minSize=tuple(parameters["minSize"]),
        maxSize=tuple(parameters["maxSize"])
    )

    return faces



def draw_detections(image, faces):

    output = image.copy()

    for (x, y, w, h) in faces:

        # Draw bounding box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Label
        cv2.putText(
            output,
            "Face",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output



def main():

    parser = argparse.ArgumentParser(
        description="Fine-Tuned Haar Cascade Face Detection"
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to save detected image"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FINE-TUNED HAAR CASCADE FACE DETECTION")
    print("=" * 60)



    if not os.path.exists(CASCADE_PATH):

        print("\nERROR:")
        print("Haar Cascade classifier not found:")
        print(CASCADE_PATH)

        return

    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    if cascade.empty():

        print("\nERROR:")
        print("Failed to load Haar Cascade classifier.")

        return

    print("\nHaar Cascade loaded.")


    parameters = load_parameters()

    if parameters is None:
        return

    print("\nFine-tuned parameters:")

    print(
        f"scaleFactor  : "
        f"{parameters['scaleFactor']}"
    )

    print(
        f"minNeighbors : "
        f"{parameters['minNeighbors']}"
    )

    print(
        f"minSize      : "
        f"{parameters['minSize']}"
    )

    print(
        f"maxSize      : "
        f"{parameters['maxSize']}"
    )


    image = cv2.imread(args.image)

    if image is None:

        print("\nERROR:")
        print(f"Could not read image: {args.image}")

        return

    print(f"\nInput image: {args.image}")



    faces = detect_faces(
        image,
        cascade,
        parameters
    )

    print(
        f"\nFaces detected: {len(faces)}"
    )



    output = draw_detections(
        image,
        faces
    )


    output_directory = os.path.dirname(args.output)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    cv2.imwrite(
        args.output,
        output
    )

    print(
        f"Result saved to: {args.output}"
    )


    cv2.imshow(
        "Fine-Tuned Haar Cascade Face Detection",
        output
    )

    print("\nPress any key to close the image.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("\nDetection completed.")


if __name__ == "__main__":
    main()