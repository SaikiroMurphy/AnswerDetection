# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np
import onnxruntime as ort
import time
from loguru import logger


class YOLOv8:
    """YOLOv8 object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model, input_image, confidence_thres, iou_thres):
        """
        Initializes an instance of the YOLOv8 class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model = onnx_model
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        # Load the class names from the COCO dataset
        self.classes = ['SBD', 'MDT', 'DA', 'O', 'X']

        # Generate a color palette for the classes
        self.color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """

        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create the label text with class name and score
        # label = f"{self.classes[class_id]}: {score:.2f}"

        # Calculate the dimensions of the label text
        # (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        # label_x = x1
        # label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # # Draw a filled rectangle as the background for the label text
        # cv2.rectangle(
        #     img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        # )

        # # Draw the label text on the image
        # cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def resize_and_pad_image(self, target_size):
        """
        Resize and pad the image to the target size while maintaining aspect ratio.

        Args:
            image (np.ndarray): Input image.
            target_size (tuple): Target size (width, height).

        Returns:
            np.ndarray: Resized and padded image.
        """
        target_width, target_height = target_size

        # Calculate the scaling factor to maintain aspect ratio
        scaling_factor = min(target_width / self.img_width, target_height / self.img_height)

        # Resize the image
        resized_width = int(self.img_width * scaling_factor)
        resized_height = int(self.img_height * scaling_factor)
        # print(self.img)
        resized_image = cv2.resize(self.img, (resized_width, resized_height))
        # print(resized_image.shape)
        # Calculate padding to center the resized image
        pad_top = (target_height - resized_height) // 2
        pad_bottom = target_height - resized_height - pad_top
        pad_left = (target_width - resized_width) // 2
        pad_right = target_width - resized_width - pad_left

        # Pad the resized image to make it exactly target_size
        padded_image = cv2.copyMakeBorder(
            resized_image,
            pad_top, pad_bottom, pad_left, pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Padding color (black)
        )

        return padded_image

    def preprocess(self, target_size):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imdecode(self.input_image, cv2.IMREAD_UNCHANGED)

        self.img_height, self.img_width = self.img.shape[:2]

        # print(self.img.shape)
        # exit(0)

        padded_image = self.resize_and_pad_image(target_size=target_size)

        # print(padded_image.shape)
        blob = cv2.dnn.blobFromImage(padded_image, scalefactor=1 / 255.0, size=target_size, swapRB=True, crop=False)

        # print(blob.shape)
        # exit(0)

        # Return the preprocessed image data
        return blob

    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """

        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []
        normalize_boxes = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                right = int((x + w / 2) * x_factor)
                bottom = int((y + h / 2) * y_factor)

                x1 = (x - w/2) / 640.0
                y1 = (y - h/2) / 640.0
                x2 = (x + w/2) / 640.0
                y2 = (y + h/2) / 640.0

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])
                normalize_boxes.append([x1, y1, x2, y2])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        # for i in indices:
            # Get the box, score, and class ID corresponding to the index
            # box = boxes[i]
            # score = scores[i]
            # class_id = class_ids[i]

            # Draw the detection on the input image
            # self.draw_detections(input_image, box, class_id)

        filtered_boxes = [normalize_boxes[i] for i in indices]
        filtered_scores = [scores[i] for i in indices]
        filtered_classes = [class_ids[i] for i in indices]


        # Return the modified input image
        return filtered_boxes, filtered_scores, filtered_classes

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        preprocess_time = time.time()

        # Create an inference session using the ONNX model and specify execution providers
        session = ort.InferenceSession(self.onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

        # Get the model inputs
        model_inputs = session.get_inputs()

        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Preprocess the image data
        img_data = self.preprocess((640, 640))

        logger.info(f"Preprocess time: {time.time() - preprocess_time:.03f}s")

        # Run inference using the preprocessed image data
        start_infer_time = time.time()

        outputs = session.run(None, {model_inputs[0].name: img_data})

        logger.info(f"Infer time: {time.time() - start_infer_time:.03f}s")

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(outputs)  # output image
