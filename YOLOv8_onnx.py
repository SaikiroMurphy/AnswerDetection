# Ultralytics YOLO 🚀, AGPL-3.0 license

import cv2
import numpy as np
import onnxruntime as ort
import time
# from loguru import logger


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
        self.scaling_factor = None
        self.input_image = input_image
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres

        self.onnx_model = onnx_model
        self.session = ort.InferenceSession(self.onnx_model,
                                            providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        model_inputs = self.session.get_inputs()
        input_shape = model_inputs[0].shape
        self.input_width = input_shape[2]
        self.input_height = input_shape[3]

        # Load the class names from the COCO dataset
        self.classes = ['SBD', 'MDT', 'DA', 'O', 'X']

    def preprocess(self, target_size):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Decode the image
        self.input_image = cv2.imdecode(self.input_image, cv2.IMREAD_UNCHANGED)
        self.img_height, self.img_width = self.input_image.shape[:2]

        # Calculate the scaling factor for resizing
        scaling_factor = min(target_size[0] / self.img_height, target_size[1] / self.img_width)

        # Resize the image with padding (if needed)
        resized_image = cv2.resize(self.input_image, (0, 0), fx=scaling_factor, fy=scaling_factor)
        padded_image = cv2.copyMakeBorder(
            resized_image,
            0,
            target_size[1] - resized_image.shape[0],
            0,
            target_size[0] - resized_image.shape[1],
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

        # Convert to blob format
        blob = cv2.dnn.blobFromImage(padded_image, scalefactor=1 / 255.0, size=target_size, swapRB=True, crop=False)

        # Update scaling factor in the object
        self.scaling_factor = 1 / scaling_factor

        # Return the preprocessed image data
        return blob

    def postprocess(self, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The filtered bounding boxes, scores, and class IDs.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.squeeze(output[0]).T

        # Extract the class scores and bounding boxes
        class_scores = outputs[:, 4:]
        bbox_coords = outputs[:, :4]

        # Calculate the maximum scores and corresponding class IDs
        max_scores = np.amax(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        # Apply the confidence threshold
        valid_indices = max_scores >= self.confidence_thres
        valid_scores = max_scores[valid_indices]
        valid_boxes = bbox_coords[valid_indices]
        valid_class_ids = class_ids[valid_indices]

        # Calculate the scaled bounding box coordinates
        left = valid_boxes[:, 0] - valid_boxes[:, 2] / 2
        top = valid_boxes[:, 1] - valid_boxes[:, 3] / 2
        right = valid_boxes[:, 0] + valid_boxes[:, 2] / 2
        bottom = valid_boxes[:, 1] + valid_boxes[:, 3] / 2

        x1 = left * self.scaling_factor / self.img_width
        y1 = top * self.scaling_factor / self.img_height
        x2 = right * self.scaling_factor / self.img_width
        y2 = bottom * self.scaling_factor / self.img_height

        normalize_boxes = np.vstack((x1, y1, x2, y2)).T
        boxes = np.vstack((left, top, valid_boxes[:, 2], valid_boxes[:, 3])).T

        # Apply non-maximum suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), valid_scores.tolist(), self.confidence_thres, self.iou_thres)

        # Filter the boxes, scores, and class IDs based on NMS
        filtered_boxes = normalize_boxes[indices].tolist()
        filtered_scores = valid_scores[indices].tolist()
        filtered_classes = valid_class_ids[indices].tolist()

        return filtered_boxes, filtered_scores, filtered_classes

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """

        preprocess_time = time.time()

        # Preprocess the image data
        img_data = self.preprocess((640, 640))

        # logger.info(f"Preprocess time: {time.time() - preprocess_time:.03f}s")

        # Run inference using the preprocessed image data
        start_infer_time = time.time()

        outputs = self.session.run(None, {self.session.get_inputs()[0].name: img_data})

        # logger.info(f"Infer time: {time.time() - start_infer_time:.03f}s")

        # Perform post-processing on the outputs to obtain output image.
        return self.postprocess(outputs)  # output image
