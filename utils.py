import loguru
from statistics import mean
import numpy as np


def non_maximum_suppression(boxes, scores, iou_threshold):
    # Ensure boxes and scores are numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)

    # Initialize a list to hold the indices of the final boxes
    keep_indices = []

    # Sort the boxes based on the scores in descending order
    sorted_indices = np.argsort(scores)[::-1]

    while len(sorted_indices) > 0:
        # Pick the box with the highest score
        current_index = sorted_indices[0]
        keep_indices.append(current_index)

        # Compute the IoU (Intersection over Union) of the picked box with the rest
        current_box = boxes[current_index]
        rest_boxes = boxes[sorted_indices[1:]]

        iou = compute_iou(current_box, rest_boxes)

        # Keep only boxes with IoU less than the threshold
        filtered_indices = np.where(iou < iou_threshold)[0]
        sorted_indices = sorted_indices[filtered_indices + 1]

    return keep_indices


def compute_iou(box, boxes):
    # Calculate the coordinates of the intersection box
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    # Calculate the area of intersection
    intersection_area = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

    # Calculate the area of the boxes
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)

    # Calculate the union area
    union_area = box_area + boxes_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area
    return iou


def cellListH(box, cls, idx, indices):
    filtered_list = [[cls[cnt], box[cnt]]
        for cnt in indices
        if int(cls[cnt]) > 2 and
        float(box[idx][0]) < mean([float(box[cnt][0]), float(box[cnt][2])]) < float(box[idx][2]) and
        float(box[idx][1]) < mean([float(box[cnt][1]), float(box[cnt][3])]) < float(box[idx][3])
    ]

    filtered_list.sort(key=lambda x: x[1][0])
    # loguru.logger.debug(len(filtered_list))

    sortedList = []
    xComp = filtered_list[0]
    colList = []

    for item in filtered_list:
        item_mean_x = mean([float(item[1][0]), float(item[1][2])])
        if xComp[1][0] < item_mean_x < xComp[1][2]:
            colList.append(item)
        else:
            # loguru.logger.debug(sortedList)
            sortedList.append(colList.copy())
            # loguru.logger.debug(sortedList)

            colList.clear()
            colList.append(item)
            xComp = item

    sortedList.append(colList.copy())
    # loguru.logger.debug(len(sortedList))

    for c in sortedList:
        c.sort(key=lambda x: x[1][1])
    # loguru.logger.debug(sortedList)

    return sortedList


def cellListV(box, cls, idx, indices):
    filtered_list = [[cls[cnt], box[cnt]]
        for cnt in indices
        if int(cls[cnt]) > 2 and
        float(box[idx][0]) < mean([float(box[cnt][0]), float(box[cnt][2])]) < float(box[idx][2]) and
        float(box[idx][1]) < mean([float(box[cnt][1]), float(box[cnt][3])]) < float(box[idx][3])
    ]

    filtered_list.sort(key=lambda x: x[1][1])

    sortedList = []
    xComp = filtered_list[0]
    rowList = []

    for item in filtered_list:
        item_mean_y = mean([float(item[1][1]), float(item[1][3])])
        if xComp[1][1] < item_mean_y < xComp[1][3]:
            rowList.append(item)
        else:
            sortedList.append(rowList.copy())
            rowList.clear()
            rowList.append(item)
            xComp = item

    sortedList.append(rowList.copy())
    # loguru.logger.debug(sortedList)

    for r in sortedList:
        # loguru.logger.info(r)
        r.sort(key=lambda x: x[1][0])

    return sortedList


def mkDict(sortedList, bigDict, col):
    innerDict = [
        {
            "box": [
                float(cell[1][0]),
                float(cell[1][1]),
                float(cell[1][2]),
                float(cell[1][3])
            ],
            "label": "o" if int(cell[0]) == 3 else "x"
        }
        for coll in sortedList
        for cell in coll
        if int(cell[0]) in {3, 4}
    ]

    arr = np.array(innerDict)
    # loguru.logger.info(len(arr))
    reshape = arr.reshape(round(len(arr)/col), col)

    bigDict["line"].extend(reshape.tolist())
    return bigDict


def sortAns(ansDict):
    # Filter out items with label "DA" and sort by the first value in the "box" list
    listDA = sorted((item for item in ansDict if item["label"] == "DA"), key=lambda x: x["box"][0])

    comp = listDA[0]
    sortedList = [[]]
    colN = 0

    for item in listDA:
        item_mean_x = mean([float(item["box"][0]), float(item["box"][2])])
        if comp["box"][0] < item_mean_x < comp["box"][2]:
            sortedList[colN].append(item)
        else:
            colN += 1
            sortedList.append([item])
            comp = item

    # Sort each row by the second value in the "box" list
    for row in sortedList:
        row.sort(key=lambda x: x["box"][1])

    # Flatten the sorted list
    listDA = [item for sublist in sortedList for item in sublist]

    return listDA
