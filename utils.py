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


def cellListH(box, cls, idx, indices, row, col):
    list = []
    sortedList = [[0] * row for i in range(col)]

    # loguru.logger.info(len(indices))

    for cnt in indices:
        if int(cls[cnt]) > 2:
            if (float(box[idx][0]) < mean([float(box[cnt][0]), float(box[cnt][2])]) < float(box[idx][2]) and
                    float(box[idx][1]) < mean([float(box[cnt][1]), float(box[cnt][3])]) < float(box[idx][3])):
                list.append([cls[cnt], box[cnt]])

    list.sort(key=lambda x: x[1][0])

    # loguru.logger.info(len(list))
    xComp = list[0]

    colN = 0
    rowN = 0
    for item in list:
        if (rowN < row and
                xComp[1][0] < mean([float(item[1][0]), float(item[1][2])]) < xComp[1][2]):
            # loguru.logger.debug(rowN)
            sortedList[colN][rowN] = item
            rowN += 1

        else:
            colN += 1
            rowN = 0
            # loguru.logger.debug(colN)
            sortedList[colN][rowN] = item
            xComp = item
            rowN += 1

    for col in sortedList:
        col.sort(key=lambda x: x[1][1])
        # for cell in range(0, len(col)):
        #     for i in range(cell + 1, len(col)):
        #         if float("{:.3f}".format(col[cell][1][1])) > float("{:.3f}".format(col[i][1][1])):
        #             temp = col[i]
        #             col[i] = col[cell]
        #             col[cell] = temp

    # loguru.logger.info(f"Sorted List: {sortedList}")
    return sortedList


def cellListV(box, cls, idx, indices, row, col):
    list = []
    sortedList = [[0] * col for i in range(row)]

    # loguru.logger.info(len(indices))

    for cnt in indices:
        if int(cls[cnt]) > 2:
            if (float(box[idx][0]) < mean([float(box[cnt][0]), float(box[cnt][2])]) < float(box[idx][2]) and
                    float(box[idx][1]) < mean([float(box[cnt][1]), float(box[cnt][3])]) < float(box[idx][3])):
                list.append([cls[cnt], box[cnt]])

    list.sort(key=lambda x: x[1][1])

    # loguru.logger.info(len(list))
    xComp = list[0]

    colN = 0
    rowN = 0
    for item in list:
        if (colN < col and
                xComp[1][1] < mean([float(item[1][1]), float(item[1][3])]) < xComp[1][3]):
            sortedList[rowN][colN] = item
            colN += 1

        else:
            rowN += 1
            colN = 0
            sortedList[rowN][colN] = item
            xComp = item
            colN += 1

    for row in sortedList:
        row.sort(key=lambda x: x[1][0])

        for cell in range(0, len(row)):
            for i in range(cell + 1, len(row)):
                if float("{:.3f}".format(row[cell][1][0])) > float("{:.3f}".format(row[i][1][0])):
                    temp = row[i]
                    row[i] = row[cell]
                    row[cell] = temp

    # loguru.logger.info(f"Sorted List: {sortedList}")
    return sortedList


def mkDict(sortedList, bigDict, row, col):
    innerDict = []
    for coll in sortedList:
        for cell in coll:
            if int(cell[0]) == 3:
                itemD = {
                    "box": [
                        float(cell[1][0]),
                        float(cell[1][1]),
                        float(cell[1][2]),
                        float(cell[1][3])
                    ],
                    "label": "o",
                },

            elif int(cell[0]) == 4:
                itemD = {
                    "box": [
                        float(cell[1][0]),
                        float(cell[1][1]),
                        float(cell[1][2]),
                        float(cell[1][3])],
                    "label": "x",
                },

            innerDict.extend(itemD)

    arr = np.array(innerDict)
    reshape = arr.reshape(col, row)

    bigDict["line"].extend(reshape.tolist())
    return bigDict


def sortAns(ansDict):
    listDA = []
    for item in ansDict:
        if item["label"] == "DA":
            listDA.append(item)

    listDA.sort(key=lambda x: x["box"][0])

    # loguru.logger.info(f"Answer List: {listDA}")

    comp = listDA[0]
    sortedList = [[0] * 5 for i in range(2)]
    colN = 0
    rowN = 0

    for item in listDA:
        if comp["box"][0] < mean([float(item["box"][0]), float(item["box"][2])]) < comp["box"][2]:
            sortedList[colN][rowN] = item
            rowN += 1

        else:
            colN += 1
            rowN = 0
            sortedList[colN][rowN] = item
            comp = item
            rowN += 1

    # loguru.logger.info(sortedList)

    for row in sortedList:
        for cell in range(0, len(row)):
            for i in range(cell + 1, len(row)):
                if row[cell]["box"][1] > row[i]["box"][1]:
                    temp = row[i]
                    row[i] = row[cell]
                    row[cell] = temp

    listDA.clear()
    for item in sortedList:
        for i in item:
            listDA.append(i)

    return listDA
