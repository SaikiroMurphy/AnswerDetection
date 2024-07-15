import torch
from statistics import mean
import numpy as np

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    Each box is defined by [x_min, y_min, x_max, y_max].
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_min < inter_x_max and inter_y_min < inter_y_max:
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    else:
        return 0

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = area1 + area2 - inter_area

    return inter_area / union_area

def remove_element(tensor, index):
    return torch.cat((tensor[:index], tensor[index+1:]))

def find_duplicates(bboxes, cls, conf, threshold=0.8):
    idxDel = set()

    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            iou = calculate_iou(bboxes[i], bboxes[j])
            if iou >= threshold:
                if float(conf[j]) < float(conf[i]):
                    idxDel.add(j)
                else:
                    idxDel.add(i)

    idxDel = sorted(idxDel, reverse=True)
    for i in idxDel:
        bboxes = remove_element(bboxes, i)
        cls = remove_element(cls, i)
        conf = remove_element(conf, i)

    return bboxes, cls, conf

def cellListH(box, cls, idxB, row, col):
    list = []
    colList = []
    sortedList = [ [0] * (row) for i in range(col) ]

    for idxL, itemL in enumerate(cls):
        if int(itemL) > 2:
            if float((box[idxL][0]) > float(box[idxB][0]) and
                    float(box[idxL][1]) > float(box[idxB][1]) and
                    float(box[idxL][2]) < float(box[idxB][2]) and
                    float(box[idxL][3]) < float(box[idxB][3])):

                list.append([itemL, box[idxL]])

    list.sort(key=lambda x: x[1][0])

    xComp = list[1]

    colN = 0
    rowN = 0
    for item in list:
        if (rowN < row and
                mean([float(item[1][0]), float(item[1][2])]) > xComp[1][0] and
                mean([float(item[1][0]), float(item[1][2])]) < xComp[1][2]):
            sortedList[colN][rowN] = item
            rowN += 1

        else:
            colN += 1
            rowN = 0
            sortedList[colN][rowN] = item
            xComp = item
            rowN += 1

    for col in sortedList:
        for cell in range(0, len(col)):
            for i in range(cell + 1, len(col)):
                if float("{:.3f}".format(col[cell][1][1])) > float("{:.3f}".format(col[i][1][1])):
                    temp = col[i]
                    col[i] = col[cell]
                    col[cell] = temp

    return sortedList

def cellListV(box, cls, idxB, row, col):
    list = []
    colList = []
    sortedList = [ [0] * (col) for i in range(row) ]

    for idxL, itemL in enumerate(cls):
        if int(itemL) > 2:
            if float((box[idxL][0]) > float(box[idxB][0]) and
                    float(box[idxL][1]) > float(box[idxB][1]) and
                    float(box[idxL][2]) < float(box[idxB][2]) and
                    float(box[idxL][3]) < float(box[idxB][3])):

                list.append([itemL, box[idxL]])

    list.sort(key=lambda x: x[1][1])

    xComp = list[1]

    colN = 0
    rowN = 0
    for item in list:
        if (colN < col and
                mean([float(item[1][1]), float(item[1][3])]) > xComp[1][1] and
                mean([float(item[1][1]), float(item[1][3])]) < xComp[1][3]):
            sortedList[rowN][colN] = item
            colN += 1

        else:
            rowN += 1
            colN = 0
            sortedList[rowN][colN] = item
            xComp = item
            colN += 1

    for row in sortedList:
        for cell in range(0, len(row)):
            for i in range(cell + 1, len(row)):
                if float("{:.3f}".format(row[cell][1][0])) > float("{:.3f}".format(row[i][1][0])):
                    temp = row[i]
                    row[i] = row[cell]
                    row[cell] = temp

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

def sortAns(jsonDict):
    listDA = []
    for item in jsonDict:
        if item["label"] == "DA":
            listDA.append(item)

    listDA.sort(key=lambda x: x["box"][0])

    comp = listDA[0]
    sortedList = [ [0] * (5) for i in range(2) ]
    colN = 0
    rowN = 0

    for item in listDA:
        if (mean([float(item["box"][0]), float(item["box"][2])]) > comp["box"][0] and
                mean([float(item["box"][0]), float(item["box"][2])]) < comp["box"][2]):
            sortedList[colN][rowN] = item
            rowN += 1

        else:
            colN += 1
            rowN = 0
            sortedList[colN][rowN] = item
            comp = item
            rowN += 1

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