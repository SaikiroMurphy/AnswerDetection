import loguru
from statistics import mean
import numpy as np



def cellListH(filtered_list):

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


def cellListV(filtered_list):

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
    sortedList = []
    subList = []

    for item in listDA:
        item_mean_x = mean([float(item["box"][0]), float(item["box"][2])])
        if comp["box"][0] < item_mean_x < comp["box"][2]:
            subList.append(item)
        else:
            sortedList.append(subList.copy())
            subList.clear()
            subList.append(item)
            comp = item

    sortedList.append(subList.copy())

    # Sort each row by the second value in the "box" list
    for row in sortedList:
        # loguru.logger.debug(row)
        row.sort(key=lambda x: x["box"][1])
        # loguru.logger.debug(row)


    # Flatten the sorted list
    listDA = [item for sublist in sortedList for item in sublist]

    return listDA
def sortAnsD(ansDict):
    # Filter out items with label "DA" and sort by the 2nd value in the "box" list
    listDA = sorted((item for item in ansDict if item["label"] == "DA"), key=lambda x: x["box"][1])

    comp = listDA[0]
    sortedList = []
    subList = []

    for item in listDA:
        item_mean_x = mean([float(item["box"][1]), float(item["box"][3])])
        if comp["box"][1] < item_mean_x < comp["box"][3]:
            subList.append(item)
        else:
            sortedList.append(subList.copy())
            subList.clear()
            subList.append(item)
            comp = item

    sortedList.append(subList.copy())

    # Sort each row by the first value in the "box" list
    for row in sortedList:
        # loguru.logger.debug(row)
        row.sort(key=lambda x: x["box"][0])
        # loguru.logger.debug(row)


    # Flatten the sorted list
    listDA = [item for sublist in sortedList for item in sublist]

    return listDA
