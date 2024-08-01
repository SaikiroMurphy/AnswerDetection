import os, sys
import utils
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.responses import JSONResponse
import uvicorn
import multiprocessing
from typing import List
import time
from loguru import logger
from starlette.requests import Request
import uuid
from ultralytics import YOLO
from statistics import mean
import numpy as np

app = FastAPI()

#
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    return response


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path("./")

def relative_to_assets(path: str) -> str:
    return str(ASSETS_PATH / Path(path))

modelPath = relative_to_assets("Yolov8s-p2.pt")

model = YOLO(modelPath)

@app.post("/predict")
async def predict(image: UploadFile):
    img = Image.open(image.file).convert("RGB")

    start_infer_time = time.time()
    results = model.predict(img,
                            save=True,
                            show_labels=False,
                            imgsz=640,
                            max_det=800,
                            conf=0.25,
                            iou=0.4,
                            )

    logger.info(f"Infer time: {time.time() - start_infer_time:.03f}s")

    origin_cls = results[0].boxes.cls.numpy()
    origin_box = results[0].boxes.xyxyn.numpy()
    origin_conf = results[0].boxes.conf

    start_find_dup_time = time.time()


    indices = utils.non_maximum_suppression(origin_box, origin_conf, iou_threshold=0.98)
    # logger.info(len(indices))
    logger.info(f"Find duplicate time: {time.time() - start_find_dup_time:.03f}s")

    time_process = time.time()

    jsonDict = []
    ansDict = []

    # box = utils.remove_duplicate_boxes(origin_box)
    # print(box)

    elimSBD = 0
    elimMDT = 0

    for idx in indices:
        origin_cls_val = origin_cls[idx]

        if origin_cls_val < 3:
            origin_box_val = origin_box[idx]
            bigDict = {
                "box": [
                    float(origin_box_val[0]),
                    float(origin_box_val[1]),
                    float(origin_box_val[2]),
                    float(origin_box_val[3])
                ],
                "line": []
            }

            if origin_cls_val == 0:
                row1 = 10
                if origin_conf[idx] > elimSBD:
                    bigDict["label"] = 'SBD'
                    elimSBD = origin_conf[idx]
                    filtered_list = [[origin_cls[cnt], origin_box[cnt]]
                                     for cnt in indices
                                     if int(origin_cls[cnt]) > 2 and
                                     float(origin_box[idx][0]) < mean([float(origin_box[cnt][0]), float(origin_box[cnt][2])]) < float(
                            origin_box[idx][2]) and
                                     float(origin_box[idx][1]) < mean([float(origin_box[cnt][1]), float(origin_box[cnt][3])]) < float(
                            origin_box[idx][3])
                                     ]
                    sortedList = utils.cellListH(filtered_list)
                    bigDict = utils.mkDict(sortedList, bigDict, row1)
                    jsonDict.append(bigDict)

            elif origin_cls_val == 1:
                row2 = 10
                if origin_conf[idx] > elimMDT:
                    bigDict["label"] = 'MDT'
                    elimMDT = origin_conf[idx]
                    filtered_list = [[origin_cls[cnt], origin_box[cnt]]
                                     for cnt in indices
                                     if int(origin_cls[cnt]) > 2 and
                                     float(origin_box[idx][0]) < mean([float(origin_box[cnt][0]), float(origin_box[cnt][2])]) < float(
                            origin_box[idx][2]) and
                                     float(origin_box[idx][1]) < mean([float(origin_box[cnt][1]), float(origin_box[cnt][3])]) < float(
                            origin_box[idx][3])
                                     ]
                    sortedList = utils.cellListH(filtered_list)
                    bigDict = utils.mkDict(sortedList, bigDict, row2)
                    jsonDict.append(bigDict)

            elif origin_cls_val == 2:
                bigDict["label"] = 'DA'
                filtered_list = [[origin_cls[cnt], origin_box[cnt]]
                                 for cnt in indices
                                 if int(origin_cls[cnt]) > 2 and
                                 float(origin_box[idx][0]) < mean(
                        [float(origin_box[cnt][0]), float(origin_box[cnt][2])]) < float(
                        origin_box[idx][2]) and
                                 float(origin_box[idx][1]) < mean(
                        [float(origin_box[cnt][1]), float(origin_box[cnt][3])]) < float(
                        origin_box[idx][3])
                                 ]
                logger.debug(len(filtered_list))
                if len(filtered_list) == 43:
                    col3 = 11
                    sortedList = utils.cellListH(filtered_list)
                    # logger.debug(sortedList)
                    sortedList[-1].insert(0, [4, np.array([0, 0, 0, 0], dtype='float32')])

                elif len(filtered_list) == 8:
                    col3 = 2
                    sortedList = utils.cellListV(filtered_list)
                    # logger.debug(sortedList)

                else:
                    col3 = 4
                    sortedList = utils.cellListV(filtered_list)
                bigDict = utils.mkDict(sortedList, bigDict, col3)
                ansDict.append(bigDict)

    # logger.info(len(ansDict))
    listDA = utils.sortAns(ansDict)
    # logger.debug(listDA)
    for i in listDA:
        jsonDict.append(i)

    sheetDict = {"sheet": jsonDict}

    logger.info(f"Post processing time: {time.time() - time_process:.03f}s")
    logger.info(f"Total processing time: {time.time() - start_infer_time:.03f}s")

    return JSONResponse(content=sheetDict, status_code=200)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host="0.0.0.0", port=6969, reload=False)
