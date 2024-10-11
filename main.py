import logging
import multiprocessing
import os
import sys
import time
import uuid
from pathlib import Path
from statistics import mean

import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
# from loguru import logger
from starlette.requests import Request

import utils
from YOLOv8_onnx import YOLOv8
import warnings

# Suppress all UserWarnings
warnings.simplefilter("ignore", category=UserWarning)

app = FastAPI()


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


def filter_list(origin_cls, origin_box, idx):
    return [[origin_cls[cnt], box]
            for cnt, box in enumerate(origin_box)
            if origin_cls[cnt] >= 3 and
            float(origin_box[idx][0]) < mean([float(box[0]), float(box[2])]) < float(origin_box[idx][2]) and
            float(origin_box[idx][1]) < mean([float(box[1]), float(box[3])]) < float(origin_box[idx][3])]


modelPath = relative_to_assets("Yolov8s-p2.onnx")


# model = YOLO(modelPath, task='detect')


@app.post("/predict")
async def predict(image: UploadFile = File(...), rows: bool = Form(...)):
    # Read the uploaded file
    contents = await image.read()

    # Convert the file contents to a numpy array
    np_arr = np.frombuffer(contents, np.uint8)
    # print(np_arr.shape)

    # start_infer_time = time.time()

    result = YOLOv8(modelPath, np_arr, confidence_thres=0.25, iou_thres=0.4)
    origin_box, origin_conf, origin_cls = result.main()

    # time_process = time.time()

    jsonDict = []
    ansDict = []

    elimSBD = 0
    elimMDT = 0

    for idx, cls in enumerate(origin_cls):
        origin_cls_val = cls

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
                    filtered_list = filter_list(origin_cls, origin_box, idx)
                    sortedList = utils.cellListH(filtered_list)
                    bigDict = utils.mkDict(sortedList, bigDict, row1)
                    jsonDict.append(bigDict)

            elif origin_cls_val == 1:
                row2 = 10
                if origin_conf[idx] > elimMDT:
                    bigDict["label"] = 'MDT'
                    elimMDT = origin_conf[idx]
                    filtered_list = filter_list(origin_cls, origin_box, idx)
                    sortedList = utils.cellListH(filtered_list)
                    bigDict = utils.mkDict(sortedList, bigDict, row2)
                    jsonDict.append(bigDict)

            elif origin_cls_val == 2:
                bigDict["label"] = 'DA'
                filtered_list = filter_list(origin_cls, origin_box, idx)
                if len(filtered_list) == 43:
                    col3 = 11
                    sortedList = utils.cellListH(filtered_list)
                    sortedList[-1].insert(0, [4, np.array([0, 0, 0, 0], dtype='float32')])

                elif len(filtered_list) == 8:
                    col3 = 2
                    sortedList = utils.cellListV(filtered_list)

                else:
                    col3 = 4
                    sortedList = utils.cellListV(filtered_list)

                bigDict = utils.mkDict(sortedList, bigDict, col3)
                ansDict.append(bigDict)

    # logger.info(len(ansDict))

    if rows:
        listDA = utils.sortAns(ansDict)
    else:
        listDA = utils.sortAnsD(ansDict)

    for i in listDA:
        jsonDict.append(i)

    sheetDict = {"sheet": jsonDict}

    # logger.info(f"Post processing time: {time.time() - time_process:.03f}s")
    # logger.info(f"Total processing time: {time.time() - start_infer_time:.03f}s")

    return JSONResponse(content=sheetDict, status_code=200)


def is_running_in_console():
    return sys.stdout is not None and sys.stdout.isatty()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    config = uvicorn.Config(app, log_config=None, host="0.0.0.0", port=6969, reload=False)
    if is_running_in_console():
        logging.basicConfig(level=logging.ERROR)
    else:
        logging.basicConfig(filename='app.log', level=logging.ERROR)
    server = uvicorn.Server(config)
    server.run()
