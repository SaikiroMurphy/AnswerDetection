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

modelPath = relative_to_assets("Yolov8s.pt")

model = YOLO(modelPath)

@app.post("/predict")
async def predict(image: UploadFile):
    img = Image.open(image.file).convert("RGB")

    start_infer_time = time.time()
    results = model.predict(img,
                            save=True,
                            show_labels=False,
                            imgsz=1024,
                            max_det=800,
                            conf=0.4,
                            iou=0.7,
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

        if origin_cls[idx] < 3:

            bigDict = {
                "box": [
                    float(origin_box[idx][0]),
                    float(origin_box[idx][1]),
                    float(origin_box[idx][2]),
                    float(origin_box[idx][3])],
                "line": [],
            }

            if origin_cls[idx] == 0:
                row1 = 10
                col1 = 6
                if origin_conf[idx] > elimSBD:
                    bigDict.update({"label": 'SBD'})
                    elimSBD = origin_conf[idx]
                    sortedList = utils.cellListH(origin_box, origin_cls, idx, indices, row1, col1)
                    bigDict = utils.mkDict(sortedList, bigDict, row1, col1)

                    jsonDict.append(bigDict)

                else:
                    continue
            elif origin_cls[idx] == 1:
                row2 = 10
                col2 = 3
                if origin_conf[idx] > elimMDT:
                    bigDict.update({"label": 'MDT'})
                    elimMDT = origin_conf[idx]
                    sortedList = utils.cellListH(origin_box, origin_cls, idx, indices, row2, col2)
                    bigDict = utils.mkDict(sortedList, bigDict, row2, col2)

                    jsonDict.append(bigDict)

                else:
                    continue

            elif origin_cls[idx] == 2:
                row3 = 5
                col3 = 4
                bigDict.update({"label": 'DA'})
                sortedList = utils.cellListV(origin_box, origin_cls, idx, indices, row3, col3)
                bigDict = utils.mkDict(sortedList, bigDict, col3, row3)

                ansDict.append(bigDict)

    # logger.info(len(ansDict))
    listDA = utils.sortAns(ansDict)
    for i in listDA:
        jsonDict.append(i)

    sheetDict = {"sheet": jsonDict}

    logger.info(f"Post processing time: {time.time() - time_process:.03f}s")

    return JSONResponse(content=sheetDict, status_code=200)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run("main:app", host="127.0.0.1", port=6969, reload=False)
