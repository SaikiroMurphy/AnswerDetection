import os, sys
import utils
from ultralytics import YOLO
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from PIL import Image
from fastapi.responses import JSONResponse
import uvicorn
import multiprocessing

app = FastAPI()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(".venv/")

def relative_to_assets(path: str) -> str:
    return str(ASSETS_PATH / Path(path))

modelPath = relative_to_assets("best.pt")

model = YOLO(modelPath)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    img = Image.open(image.file).convert("RGB")

    results = model.predict(img,
                            save=False,
                            # show_conf=True,
                            # show_labels=False,
                            max_det = 800,
                            # classes=[0],
                            # conf=0.45,
                            )

    # for result in results:
    origin_cls = results[0].boxes.cls
    origin_box = results[0].boxes.xyxyn
    origin_conf = results[0].boxes.conf

    box, cls, conf = utils.find_duplicates(origin_box, origin_cls, origin_conf)

    jsonDict = []
    ansDict = []

    # box = utils.remove_duplicate_boxes(origin_box)
    # print(box)

    elimSBD = 0
    elimMDT = 0

    for idxB, itemB in enumerate(cls):

        if int(itemB) < 3:

            bigDict = {
                "box": [
                    float(box[idxB][0]),
                    float(box[idxB][1]),
                    float(box[idxB][2]),
                    float(box[idxB][3])],
                "line": [],
            }

            if itemB == 0:
                row1 = 10
                col1 = 6
                if conf[idxB] > elimSBD:
                    bigDict.update({"label": 'SBD'})
                    elimSBD = conf[idxB]
                    sortedList = utils.cellListH(box, cls, idxB, row1, col1)
                    bigDict = utils.mkDict(sortedList, bigDict, row1, col1)

                    jsonDict.append(bigDict)

                else:
                    continue
            elif itemB == 1:
                row2 = 10
                col2 = 3
                if conf[idxB] > elimMDT:
                    bigDict.update({"label": 'MDT'})
                    elimMDT = conf[idxB]
                    sortedList = utils.cellListH(box, cls, idxB, row2, col2)
                    bigDict = utils.mkDict(sortedList, bigDict, row2, col2)

                    jsonDict.append(bigDict)

                else:
                    continue

            elif itemB == 2:
                row3 = 5
                col3 = 4
                bigDict.update({"label": 'DA'})
                sortedList = utils.cellListV(box, cls, idxB, row3, col3)
                bigDict = utils.mkDict(sortedList, bigDict, col3, row3)

                ansDict.append(bigDict)

    listDA = utils.sortAns(ansDict)
    for i in listDA:
        jsonDict.append(i)

    sheetDict = {"sheet": jsonDict}

    return JSONResponse(content=sheetDict, status_code=200)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    uvicorn.run(app, host="127.0.0.1", port=6969, reload=False)
