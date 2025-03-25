from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from my_emergency_final import *

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# 요약, 등급, 등급이 3 이상이면 Null 출력, 아니면 병원 정보 출력.

@app.get("/hospital_by_module")
def emergency(request : str, lat : float, lon : float, num : int):
    load_key('./')
    summary = text2summary(request)
    pred, _ = model_prediction('./', summary)
    
    if pred+1 <= 2 :
        result = recom_em('./', lat, lon, num)
        res = []
        for i in range(num) :
            res.append({"hospitalName" : result[i][1], "address" : result[i][2], "phone" : result[i][3], "distance" : result[i][0]})
        return res
    
