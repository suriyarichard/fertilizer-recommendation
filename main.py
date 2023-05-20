from flask import Flask, request, render_template
import pickle
from flask import jsonify
from fastapi import FastAPI , Request
import uvicorn
from pydantic import BaseModel
from fastapi.responses import JSONResponse


app = FastAPI()

#importing pickle files
model = pickle.load(open('classifier.pkl','rb'))
ferti = pickle.load(open('fertilizer.pkl','rb'))

class Data(BaseModel):
    temp: int
    humi: int
    mois: int
    soil: int
    crop: int
    nitro: int
    pota: int
    phosp: int

@app.get('/')
def welcome():
    return "Sample hello call"

@app.post('/predict')
def predict(data1 : Data):
    data = dict(data1)
    temp = data['temp']
    humi = data['humi']
    mois = data['mois']
    soil = data['soil']
    crop = data['crop']
    nitro = data['nitro']
    pota = data['pota']
    phosp = data['phosp']
    input = [int(temp),int(humi),int(mois),int(soil),int(crop),int(nitro),int(pota),int(phosp)]
    # temp = request.form.get('temp')
    # humi = request.form.get('humid')
    # mois = request.form.get('mois')
    # soil = request.form.get('soil')
    # crop = request.form.get('crop')
    # nitro = request.form.get('nitro')
    # pota = request.form.get('pota')
    # phosp = request.form.get('phosp')
    # input = [int(temp),int(humi),int(mois),int(soil),int(crop),int(nitro),int(pota),int(phosp)]

    res = ferti.classes_[model.predict([input])]

    return str(res[0])



@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )

if __name__ == "__main__":
    uvicorn.run(app)