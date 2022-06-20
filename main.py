import uvicorn
from deepface import DeepFace
from fastapi import FastAPI, File, UploadFile
# from prediction import read_image , preprocess

from PIL import Image
from io import BytesIO
import numpy as np



app = FastAPI()

@app.get('/index')
def hellow_world(name: str):
    return f"Hello {name}!"



@app.get("/")
def home():
    return {"message":"Hello TutLinks.com"}

    
input_shape = (512, 512)

def read_image(file):
    pil_image = Image.open(BytesIO(file))
    return pil_image

def preprocess(image):
    image = np.array(image)
    return image




@app.post("/images")
async def images(img1: UploadFile  = File(...), img2: UploadFile  = File(...)):
    # **do something**
    imgRead1 = read_image(await img1.read())
    imgRead2 = read_image(await img2.read())
    preprocess1 = preprocess(imgRead1)
    preprocess2 = preprocess(imgRead2)
    # predictions= DeepFace.analyze(preprocess1)
    # predictions= DeepFace.analyze(preprocess2)
    # model_name = 'Facenet'
    model_name = 'VGG-Face'

    result= DeepFace.verify(img1_path = preprocess1, img2_path = preprocess2, model_name = model_name)

    return result




@app.post('/api/predict')
async def image_filter(file: UploadFile = File(...)):
  # img1_path = file.read()
        img = read_image(await file.read())
        img = preprocess(img)
        img1= DeepFace.detectFace(img)
        model_name = 'Facenet'
        # result= DeepFace.verify(img1_path = img, img2_path = img, model_name = model_name)
        model_name = 'Facenet'

        result= DeepFace.analyze(img1 , enforce_detection=False)      
        print(result)

if __name__ == '__main__':
    uvicorn.run(app, port = 8000, host = '127.0.0.1')
