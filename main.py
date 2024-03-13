from fastapi import FastAPI, BackgroundTasks
import cv2
import operator
from string import ascii_uppercase
from keras.models import model_from_json
import numpy as np

app = FastAPI()

class Application:
    def __init__(self):
        self.loaded_model = self.load_model("model_new.json", "model_new.h5")
        self.loaded_model_dru = self.load_model("model-bw_dru.json", "model-bw_dru.h5")
        self.loaded_model_tkdi = self.load_model("model-bw_tkdi.json", "model-bw_tkdi.h5")
        self.loaded_model_smn = self.load_model("model-bw_smn.json", "model-bw_smn.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        self.cword = ""

        for i in ascii_uppercase:
            self.ct[i] = 0
        
        print("Loaded models from disk")

        self.str = ""
        self.word = " "
        self.current_symbol = "Empty"
        self.photo = "Empty"
        self.resultword = ""
        
    def set_resultword(self, resultword):
        self.resultword = resultword

    def get_resultword(self):
        return self.resultword
    

    def load_model(self, json_file, weights_file):
        json_file = open(json_file, "r")
        model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(model_json)
        loaded_model.load_weights(weights_file)
        return loaded_model

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1))

        prediction = {'blank': result[0][0]}
        inde = 1
        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]

        if self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            if prediction1[0][0] == 'S':
                self.current_symbol = prediction1[0][0]
            else:
                self.current_symbol = prediction[0][0]
        
        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 40:
            self.cword += self.current_symbol
            self.ct['blank'] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

        return self.cword

application = Application()

def video_loop():
    vs = cv2.VideoCapture("video1.mp4")
    while True:
        ok, frame = vs.read()
        if ok:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            application.resultword = application.predict(res)
            print(application.resultword)
            if len(application.resultword) > 1:
                break
        else:
            print("No data found")
    vs.release()
    

@app.post("/video_feed")
async def process_video_feed(background_tasks: BackgroundTasks):
    background_tasks.add_task(video_loop)
    

@app.get("/result")
async def get_result():
    return {"resultword": application.get_resultword()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
