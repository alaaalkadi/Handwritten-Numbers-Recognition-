import numpy as np
import cv2
import pickle

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

#Parametars:

width = 640
highet = 480
threshold = 0.65

cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,highet)


pickle_in = open("model_trained_10.p","rb")
model = pickle.load(pickle_in)



def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return (img)

while True:
    success, imgOrginal = cap.read()
    img = np.asarray(imgOrginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("preProcessed Images",img)
    img = img.reshape(1,32,32,1)

    #Predict:
    classIndex = int (model.predict_classes(img))
   # print (classIndex)
    predictions = model.predict(img)
    #print (predictions)
    probVal = np.amax(predictions)
    print(classIndex,probVal)

    if probVal > threshold:
        cv2.putText(imgOrginal,str(classIndex) + "  "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

        data = pd.DataFrame()
        dataToExcel = pd.ExcelWriter("FromPython.xlsx", engine='openpyxl',mode="w")
        data.to_excel(dataToExcel, sheet_name='Sheet1')

        dataToExcel.save()
    cv2.imshow("Orginal Image",imgOrginal)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

