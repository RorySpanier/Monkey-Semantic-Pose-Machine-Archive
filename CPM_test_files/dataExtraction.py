import json
import numpy as np
import pickle

def jsonLoadandLook():
    f = open("train_annotation.json")
    data = json.load(f)
    length = len(data["data"])

    print(data["data"][5])
    # print(data["data"][0]['landmarks'])


    f.close()

def saveLandmarks(file,savePath):
    f = open(file)
    data = json.load(f)
    length = len(data["data"])
    landmarkList = []
    for i in range(length):
        entryLandmarks = data["data"][i]["landmarks"]
        xList = entryLandmarks[0::2]
        yList = entryLandmarks[1::2]
        visList = data["data"][i]["visibility"]
        landmarkList.append([xList,yList,visList])
    landmarkList = np.transpose(landmarkList,(0,2,1))
    with open(savePath, 'wb') as f:
        pickle.dump(landmarkList, f)
    f.close()

if __name__ == "__main__":
    # jsonLoadandLook()

    saveLandmarks("train_annotation.json","train_landmarks.data")
    saveLandmarks("val_annotation.json","val_landmarks.data")

    # with open('val_landmarks.data', 'rb') as f:
    #     val_landmarks = pickle.load(f)
    # with open('train_landmarks.data', 'rb') as f:
    #     test_landmarks = pickle.load(f)
    # # print(len(val_landmarks))
    # # print(len(test_landmarks))
    # print(np.asarray(val_landmarks).shape)
    # print(np.asarray(test_landmarks).shape)
