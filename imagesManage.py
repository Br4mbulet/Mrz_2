import numpy as np
import json

class manager:
    def getImages():
        with open('img.json', 'r', encoding='utf-8') as f:
            imgs = json.load(f) 
        temp = np.array(imgs)                                         
        return temp


    def distortImage(img, distortPercent):
        toDistort = img.copy()
        rands = np.random.choice(toDistort.size, size=int(distortPercent * toDistort.size / 100), replace=False)
        for rand in rands:
            temp = np.random.randint(-1, 1)
            while temp == toDistort[rand]:
                temp = np.random.randint(-1, 1)
            toDistort[rand] = temp
        return toDistort


