import numpy as np

class net:
    def __init__(self, toRememb):
        self.process = "proj"
        self.imges = toRememb
        self.Wi = np.zeros((toRememb[0].size, toRememb[0].size))

    def learnImages(self):
        if self.process == "proj":
            for img in self.imges:
                self.Wi += 1 / (img.reshape(1, img.size) @ img.reshape(img.size, 1) - (img.reshape(1, img.size) @ self.Wi @ img.reshape(img.size, 1))) * (self.Wi @ img.reshape(img.size, 1) - img.reshape(img.size, 1)) @ (self.Wi @ img.reshape(img.size, 1) - img.reshape(img.size, 1)).transpose()
        if self.process == "easy":
            for img in self.imges:
                tmp = img.reshape(img.size, 1) @ img.reshape(1, img.size)
                self.Wi += tmp
                for i in range(self.imges[0].size): self.Wi[i, i] = 0

    def identify(self, distorted, corr):
        previous = distorted
        iterator = 0
        while True:
            iterator =  iterator + 1
            diff = np.sum(np.abs(previous - distorted))
            distorted = self.Wi @ distorted.reshape(distorted.size, 1)
            distorted = np.tanh(distorted)
            distorted = distorted.flatten()
            if self.process == "proj":
                if (np.abs(np.abs(previous - distorted)) < 0.001).all(): return self.mIn(distorted), iterator
            if self.process == "easy":
                if (diff - np.sum(np.abs(previous - distorted)) < 0.001): return None, iterator
            previous = distorted

    def mIn(self, img):
        if self.process == "proj":
            img = np.where(img > 0, 1, -1)
            for i in range(self.imges.shape[0]):
                if (self.imges[i] == img).all():
                    return i
            return False
        if self.process == "easy":
            for i in range(self.imges.shape[0]):
                if (self.imges[i] == img).all():
                    return i
            return False

    def getImages(self):
        return self.imges
