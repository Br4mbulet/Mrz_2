from net import net
from imagesManage import manager
import numpy as np
from PIL import Image
from loguru import logger
from sys import stderr

logger.remove()
logger.add(
    stderr, format="<level>{level: <8}</level> | <cyan>{line}</cyan> | <white>{message}</white>")

if __name__ == '__main__':

    imagesSize = 100

    logger.info("Starting loading images from file...")
    nw = net(manager.getImages())
    logger.success(f"Successfully loaded 3 images\n")

    logger.info("Starting learning images...")
    nw.learnImages()
    logger.success(f"Successfully learned 3 images\n")

    origImages = nw.getImages()
    choice = input('Show images y/n: ')
    if choice == 'y':
        for img in range(len(origImages)):
            print(f"{np.array(origImages[img].reshape(10, 10))}\n")
            #print(f"Image №{img} \n{origImages[img]}\n")

    result = []
    distortPercent = int(input("Enter percent of distortion: "))
    for i in range(len(origImages)):
        logger.info(f"Distort image №{i+1}...")
        distortedImage = manager.distortImage(origImages[i], distortPercent)
        logger.success(f"Successfully distort image\n")
        print(f"Distorted image №{i+1}\n{np.array(distortedImage.reshape(10, 10))}\n")

        logger.info(f"Identify image...")
        answer, iteration = nw.identify(distortedImage, None)
        logger.success(
            f"Successfully identify image №{answer+1} in {iteration} iterations\n")

        # Save original image
        origImg = []
        for pxel in origImages[i]:
            if (pxel == -1):
                origImg.append([0, 0, 0])
            if (pxel == 0):
                origImg.append([0, 124, 124])
            if (pxel == 1):
                origImg.append([0, 255, 255])
        origImg = np.uint8(origImg)
        origImg = Image.fromarray(np.array(origImg).reshape(10, 10, 3)).resize(
            (imagesSize, imagesSize), Image.ANTIALIAS).save(f'img\\{i+1}orig.png')

        # Save distorted image
        distortedImg = []
        for pxel in distortedImage:
            match pxel:
                case -1: distortedImg.append([0, 0, 0])
                case 0: distortedImg.append([0, 124, 124])
                case 1: distortedImg.append([0, 255, 255])
        distortedImg = np.uint8(distortedImg)
        distortedImg = Image.fromarray(np.array(distortedImg).reshape(10, 10, 3)).resize(
            (imagesSize, imagesSize), Image.ANTIALIAS).save(f'img\\{i+1}distorted.png')

        if i == answer:
            result.append(True)

    logger.success(f"Total identifyed images: {np.sum(result)}")
