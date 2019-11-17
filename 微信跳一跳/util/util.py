import cv2
import copy
import os
#将识别结果显示出来
def showResult(img,array):
    cv2.rectangle(img, (array[1], array[0]), (array[3], array[2]), (0, 0, 255), 5)

#显示中心轴
def showAxis(img):
    cv2.rectangle(img,(0,1920//2),(1080,1920//2),(255,255,0),2)
    cv2.rectangle(img, (1080//2, 0), (1080//2, 1920), (255, 255, 0), 2)


def showPoint(img,point):
    cv2.rectangle(img,(point),(point),(86, 30, 226),20)


def showTargetArea(img,findArea):
    cv2.rectangle(img,(findArea[0],findArea[1]),(findArea[2],findArea[3]),(255,0,0),2)
#显示图片
def showImage(img):
    image=copy.deepcopy(img)
    image = cv2.resize(image, (int(image.shape[1] // 2.5), int(image.shape[0] // 2.5)))
    cv2.imshow("IMAGE", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#屏幕截屏
def pull_screenshot():
    os.system("adb shell /system/bin/screencap -p /sdcard/screenshot.png")
    os.system("adb pull /sdcard/screenshot.png .")

