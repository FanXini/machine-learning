import numpy as np
import cv2
from 微信跳一跳.util import util

def multi_scale_searchJump(pivot, screen, range=0.3, num=10):
    H, W = screen.shape[:2]
    h, w = pivot.shape[:2]

    found = None
    for scale in np.linspace(1-range, 1+range, num)[::-1]:
        resized = cv2.resize(screen, (int(W * scale), int(H * scale)))
        r = W / float(resized.shape[1])
        if resized.shape[0] < h or resized.shape[1] < w:
            break
        res = cv2.matchTemplate(resized, pivot, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= res.max())
        pos_h, pos_w = list(zip(*loc))[0]

        if found is None or res.max() > found[-1]:
            found = (pos_h, pos_w, r, res.max())

    if found is None: return (0,0,0,0,0)
    pos_h, pos_w, r, score = found
    start_h, start_w = int(pos_h * r), int(pos_w * r)
    end_h, end_w = int((pos_h + h) * r), int((pos_w + w) * r)
    return [start_h, start_w, end_h, end_w, score]


def multi_scale_searchTarget(screen, range=0.3, num=5):
    targetNum=0
    H, W = screen.shape[:2]
    targetPath="target1/"
    for i in np.arange(0,34):
        targetPath+=str(targetNum)+".png"
        pivot=cv2.imread(targetPath,0)
        h, w = pivot.shape[:2]

        found = None
        for scale in np.linspace(1-range, 1+range, num)[::-1]:
            resized = cv2.resize(screen, (int(W * scale), int(H * scale)))
            r = W / float(resized.shape[1])
            if resized.shape[0] < h or resized.shape[1] < w:
                break
            res = cv2.matchTemplate(resized, pivot, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= res.max())
            pos_h, pos_w = list(zip(*loc))[0]

            if found is None or res.max() > found[-1]:
                found = (pos_h, pos_w, r, res.max())

        if found is not None:
            break
    pos_h, pos_w, r, score = found
    start_h, start_w = int(pos_h * r), int(pos_w * r)
    end_h, end_w = int((pos_h + h) * r), int((pos_w + w) * r)
    return [start_h, start_w, end_h, end_w, score]


def showResult(img,array):
    cv2.rectangle(img, (array[1], array[0]), (array[3], array[2]), (7, 249, 151), 2)
    img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    cv2.imshow("IMAGE", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




if __name__=="__main__":
    num = 1
    targetPath = 'object.png'
    for i in range(1,115):
        imagePath = "Pictures1/"+str(num)+".png"
        num+=1
        img=cv2.imread(imagePath)
        print(img.shape)
        imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        tartet=cv2.imread('object.png',0)
        #start_h, start_w, end_h, end_w, score
        jumpLocation=multi_scale_searchJump(tartet,imgGray)
        #jumpImage=img[jumpLocation[0]:jumpLocation[2],jumpLocation[1]:jumpLocation[3]]
        #util.showImage(jumpImage)
        print(jumpLocation)
        jumpCenter_w=(jumpLocation[1]+jumpLocation[3])//2  #跳体Width
        jumpCenter_h=jumpLocation[2]-10  #跳体Height
        #cv2.rectangle(img, (jumpCenter_w, jumpCenter_h), (jumpCenter_w,jumpCenter_h ), (7, 249, 151), 10)
        util.showPoint(img,(jumpCenter_w,jumpCenter_h))
        target_w=1080-jumpCenter_w
        target_h=1920-jumpCenter_h
        util.showPoint(img,(target_w,target_h))
        #util.showAxis(img)
        util.showTargetArea(img,[target_w,target_h],300)
        find_area=img[target_h-300:target_h+300,target_w-300:target_w+300] #选择部分区域的图片。先h,在w
        #util.showImage(find_area)
        # cv2.imshow("target",find_area)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # targetLocat=multi_scale_searchTarget(find_area)
        util.showResult(img,jumpLocation)
        img = cv2.resize(img, (int(img.shape[1] // 2.5), int(img.shape[0] // 2.5)))
        util.showImage(img)