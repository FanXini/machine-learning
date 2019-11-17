import numpy as np
import cv2
from 微信跳一跳.util import util

scope=300
height=1920
width=1080

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



def find_target(find_area,value):
    targetPath="target1/"
    for i in range(66):
        found=None
        targetPath+=str(i)+".png"  #迭代匹配图片库
        template = cv2.imread(targetPath)
        template_gray=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        h,w=template_gray.shape[0:2]
        res = cv2.matchTemplate(find_area, template_gray, cv2.TM_CCOEFF_NORMED)
        if res.max()>value:
            loc = np.where(res >= value)
            pos_h, pos_w = list(zip(*loc))[0]
            return [pos_h,pos_w,pos_h+h,pos_w+w,res.max()]
        targetPath="target1/"
    #没有找到目标块
    return None

#返回的是一个数组,数组存储的是搜索区域的范围[start_w,start_h,end_w,end_h]
def getFindArea(start_w,start_h,scope):
    findArea=[]
    if start_w-scope<0:
        findArea.append(5)
    else:findArea.append(start_w-scope)
    if start_h-scope<0:
        findArea.append(5)
    else:findArea.append(start_h-scope)
    if(start_w+scope>width):
        findArea.append(width-5)
    else:findArea.append(start_w+scope)
    if(start_h+scope>height):
        findArea.append(height-5)
    else:findArea.append(start_h+scope)
    return findArea



if __name__=="__main__":
    targetPath = 'object.png'
    for i in range(1,115):
        imagePath = "Pictures1/"+str(i)+".png"
        img=cv2.imread(imagePath)
        #显示中心轴
        util.showAxis(img)
        #创建一个原始图像的灰度版本，所有操作在灰度版本中处理，然后在RGB图像中使用相同坐标还原
        imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        tartet=cv2.imread('object.png',0)
        #start_h, start_w, end_h, end_w, score
        jumpLocation=multi_scale_searchJump(tartet,imgGray)
        print(jumpLocation)
        jumpCenter_w=(jumpLocation[1]+jumpLocation[3])//2  #跳体Width
        jumpCenter_h=jumpLocation[2]-10  #跳体Height
        #cv2.rectangle(img, (jumpCenter_w, jumpCenter_h), (jumpCenter_w,jumpCenter_h ), (7, 249, 151), 10)
        util.showPoint(img,(jumpCenter_w,jumpCenter_h))
        target_w=1080-jumpCenter_w
        target_h=1920-jumpCenter_h
        util.showPoint(img, (target_w, target_h))
        #getFindArea返回的是一个数组,数组存储的是搜索区域的范围[start_w,start_h,end_w,end_h]
        findArea=getFindArea(target_w,target_h,scope)
        util.showTargetArea(img,findArea)
        find_area=imgGray[findArea[1]:findArea[3],findArea[0]:findArea[2]] #选择部分区域的图片。先h,在w
        util.showResult(img,jumpLocation)
        #start_h,start_w,end_h,end_w,score
        targetLocation=find_target(find_area,0.9)
        if targetLocation !=None:
            targetLocation[0]+=findArea[1]
            targetLocation[1]+=findArea[0]
            targetLocation[2]+=findArea[1]
            targetLocation[3]+=findArea[0]
            util.showResult(img, targetLocation)
            print("第{0}张照片的位置{1}".format(i,targetLocation))
        else:print("第{0}张照片没有没有找到目标块".format(i))
        util.showImage(img)
