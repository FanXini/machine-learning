import numpy as np
import cv2
from 微信跳一跳.util import util
import math
import os
import time

scope=300
height=1920
width=1080
press_coefficient=1.475

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
        targetPath+=str(i)+".png"
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


def set_button_position():
    """
    将 swipe 设置为 `再来一局` 按钮的位置
    """
    global swipe_x1, swipe_y1, swipe_x2, swipe_y2
    left = int(width / 2)
    top = int(1584 * (height / 1920.0))
    left = int(np.random.uniform(left-50, left+50))
    top = int(np.random.uniform(top-10, top+10))    # 随机防 ban
    swipe_x1, swipe_y1, swipe_x2, swipe_y2 = left, top, left, top

def jump(distant):
    press_time = distance * press_coefficient
    press_time = max(press_time, 200)  # 设置 200ms 是最小的按压时间
    press_time = int(press_time)
    cmd = 'adb shell input swipe {x1} {y1} {x2} {y2} {duration}'.format(
        x1=swipe_x1,
        y1=swipe_y1,
        x2=swipe_x2,
        y2=swipe_y2,
        duration=press_time
    )
    print(cmd)
    os.system(cmd)
    return press_time

if __name__=="__main__":
    targetPath = 'object.png'
    imagePath = "screenshot.png"
    while True:
        #截图
        util.pull_screenshot()
        img=cv2.imread(imagePath)
        #显示中心轴
        util.showAxis(img)
        imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        tartet=cv2.imread(targetPath,0)
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
        targetCenter_w=0
        targetCenter_h=0
        if targetLocation !=None:
            targetLocation[0]+=findArea[1]
            targetLocation[1]+=findArea[0]
            targetLocation[2]+=findArea[1]
            targetLocation[3]+=findArea[0]
            targetCenter_h=int((targetLocation[0]+targetLocation[2])//2)
            targetCenter_w=int((targetLocation[1]+targetLocation[3])//2)
            util.showResult(img, targetLocation)
            print("目标块的位置{0}".format(targetLocation))
        else:
            targetCenter_w=target_w
            targetCenter_h=target_h
            print("没有没有找到目标块")
        #util.showImage(img)
        distance= math.sqrt((targetCenter_w - jumpCenter_w) ** 2 + (targetCenter_h - jumpCenter_h) ** 2)
        set_button_position()
        jump(distance)
        # 为了保证截图的时候应落稳了，多延迟一会儿，随机值防 ban
        time.sleep(np.random.uniform(0.9, 1.2))
