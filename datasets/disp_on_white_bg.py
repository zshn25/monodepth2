import numpy as np
import cv2

bg = np.zeros([640,192,3],dtype=np.uint8)
bg.fill(255)
cv2.imshow("white bg", bg)
cv2.waitKey(0)
cv2.destroyAllWindows()

def show_rects_on_white_bg(tracked_rects):
    bg = np.zeros([640,192,3],dtype=np.uint8)
    bg.fill(255)
    cv2.imshow("white bg", bg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    for rect in tracked_rects:
    
        xx = rect[0]
        yy = rect[1]
        wdht = rect[2]
        hght = rect[3]
        SmoothDist = rect[5]
        ID = rect[7]
        MissCount = rect[8]
        TTC = rect[13]
        VehDynamics  = rect[12] # coming or going
        strp = 'ID_' +  str(ID) +  ':' + str(SmoothDist)
        if TTC > 0:
            strp = strp + '>' + str(TTC)

        if (MissCount>=MAX_MISS_COUNT-1 and VehDynamics>=1 and SmoothDist <= 12): # not missed this rect
            if SmoothDist <= 6:
                cv2.rectangle(bg, (xx, yy), (wdht, hght), R, 1)
                cv2.putText(bg, strp, (xx, yy - 5), cv2.LINE_AA, 0.5, R , 2)
            if(SmoothDist > 6 ):
                cv2.rectangle(bg, (xx, yy), (wdht, hght), Y, 1)
                cv2.putText(bg, strp, (xx, yy - 5), cv2.LINE_AA, 0.5, Y , 2)
        else:
            cv2.rectangle(bg, (xx, yy), (wdht, hght), G, 1)
            cv2.putText(bg, strp, (xx, yy - 5), cv2.LINE_AA, 0.5, G , 2)