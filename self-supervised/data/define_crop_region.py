#!/usr/bin/env python
import os
import cv2
import numpy as np
import argparse


class Mouse:
    def __init__(self):
        self.l, self.m, self.r = [[52*2,108*2],[112*2,270*2]], [[67*2,108*2],[170*2,265*2]], [None,None]
        self.m_count, self.l_count, self.r_count = 2, 2, 0

    # mouse callback function
    def get_click(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.l[self.l_count % 2] = [x,y]
            self.l_count += 1
        if event == cv2.EVENT_MBUTTONDOWN:
            self.m[self.m_count % 2] = [x,y]
            self.m_count += 1
        #if event == cv2.EVENT_RBUTTONDOWN:
        #    self.r[0],self.r[1] = x,y
        #    self.r_count += 1

def points2line(p1, p2):
    a = (p2[1] - p1[1])/(p2[0] - p1[0])
    b = p1[1] - a * p1[0]
    print("line parms: a {}, b {}".format(a,b))
    return (a, b)

# broken sometimes
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    #rect = order_points(pts)
    rect = np.zeros((4, 2), dtype = "float32")
    rect[0], rect[3] = pts[0], pts[3]
    rect[1], rect[2] = pts[1], pts[2]
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def label_frame(txt_file, frame, scale=1.0):
    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    while (1):
        tmp = frame.copy()
        height, width = tmp.shape[:2]

        for i, point in enumerate(mouse.l):
            labels = ["tl", "bl"]
            if point is not None:
                x, y = point
                tmp = cv2.circle(tmp, (x, y), 2, (0,0,255), 2)
                #tmp = cv2.putText(tmp, labels[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA, False)
        for i, point in enumerate(mouse.m):
            labels = ["tr", "br"]
            if point is not None:
                x, y = point
                tmp = cv2.circle(tmp, (x, y), 2, (0,0,255), 2)
                #tmp = cv2.putText(tmp, labels[i], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA, False)

        if mouse.l_count > 1 and mouse.m_count > 1:
            tmp = cv2.line(tmp, tuple(mouse.l[0]), tuple(mouse.m[0]), (0,0,255), 2)
            tmp = cv2.line(tmp, tuple(mouse.m[0]), tuple(mouse.m[1]), (0,0,255), 3)
            tmp = cv2.line(tmp, tuple(mouse.m[1]), tuple(mouse.l[1]), (0,0,255), 2)
            tmp = cv2.line(tmp, tuple(mouse.l[1]), tuple(mouse.l[0]), (0,0,255), 2)
            #print("mouse.l[0]: {}, mouse.l[1]: {}".format(mouse.l[0],mouse.l[1]))

            pts = np.array([mouse.l[0],mouse.m[0],mouse.m[1],mouse.l[1]])
            warped = four_point_transform(frame, pts)

            cv2.imshow("warped", warped)

        cv2.imshow("image", tmp)

        k = cv2.waitKey(100) & 0xFF
        #print k
        if k == 27:
            file = open(txt_file, "w")
            if mouse.l_count > 1 and mouse.m_count > 1:
                file.write("tl,tr,br,bl\n")
                x,y = mouse.l[0]
                file.write("{},{}\n".format(int(x//scale),int(y//scale)))
                x,y = mouse.m[0]
                file.write("{},{}\n".format(int(x//scale),int(y//scale)))
                x,y = mouse.m[1]
                file.write("{},{}\n".format(int(x//scale),int(y//scale)))
                x,y = mouse.l[1]
                file.write("{},{}".format(int(x//scale),int(y//scale)))
                #line_l = points2line([mouse.m[0]/width, mouse.m[1]/height],
                #                     [mouse.l[0]/width, mouse.l[1]/height])
                #line_r = points2line([mouse.m[0]/width, mouse.m[1]/height],
                #                     [mouse.r[0]/width, mouse.r[1]/height])
                #file.write("line_l:{:.6f},{:.6f};".format(line_l[0], line_l[1]))
                #file.write("line_r:{:.6f},{:.6f}\n".format(line_r[0], line_r[1]))
            file.close()
            break


if __name__ == "__main__":
     # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--frame", type=str,
                    default=None, help="Path to frame")
    ap.add_argument("-s", "--scale", type=float,
                    default=2.0, help="Path to frame")
    args = vars(ap.parse_args())

    # Create a window
    cv2.namedWindow('image')
    cv2.moveWindow('image', 200, 200)
    mouse = Mouse()
    cv2.setMouseCallback('image',mouse.get_click)

    if args["frame"]:
        txt_file = "roi.txt"
        frame = cv2.imread(args["frame"], -1)
        label_frame(txt_file, frame, scale=args["scale"])


    cv2.destroyAllWindows()
