import cv2
import numpy as np

# Read image 
img = cv2.imread('1562350.jpeg', cv2.IMREAD_COLOR)
# Convert the image to gray-scale

imgGauss = cv2.GaussianBlur(img,(11,11),0)
Z = imgGauss.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0) #Max iter = 10, epsilon=1.0
K = 10
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

img2 = cv2.cvtColor(res2, cv2.COLOR_BGR2HSV)

lowG = (20,20,50)
highG = (80,200,200)

lowB = (10,0,0)
highB = (20,150,150)

mask1 = cv2.inRange(img2, lowG, highG)
mask2 = cv2.inRange(img2, lowB, highB)


masks = [np.uint8(mask1),np.uint8(mask2)]
edgs = []
imgC = np.zeros(img.shape)

for full_mask in masks:
    blur = cv2.medianBlur(full_mask,5)

    img3 = np.zeros(img.shape)
    contours, hierarchy = cv2.findContours(blur.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    
    valid_cntrs = []
    for cntr in contours:
        if cv2.contourArea(cntr) > 9000 and cv2.contourArea(cntr) < 200000:
            valid_cntrs.append(cntr)

    cv2.drawContours(image=img3, contours=valid_cntrs, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)

    img3 = np.uint8(img3)
    # Find the edges in the image using canny detector
    edges = cv2.Canny(img3,35,40,None,3)
    edgs.append(edges)

    # Detect points that form a line
    lines = cv2.HoughLinesP(edges,1, np.pi/180, 55,None, minLineLength=85, maxLineGap=25)
    # Draw lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 5)

# Show result 
cv2.imshow("Mask", mask1)#edgs[0])
cv2.imshow("Contours", imgC)
cv2.imshow("K-Means", res2)
cv2.imshow("Result Image", img)
cv2.waitKey(0)