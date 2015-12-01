from PIL import Image
from numpy import *
from math import sqrt
import sys

if (len(sys.argv) < 2):
    print("Pass the path to input file as an argument")
    exit()
input_image_file = sys.argv[1]
output_image_file = 'output.png'

image = Image.open(input_image_file)
imageWidth, imageHeight = image.size
imageArray = array(image)

grayScaleImage = array(image.convert('L'))

def imageGradient(image, isHorizontal):
    grad = zeros((imageHeight, imageWidth))

    for h in range(1, imageHeight - 1):
        for w in range(1, imageWidth - 1):
            if isHorizontal:
              grad[h,w] = float(int(image[h-1,w]) - int(image[h+1,w])) / 255
            else:
              grad[h,w] = float(int(image[h,w-1]) - int(image[h,w+1])) / 255

    return grad

dx = imageGradient(grayScaleImage, True)
dy = imageGradient(grayScaleImage, False)

fragmentWidth = 5
fragmentHeight = 5
k = 0.05
z = 1e-2
clusterRadius = 15

gaussianCore = [[ 2,  4,  5,  4, 2],
                [ 4,  9, 12,  9, 4],
                [ 5, 12, 15, 12, 5],
                [ 4,  9, 12,  9, 4],
                [ 2,  4,  5,  4, 2]]

gaussianCoreNormalized = [[float(x) / 159 for x in list] for list in gaussianCore]

def getStructureTensor(u, v):
    mA = 0
    mB = 0
    mC = 0
    mD = 0

    for h in range(0, fragmentHeight):
        for w in range(0, fragmentWidth):
            g = gaussianCoreNormalized[h][w]
            dxi = dx[u + h - fragmentHeight / 2, v + w - fragmentWidth / 2]
            dyi = dy[u + h - fragmentHeight / 2, v + w - fragmentWidth / 2]
            mA += g * dxi * dxi
            mB += g * dxi * dyi
            mC += g * dxi * dyi
            mD += g * dyi * dyi

    return [[mA, mB], [mC, mD]]

def cornerMeasure(u, v):
    [[a, b], [c, d]] = getStructureTensor(u, v)
    det = a * d - b * c
    trace = a + d
    return det - k * (trace ** 2);

def harris():
    corners = []
    for h in range(fragmentHeight / 2, imageHeight - fragmentHeight / 2):
        for w in range(fragmentWidth / 2, imageWidth - fragmentWidth / 2):
            if (cornerMeasure(h, w) > z):
                corners.append((h, w))
    return corners

def distance((x, y), (u, v)):
    d = sqrt((x - u) ** 2 + (y - v) ** 2)
    return d

def save():
    resultImage = Image.fromarray(imageArray).save(output_image_file)


def drawCornersOnImageWithoutClusterization():
    corners = harris()

    for (h,w) in corners:
        imageArray[h,w] = [0,0,255]

    save()


def findCenterOf(cluster):
    firstPoint = cluster[0]
    minDist = 0

    for p in cluster:
        d = distance(firstPoint, p)
        minDist += d

    for point in cluster:
        dist = 0

        for p in cluster:
            d = distance(point, p)
            dist += d

        if dist < minDist:
            minDist = dist
            firstPoint = point

    return firstPoint

def drawClosePointsAsOne(points):
    while (points):
        cluster = []
        point = points[0]
        cluster.append(point)

        for p in points:
            d = distance(point, p)
            if d < clusterRadius:
                cluster.append(p)
                points.remove(p)

        center = findCenterOf(cluster)

        for i in range(center[0] - 1, center[0] + 1):
            for j in range(center[1] - 1, center[1] + 1):
                imageArray[i,j] = [255,0,0]

        save()

corners = harris()
drawClosePointsAsOne(corners)
#drawCornersOnImageWithoutClusterization()