# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 17:49:20 2024

@author: 82106
"""

import math
import time
import numpy as np
from typing import List, Tuple


import matplotlib.pyplot as plt

class BoundingBox:
    def __init__(self):
        self.minX = float('inf')
        self.maxX = float('-inf')
        self.minY = float('inf')
        self.maxY = float('-inf')

    def update(self, p: Tuple[float, float]) -> None:
        self.minX = min(self.minX, p[0])
        self.minY = min(self.minY, p[1])
        self.maxX = max(self.maxX, p[0])
        self.maxY = max(self.maxY, p[1])

class FASTCOVER_PP:
    def __init__(self, P: List[Tuple[float, float]], diskCenters: List[Tuple[float, float]], r):
        self.P = P
        self.diskCenters = diskCenters
        self.r = r
        self.sqrt2 = math.sqrt(2)*self.r
        self.additiveFactor = self.sqrt2 / 2
        self.sqrt2TimesOnePointFiveMinusOne = (self.sqrt2 * 1.5) - self.r
        self.sqrt2TimesZeroPointFivePlusOne = (self.sqrt2 * 0.5) - self.r

    def execute(self) -> float:
        assert len(self.P) > 0
        start = time.time()

        H = {}

        for p in self.P:
            v = math.floor(p[0] / self.sqrt2)
            h = math.floor(p[1] / self.sqrt2)
            
            verticalTimesSqrtTwo = v * self.sqrt2
            horizontalTimesSqrt2 = h * self.sqrt2

            if (v, h) in H:
                H[(v, h)][0].update(p)
                continue

            if (p[0] >= verticalTimesSqrtTwo + self.sqrt2TimesOnePointFiveMinusOne):
                if (v + 1, h) in H and (p[0] - (v + 1) * self.sqrt2 + self.additiveFactor) ** 2 + (p[1] - h * self.sqrt2 + self.additiveFactor) ** 2 <= self.r ** 2:
                    H[(v + 1, h)][0].update(p)
                    continue

            if (p[0] <= verticalTimesSqrtTwo - self.sqrt2TimesZeroPointFivePlusOne):
                if (v - 1, h) in H and (p[0] - (v - 1) * self.sqrt2 + self.additiveFactor) ** 2 + (p[1] - h * self.sqrt2 + self.additiveFactor) ** 2 <= self.r ** 2:
                    H[(v - 1, h)][0].update(p)
                    continue

            if (p[1] <= horizontalTimesSqrt2 + self.sqrt2TimesOnePointFiveMinusOne):
                if (v, h - 1) in H and (p[0] - v * self.sqrt2 + self.additiveFactor) ** 2 + (p[1] - (h - 1) * self.sqrt2 + self.additiveFactor) ** 2 <= self.r ** 2:
                    H[(v, h - 1)][0].update(p)
                    continue

            if (p[1] >= horizontalTimesSqrt2 - self.sqrt2TimesZeroPointFivePlusOne):
                if (v, h + 1) in H and (p[0] - v * self.sqrt2 + self.additiveFactor) ** 2 + (p[1] - (h + 1) * self.sqrt2 + self.additiveFactor) ** 2 <= self.r ** 2:
                    H[(v, h + 1)][0].update(p)
                    continue

            H[(v, h)] = [BoundingBox(), True]

        for aPair, value in H.items():
            v, h = aPair
            
            if not value[1]:
                continue

            if self.trytoMergeDisk(H, aPair, v, h - 1):
                continue

            if self.trytoMergeDisk(H, aPair, v, h + 1):
                continue

            if self.trytoMergeDisk(H, aPair, v + 1, 1):
                continue

            if self.trytoMergeDisk(H, aPair, v - 1, 1):
                continue

            if self.trytoMergeDisk(H, aPair, v - 1, h - 1):
                continue

            if self.trytoMergeDisk(H, aPair, v + 1, h - 1):
                continue

            if self.trytoMergeDisk(H, aPair, v + 1, h + 1):
                continue

            if self.trytoMergeDisk(H, aPair, v - 1, h + 1):
                continue

        for aPair, value in H.items():
            if value[1]:
                self.diskCenters.append((aPair[0] * self.sqrt2 + self.additiveFactor, aPair[1] * self.sqrt2 + self.additiveFactor))

        stop = time.time()
        return stop - start

    def trytoMergeDisk(self, H, iterToSourceDisk, vPrime, hPrime):
        iterToTargetDisk = H.get((vPrime, hPrime))
    
        if iterToTargetDisk is None or not isinstance(iterToTargetDisk, tuple) or len(iterToTargetDisk) != 2:
            return False
    
        if iterToTargetDisk[1]:
            minX = min(iterToSourceDisk[1][0].minX, iterToTargetDisk[0].minX)
            minY = min(iterToSourceDisk[1][0].minY, iterToTargetDisk[0].minY)
            maxX = max(iterToSourceDisk[1][0].maxX, iterToTargetDisk[0].maxX)
            maxY = max(iterToSourceDisk[1][0].maxY, iterToTargetDisk[0].maxY)

            lowerLeft = (minX, minY)
            upperRight = (maxX, maxY)
            print(lowerLeft, upperRight)
            if (lowerLeft[0] - upperRight[0]) ** 2 + (lowerLeft[1] - upperRight[1]) ** 2 <= (2 * self.r) ** 2:
                iterToSourceDisk[1][1] = False
                iterToTargetDisk[1] = False
                self.diskCenters.append(((lowerLeft[0] + upperRight[0]) / 2, (lowerLeft[1] + upperRight[1]) / 2))
                return True
        return False

def main():

    #P = [(1, 2), (3, 4), (5, 6), (7, 8)]  # Your list of points
    #points = np.array([[0.6,1.2],[0.7,1.3],[0.95,0.45],[1.6,0.6],[1.6,1.6],[1.0,0.75],[0.75,0.9],[1.3,0.7],[1.48,0.98],[1.3,1.2],[1.5,1.55],[1.5,0.45],[0.46,0.46],[2.1,1.03],[0.46,1.55],[-0.1,1.03],[0.38,1.6]])

    #points = np.array([[1.32,1.32],[1.62,1.2],[1.52,2.84],[2.03,2.01],[1.37,2.91],[1.40, 2.82],[2.83,2.1]])
    #file_path = r"3038.txt"  #1291,1889,2319, 3038
    #points = np.loadtxt(file_path, delimiter = " ")
    
    #np.random.seed(6)
    points = np.random.rand(10, 2) * 5
    r = 1
    times = []
    for i in range(1):
        diskCenters = []  # List to store disk centers
        fast_cover = FASTCOVER_PP(points, diskCenters, r)
        execution_time = fast_cover.execute()
        times.append(execution_time)
    #print("Disk centers:", diskCenters)
    print("Execution time:", np.mean(times))
    print(len(diskCenters))
 
    # 分离不同圆的数据点
    points_in_circles = [[] for j in range(len(diskCenters))]
    for point in points:
        for i, center in enumerate(diskCenters):
            if (point[0] - center[0])**2 + (point[1] - center[1])**2 <= r**2:  # 这里的1表示圆的半径
                points_in_circles[i].append(point)
                
    ax = plt.subplot(111)
    for a, samples in enumerate(points_in_circles):
        x_raw = []
        y_raw = []
        #print(a)
        for sample in samples:
            x_raw.append(sample[0])
            y_raw.append(sample[1])
        plt.scatter(x_raw,y_raw, s=2)
        cir1 = plt.Circle((diskCenters[a][0], diskCenters[a][1]), r, color="black",fill=False)
        #plt.plot(diskCenters[a][0], diskCenters[a][1], 'k+', markersize=5)
        ax.add_patch(cir1)

    plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0), useMathText=True)
    ax.set_aspect(0.84*abs(7000)/abs(6000))
    plt.show()
 

if __name__ == "__main__":
    
    main()
