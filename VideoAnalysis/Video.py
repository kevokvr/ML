import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from imutils.object_detection import non_max_suppression

# Variables declaration
startrow = 0
frame = 0
g = False
timestamp = 'c'
indx_maxy = []
indx_maxx = []
data = pd.read_csv('000.dat', header=None)
column_names = ["frame", "timestamp", "x_accel", "y_accel", "z_accel"]
processed_data = pd.DataFrame(index=range(len(data)-startrow), columns=column_names)

# Looking for the rows that start with G
def G_row(t):
    gmatch = "[G]"
    return bool(re.search(gmatch, t))

# Iterating through the data file to get gps and sensor info
for row in range(startrow, len(data)):
    g = G_row(data.iloc[row, 0])
    if g == True:
        split = str(data.iloc[row]).split("\\t")
        timestamp = (split[1].split(" ")[1])
    else:
        processed_data.loc[[row-startrow], "frame"] = frame
        processed_data.loc[[row-startrow], "timestamp"] = timestamp
        split = str(data.iloc[row]).split("\\t")
        z_accel = split[3].split("\n")[0]
        processed_data.loc[[row - startrow], "x_accel"] = float(split[1])
        processed_data.loc[[row - startrow], "y_accel"] = float(split[2])
        processed_data.loc[[row - startrow], "z_accel"] = float(z_accel)
        frame = frame + 3
processed_data = processed_data.dropna(how="all")
column_names = ["delta_sum", "delta_x", "delta_y", "delta_z"]
delta = pd.DataFrame(index=range(1), columns=column_names)
previous_x = previous_y = previous_z = None

for index, row in processed_data.iterrows():
    x, y, z = float(row["x_accel"]), float(row["y_accel"]), float(row["z_accel"])
    if previous_x == None:
        previous_x, previous_y, previous_z = x, y, z
    else:
        delta_x, delta_y, delta_z = abs(x - previous_x), abs(y - previous_y), abs(z - previous_z)
        delta_sum = delta_x + delta_y + delta_z
        new_row = pd.DataFrame([[delta_sum, delta_x, delta_y, delta_z]], columns=column_names)
        delta = delta.append(new_row, ignore_index=True)
delta = delta.dropna(how="all")

crash_index = delta["delta_y"].idxmax()
frame_num = processed_data.iloc[crash_index].frame
print("Car Accident Details and Time:")
print(frame_num)
print(processed_data.iloc[crash_index ])

hog = cv.HOGDescriptor()
hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())

cap = cv.VideoCapture('0.MOV')

cap.set(1, frame_num)

status, frame = cap.read()

cv.imshow('Frame', frame)

cv.waitKey()

# This is what detects the pedestrian
(rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

# Drawing the bound
for(x, y, w, h) in rects:
    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Help from link https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/
# Implementing the bounded box outside of the pedestrian

rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

# For the final bound box
for (xA, yA, xB, yB) in pick:
    cv.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

# show the output images
cv.imshow("Frame", frame)
cv.waitKey(0)

cv.imwrite('frame_' + 'bounded.jpg', frame)
# Writing at least one same for the bounded image
#for i in range(0:15)
#   cv.imwrite('frame_'+ str(i) + 'bounded.jpg', frame)

cv.imshow('Bounded_Frame', frame)

