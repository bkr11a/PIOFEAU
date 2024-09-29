__author__ = "Brad Rice"
__version__ = 0.1

import os
import cv2

from utils.OpticalFlowUtils import FlowReader
from utils.OpticalFlowUtils import FlowVisualiser

from assets.ml.src.CustomLosses import EPE_Loss

if __name__ == "__main__":
    reader = FlowReader()
    visualiser = FlowVisualiser()
    EPELoss = EPE_Loss()

    SINTEL_PATH = os.path.join("..", "Data", "MPI-Sintel")
    SINTEL_TRAINING_PATH = os.path.join(SINTEL_PATH, "training")
    SINTEL_TRAINING_FLOW_PATH = os.path.join(SINTEL_TRAINING_PATH, "flow")
    SINTEL_TRAINING_IMG_PATH = os.path.join(SINTEL_TRAINING_PATH, "clean")

    testPath = os.path.join(SINTEL_TRAINING_FLOW_PATH, 'alley_1')
    testFlow = os.path.join(testPath, "frame_0001.flo")
    testImg = os.path.join(SINTEL_TRAINING_IMG_PATH, 'alley_1', 'frame_0001.png')

    flow = reader.readFlow(testFlow)
    img = cv2.imread(testImg)

    prev = img.copy()
    next = cv2.imread(os.path.join(SINTEL_TRAINING_IMG_PATH, 'alley_1', 'frame_0002.png'))

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)

    pcaFlow = cv2.optflow.createOptFlow_PCAFlow()
    flowPCA = pcaFlow.calc(prev, next, None)

    visualiser.visualiseOpticalFlow(flow=flow, image=None)
    visualiser.visualiseOpticalFlow(flow=flowPCA, image=None)

    loss = EPELoss(flow.reshape(1, flow.shape[0], flow.shape[1], flow.shape[2]), flowPCA.reshape(1, flowPCA.shape[0], flowPCA.shape[1], flowPCA.shape[2]))
    print(f"PCA EPE loss: {loss}")

    visualiser.visualiseFlowError(groundTruth=flow, predicted=flowPCA, image=None)