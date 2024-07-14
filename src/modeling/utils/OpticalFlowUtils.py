__author__ = "Brad Rice"
__version__ = 0.1

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

class FlowReader:
    def __init__(self):
        pass
    
    def readFlow(self, src_file):
        """Read optical flow stored in a .flo, .pfm, or .png file
        Args:
            src_file: Path to flow file
        Returns:
            flow: optical flow in [h, w, 2] format
        Refs:
            - Interpret bytes as packed binary data
            Per https://docs.python.org/3/library/struct.html#format-characters:
            format: f -> C Type: float, Python type: float, Standard size: 4
            format: d -> C Type: double, Python type: float, Standard size: 8
        Based on:
            - To read optical flow data from 16-bit PNG file:
            https://github.com/ClementPinard/FlowNetPytorch/blob/master/datasets/KITTI.py
            Written by Clément Pinard, Copyright (c) 2017 Clément Pinard
            MIT License
            - To read optical flow data from PFM file:
            https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/pfm.py
            Written by Ruoteng Li, Copyright (c) 2017 Ruoteng Li
            License Unknown
            - To read optical flow data from FLO file:
            https://github.com/daigo0927/PWC-Net_tf/blob/master/flow_utils.py
            Written by Daigo Hirooka, Copyright (c) 2018 Daigo Hirooka
            MIT License
        """
        TAG_FLOAT = 202021.25
        # Read in the entire file, if it exists
        assert(os.path.exists(src_file))

        if src_file.lower().endswith('.flo'):

            with open(src_file, 'rb') as f:

                # Parse .flo file header
                tag = float(np.fromfile(f, np.float32, count=1)[0])
                assert(tag == TAG_FLOAT)
                w = np.fromfile(f, np.int32, count=1)[0]
                h = np.fromfile(f, np.int32, count=1)[0]

                # Read in flow data and reshape it
                flow = np.fromfile(f, np.float32, count=h * w * 2)
                flow.resize((h, w, 2))

        elif src_file.lower().endswith('.png'):

            # Read in .png file
            flow_raw = cv2.imread(src_file, -1)

            # Convert from [H,W,1] 16bit to [H,W,2] float formet
            flow = flow_raw[:, :, 2:0:-1].astype(np.float32)
            flow = flow - 32768
            flow = flow / 64

            # Clip flow values
            flow[np.abs(flow) < 1e-10] = 1e-10

            # Remove invalid flow values
            invalid = (flow_raw[:, :, 0] == 0)
            flow[invalid, :] = 0

        elif src_file.lower().endswith('.pfm'):

            with open(src_file, 'rb') as f:

                # Parse .pfm file header
                tag = f.readline().rstrip().decode("utf-8")
                assert(tag == 'PF')
                dims = f.readline().rstrip().decode("utf-8")
                w, h = map(int, dims.split(' '))
                scale = float(f.readline().rstrip().decode("utf-8"))

                # Read in flow data and reshape it
                flow = np.fromfile(f, '<f') if scale < 0 else np.fromfile(f, '>f')
                flow = np.reshape(flow, (h, w, 3))[:, :, 0:2]
                flow = np.flipud(flow)
        else:
            raise IOError

        return flow

class FlowVisualiser:
    def __init__(self):
        pass

    def imshow(self, img, greyscale = False):
        image = img.copy()

        if greyscale == False:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        fig = plt.figure(figsize = (20, 12))
        plt.imshow(image)
        plt.axis(False)
        plt.show()

    def visualiseOpticalFlow(self, flow, image = None, polar = True, normalise = True, drawArrows = False, step=16, scale=1):
        def draw_flow_arrows(image, flow, step=16, scale=1):
            """
            Draw arrows on an image to visualize optical flow.

            Args:
            - image: Input image.
            - flow: Optical flow vectors.
            - step: Step size for drawing arrows (default is 16).
            - scale: Scale factor for arrow length (default is 1).

            Returns:
            - An image with arrows drawn.
            """
            img = image.copy()
            h, w = img.shape[:2]
            for y in range(0, h, step):
                for x in range(0, w, step):
                    fx, fy = flow[y, x]
                    x2 = int(x + fx * scale)
                    y2 = int(y + fy * scale)
                    cv2.arrowedLine(img, (x, y), (x2, y2), (40, 40, 40), 1)

            return img

        u = flow[:, :, 0]
        v = flow[:, :, 1]

        hsv = np.zeros(shape = (u.shape[0], u.shape[1], 3), dtype = np.uint8)

        hsv[:, :, 2] = 255
        if polar:
            mag, ang = cv2.cartToPolar(u, v)
            hsv[:, :, 0] = ang*180/np.pi/2
            if normalise:
                mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            hsv[:, :, 1] = mag

            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if drawArrows:
            bgr = draw_flow_arrows(image=bgr, flow=flow, step=step, scale=scale)

        if image is not None:
            alpha = 0.5
            beta = 1 - alpha
            gamma = 0.0

            if len(image.shape) < 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            bgr = cv2.addWeighted(image, alpha, bgr, beta, gamma)

        self.imshow(bgr)

        return bgr

    def visualiseFlowError(self, groundTruth, predicted, image = None):
        err = np.subtract(groundTruth, predicted)
        self.visualiseOpticalFlow(flow=err, image=image)