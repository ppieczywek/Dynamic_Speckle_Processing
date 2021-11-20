import numpy as np
from scipy.optimize import curve_fit
from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
import cv2


class VideoProcessor():
    """Camera device class"""

    @staticmethod
    def SaveToFile(file_name, buffer, fps):

        w, h = buffer[0].shape
        video_writer = cv2.VideoWriter(file_name, cv2.VideoWriter_fourcc(*'RGBA'), fps, (w, h), isColor=False)

        for frame in buffer:
            video_writer.write(frame)

        video_writer.release()

    @staticmethod
    def ReadVideoFile(file_name):
        cap = cv2.VideoCapture(file_name)
        if cap.isOpened() is False:
            print("Error opening video stream or file")

        buffer = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret is True:
                buffer.append(frame.copy())
            else:
                break
        cap.release()

        return buffer

    @staticmethod
    def SpeckleContrast(buffer):

        captured_frames = len(buffer)
        roi_shape = buffer[0].shape
        # print(roi_shape)
        buffer_array = np.zeros((roi_shape[0]*roi_shape[1], captured_frames), dtype=int, order='C')

        for i in range(0, captured_frames):
            if len(roi_shape) > 2:
                if roi_shape[2] == 3:
                    gray = cv2.cvtColor(buffer[i], cv2.COLOR_BGR2GRAY)
                    buffer_array[:, i] = gray.flatten(order='C').astype(int)
                else:
                    buffer_array[:, i] = buffer[i].flatten(order='C').astype(int)
            else:
                buffer_array[:, i] = buffer[i].flatten(order='C').astype(int)

        return buffer_array.std() / buffer_array.mean()

    @staticmethod
    def InertiaMoment(buffer):

        captured_frames = len(buffer)
        roi_shape = buffer[0].shape
        # print(roi_shape)
        buffer_array = np.zeros((roi_shape[0]*roi_shape[1], captured_frames), dtype=int, order='C')

        for i in range(0, captured_frames):
            if len(roi_shape) > 2:
                if roi_shape[2] == 3:
                    gray = cv2.cvtColor(buffer[i], cv2.COLOR_BGR2GRAY)
                    buffer_array[:, i] = gray.flatten(order='C').astype(int)
                else:
                    buffer_array[:, i] = buffer[i].flatten(order='C').astype(int)
            else:
                buffer_array[:, i] = buffer[i].flatten(order='C').astype(int)

        buffer_array[buffer_array < 5] = 0

        result = greycomatrix(buffer_array, [1], [0], normed=True, levels=256, symmetric=True)
        com = result[:, :, 0, 0]
        distance_matrix = np.tile(np.arange(256), (256, 1))
        distance_matrix = np.abs(distance_matrix - distance_matrix.T) * (np.sqrt(2) / 2)
        IM = np.sum(com*distance_matrix)

        return IM

    @staticmethod
    def CropVideo(buffer, roi_top, roi_left, roi_width, roi_height):

        crop_buffer = []
        for frame in buffer:
            ROI = frame[roi_top:(roi_top + roi_height),
                        roi_left:(roi_left + roi_width)]
            crop_buffer.append(ROI.copy())

        return crop_buffer

    @staticmethod
    def EqualizeVideo(buffer, reference_value):

        video_avg_int = 0
        for frame in buffer:
            video_avg_int += frame.mean()

        video_avg_int /= len(buffer)

        correction = reference_value / video_avg_int
        equalized_buffer = []

        for frame in buffer:
            equalized_buffer.append(frame * correction)

        return equalized_buffer

    @staticmethod
    def RgbToGray(buffer):

        roi_shape = buffer[0].shape
        buffer_array = []

        for frame in buffer:
            if roi_shape[2] == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                buffer_array.append(gray.copy())
            else:
                buffer_array.append(frame.copy())

        return buffer_array

    @ staticmethod
    def DecorrelationRate(buffer, framerate, preview=False):

        captured_frames = len(buffer)
        roi_shape = buffer[0].shape

        max_lag = captured_frames // 3
        samples = 0
        correlations = np.zeros(max_lag)
        time_scale = np.zeros(max_lag)

        buffer_array = np.zeros((roi_shape[0]*roi_shape[1], captured_frames), dtype=float, order='C')

        for i in range(0, captured_frames):
            if len(roi_shape) > 2:
                if roi_shape[2] == 3:
                    gray = cv2.cvtColor(buffer[i], cv2.COLOR_BGR2GRAY)
                    buffer_array[:, i] = gray.flatten(order='C').astype(int)
                else:
                    buffer_array[:, i] = buffer[i].flatten(order='C')
            else:
                buffer_array[:, i] = buffer[i].flatten(order='C')

        for i in range(0, captured_frames-max_lag, 4):
            samples = samples + 1
            for lag in range(0, max_lag):
                out = np.corrcoef(buffer_array[:, i], buffer_array[:, i + lag])
                correlations[lag] = correlations[lag] + out.item((0, 1))

        for lag in range(0, max_lag):
            time_scale[lag] = (1/framerate)*lag

        correlations = correlations / samples
        def fitfunc(x, a, b): return np.exp(-1*((x/a)**b))

        popt, pcov = curve_fit(fitfunc,
                               xdata=time_scale,
                               ydata=correlations,
                               maxfev=6000,
                               p0=[1, 1],
                               bounds=((0.001, 0.001), (100, 100)))

        a, b = popt

        if preview is True:
            plt.plot(time_scale, correlations, 'o', label="experimental")
            plt.plot(time_scale, fitfunc(time_scale, a, b), '-', label='model')
            plt.xlabel("lag time [s]")
            plt.ylabel("correlation")
            plt.legend()
            plt.show()

        return a, b
