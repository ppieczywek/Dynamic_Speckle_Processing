from VideoProcessor import VideoProcessor
import pandas as pd
import wx

app = wx.App()
frame = wx.Frame(None, -1, 'RheoData.py')
frame.SetSize(0, 0, 200, 50)

# Create open file dialog
openFileDialog = wx.FileDialog(frame, "Open", "", "",
                               "Uncompressed video data files (*.avi)|*.avi",
                               wx.FD_OPEN | wx.FD_FILE_MUST_EXIST | wx.FD_MULTIPLE)

openFileDialog.ShowModal()
file_list = openFileDialog.GetPaths()
openFileDialog.Destroy()

frame_width = 640
frame_height = 480

roi_width = 64
roi_height = 64

center_x = [320, 160, 480, 160, 480]
center_y = [240, 120, 120, 360, 360]


# roi_top = 100
# roi_left = 100
results = []

for file_path in file_list:
    file_name = file_path.split('\\')[-1]
    print(f"Processing data from file: {file_name}")

    buffer = VideoProcessor.ReadVideoFile(file_path)
    buffer = VideoProcessor.RgbToGray(buffer)

    for idx in range(0, len(center_x)):
        roi_top = center_y[idx] - (roi_height // 2)
        roi_left = center_x[idx] - (roi_width // 2)
        crop_buffer = VideoProcessor.CropVideo(buffer, roi_top, roi_left, roi_width, roi_height)
        a, b = VideoProcessor.DecorrelationRate(crop_buffer, 30, preview=False)
        record = {'file': file_name,
                  'Tau': a,
                  'b': b}

        print(record)
        results.append(record)

if results:
    pd.DataFrame(results).to_csv('output.csv', index=False)
