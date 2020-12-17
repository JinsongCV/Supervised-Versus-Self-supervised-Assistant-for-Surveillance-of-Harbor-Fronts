##################################################
### Real time data plotter in opencv (python)  ###
## Plot integer data for debugging and analysis ##
## Contributors - Vinay @ www.connect.vin       ##
## For more details, check www.github.com/2vin  ##
##################################################

import cv2
import numpy as np

# Plot values in opencv program
class Plotter:
    def __init__(self, plot_width, plot_height):
        self.width = plot_width
        self.height = plot_height
        self.color = (255, 0 ,0)
        self.val = []
        self.plot_canvas = np.ones((self.height, self.width, 3),dtype=np.uint8)*255

    # Update new values in plot
    def plot(self, val):
        self.val.append(int(val))
        while len(self.val) > self.width:
            self.val.pop(0)
        self.plot_canvas = np.ones((self.height, self.width, 3),dtype=np.uint8)*255
        cv2.line(self.plot_canvas, (0, self.height-1), (self.width, self.height-1), (0,255,0), 1)
        cv2.line(self.plot_canvas, (0, self.height-100), (self.width, self.height-100), (0,0,255), 1)
        for i in range(len(self.val)-1):
            cv2.line(self.plot_canvas, (i, self.height - self.val[i]), (i+1, self.height - self.val[i+1]), self.color, 1)

if __name__ == "__main__":


    losses = np.load('/home/markpp/github/thermalautoencoder/output/202002172020 - 36of64/view1/crop0_losses.npy')

    # Create a plotter class object
    plot = Plotter(losses.shape[0], 500)

    # Create dummy values using for loop
    for l in losses:
        l = int(l *10000)

        # call 'plot' method for realtime plot
        plot.plot(l)

        cv2.imshow("plot", plot.plot_canvas)
        key = cv2.waitKey(30)
        if key == 27:
            break
