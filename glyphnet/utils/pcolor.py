"""
pcolor: for plotting pcolor using matplotlib
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import time

output_directory = './generated'
os.makedirs(output_directory, exist_ok=True)

class PColor:
    @staticmethod
    def plot_show_image(G_paintings2d, file_id, pa0, Dl):
        plt.clf()

        #print(np.max(np.max(G_paintings2d,axis=2), axis=0))
        #print(np.min(np.min(G_paintings2d,axis=2), axis=1))
        #print(G_paintings2d.shape)
        #plt.imshow(G_paintings2d)
        #plt.imshow((G_paintings2d * 0.2 + 0.5)*0.2)
        #img_pix_rescale = (G_paintings2d * 0.05 + 0.5)
        #img_pix_rescale = (G_paintings2d)
        #plt.imshow(img_pix_rescale, vmin=-100, vmax=100)
        img_pix_rescale = ((G_paintings2d) / 80.0 *40  ) +0.5
        plt.imshow((img_pix_rescale *128).astype(np.uint8))
        print('min max:', np.min(img_pix_rescale.ravel()), np.max(img_pix_rescale.ravel()))
        #plt.pcolor(np.mean(G_paintings2d, axis=2))
        print("@*")
        plt.text(-.5, 0, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, G_paintings2d.shape[1]*0.5, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
        plt.colorbar()



        plt.draw()
        time.sleep(0.1)

        plt.savefig( os.path.join(output_directory, 'foo-' + file_id + '.png'))

        print("saved")

        time.sleep(0.1)

    @staticmethod
    def init():
        plt.cla()
        #plt.imshow(main_artworks[0])
        plt.draw()
        #plt.ioff()
        plt.ion()
        plt.show()
        time.sleep(0.1)
        plt.ion()   # something about continuous plotting

    @staticmethod
    def last(self):
        plt.ioff()
        plt.show()
