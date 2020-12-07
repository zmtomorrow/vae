import matplotlib.pyplot as plt
import numpy as np

def show_many(image,number_sqrt):
    
    canvas_recon = np.empty((28 * number_sqrt, 28 * number_sqrt))
    count=0
    for i in range(number_sqrt):
        for j in range(number_sqrt):
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
            image[count].reshape([28, 28])
            count+=1
    plt.rcParams["axes.grid"] = False
    plt.figure(figsize=(number_sqrt, number_sqrt))
#     plt.axis('off')
    plt.imshow(canvas_recon, origin="upper", cmap="gray")
    plt.show()