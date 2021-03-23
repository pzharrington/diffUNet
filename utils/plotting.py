import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def generate_images(test_input, prediction, target, diff_precision=1E-3):
    """Visualize data"""
    sz = test_input.shape[-1]
    cmin, cmax = test_input.min(), target.max()
    xx, yy = np.meshgrid(np.linspace(-1., 1., num=sz), np.linspace(-1., 1., num=sz))

    fig = plt.figure(figsize=(9,9))
    plt.subplot(2,2,1)
    plt.pcolormesh(xx, yy, test_input[0,:,:], norm=Normalize(cmin, cmax), shading='gouraud')
    plt.title('input')
    plt.colorbar()

    plt.subplot(2,2,2)
    plt.pcolormesh(xx, yy, target[0,:,:], norm=Normalize(cmin, cmax), shading='gouraud')
    plt.title('target')
    plt.colorbar()

    plt.subplot(2,2,3)
    plt.pcolormesh(xx, yy, prediction[0,:,:], norm=Normalize(cmin, cmax), shading='gouraud')
    plt.title('prediction')
    plt.colorbar()
    
    diff = (prediction - target)/(target + diff_precision)
    cmax = np.abs(diff).max()
    plt.subplot(2,2,4)
    plt.pcolormesh(xx, yy, diff[0,:,:], norm=Normalize(-cmax, cmax), cmap='seismic', shading='gouraud')
    plt.title('residual')
    plt.colorbar()
    return fig


def PDF_and_metrics(vals, gens, params):
    # Plot PDF
    val_hist, bin_edges = np.histogram(vals, bins=50)
    gen_hist, _ = np.histogram(gens, bins=bin_edges)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig = plt.figure()
    plt.errorbar(centers, val_hist, yerr=np.sqrt(val_hist), fmt='ks--', label='target')
    plt.errorbar(centers, gen_hist, yerr=np.sqrt(gen_hist), fmt='ro', label='generated')
    plt.xlabel('Phi')
    plt.ylabel('Counts')
    plt.yscale('log')
    plt.legend()
    sqdiff = np.power(val_hist - gen_hist, 2.0)
    val_hist[val_hist<=0.] = 1.

    # other metrics
    b, s = params.bias_scale
    p = gens/s - b
    q = vals/s - b
    KLdiv = np.sum(p*np.log(p/q), axis=(2,3))
    return (fig, np.mean(np.divide(sqdiff, val_hist)), KLdiv.mean())



