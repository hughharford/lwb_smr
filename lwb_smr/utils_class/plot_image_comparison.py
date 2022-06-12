import matplotlib.pyplot as plt

def plot_comparison(X_true, y_true, y_pred, a=0, b=1, c=2, title=''):

  f, axs = plt.subplots(3, 3, figsize=(12, 12))
  f.suptitle(title)

  ax1, ax2, ax3 = axs[0,0], axs[0,1], axs[0,2]
  ax4, ax5, ax6 = axs[1,0], axs[1,1], axs[1,2]
  ax7, ax8, ax9 = axs[2,0], axs[2,1], axs[2,2]

  ax1.imshow(X_true[a,:,:,:])
  ax1.set_title('Test Image')
  ax2.imshow(y_true[a,:,:,0], cmap='gray')
  ax2.set_title('Test Label')
  ax3.imshow(y_pred[a,:,:,0],cmap='gray')
  ax3.set_title('Prediction Label')

  ax4.imshow(X_true[b,:,:,:])
  ax5.imshow(y_true[b,:,:,0],cmap='gray')
  ax6.imshow(y_pred[b,:,:,0], cmap='gray')

  ax7.imshow(X_true[c,:,:,:])
  ax8.imshow(y_true[c,:,:,0], cmap='gray')
  ax9.imshow(y_pred[c,:,:,0], cmap='gray')
