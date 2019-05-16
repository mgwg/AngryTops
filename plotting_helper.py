# This script is used for making plots related to neurel network training
# BORROWED: From Tensorflow tutorials
import matplotlib.pyplot as plt

def plot_history(history, save_dir, key='mse'):
  plt.figure(figsize=(16,10))

  val = plt.plot(history.epoch, history.history['val_'+key],
                 '--', label=' Validation')
  plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
             label=' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.savefig("{}/Training".format(save_dir))
  plt.show()
