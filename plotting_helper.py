# This script is used for making plots related to neurel network training
# BORROWED: From Tensorflow tutorials
import matplotlib.pyplot as plt

def plot_history(history, save_dir):
  plt.figure(figsize=(16,10))

  val = plt.plot(history.epoch, history.history['val_loss'], '--', label=' Validation')
  plt.plot(history.epoch, history.history['loss'], color=val[0].get_color(), label=' Train')

  plt.xlabel('Epochs')
  plt.ylabel("Loss")
  plt.legend()

  plt.xlim([0,max(history.epoch)])
  plt.savefig("{}/Training.png".format(save_dir))
