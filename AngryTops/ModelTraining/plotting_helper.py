# This script is used for making plots related to neurel network training
# BORROWED: From Tensorflow tutorials
import matplotlib.pyplot as plt

plt.rc('legend',fontsize=22)
plt.rcParams.update({'font.size': 22})

def plot_history(history, save_dir):
    """
    Plot training history
    """
    plt.figure(figsize=(16,10))

    # Plot just the los in a seperate plot
    val = plt.plot(history.epoch, history.history['val_loss'], '--', label=' Validation')
    plt.plot(history.epoch, history.history['loss'], color=val[0].get_color(), label=' Train')

    plt.xlabel('Epochs')
    plt.ylabel("Loss")
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.savefig("{}/Training.png".format(save_dir))
    plt.clf()

    # Plot all of the keys in history
    plt.figure(figsize=(16,10))
    for key in history.history.keys():
        plt.plot(history.epoch, history.history[key], label=key)
    plt.xlabel('Epochs')
    plt.legend()
    plt.xlim([0,max(history.epoch)])

    plt.savefig("{}/Training_AllKeys.png".format(save_dir))
