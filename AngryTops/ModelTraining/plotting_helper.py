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
    plt.clf()

    # Plot the loss and mse together
    plt.plot(history.epoch, history.history['val_loss'], linestyle = 'solid', color = 'C0', label=' Validation Loss') #blue
    plt.plot(history.epoch, history.history['loss'], linestyle = 'dotted', color = 'C0', label=' Train Loss')

    plt.plot(history.epoch, history.history['val_mse'], linestyle = 'dashed', color = 'C8', label=' Validation MSE') #yellow
    plt.plot(history.epoch, history.history['mse'], linestyle = 'dashdot', color = 'C8', label=' Train MSE')

    plt.xlabel('Epochs')
    plt.ylabel('Loss and MSE')
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.savefig("{}/Training_LossMSE.png".format(save_dir))
    plt.clf()

    # Plot all keys with markers
    plt.plot(history.epoch, history.history['val_loss'], linestyle = 'solid', color = 'C0', label=' Validation Loss')
    plt.plot(history.epoch, history.history['loss'], marker = ".", color = 'C1', label=' Train Loss')

    plt.plot(history.epoch, history.history['val_mse'], marker = "+", color = 'C9', label=' Validation MSE')
    plt.plot(history.epoch, history.history['mse'], marker = "*", color = 'C8', label=' Train MSE')

    plt.plot(history.epoch, history.history['val_mae'], marker = "v", color = 'C4', label=' Validation MAE')
    plt.plot(history.epoch, history.history['mae'], marker = "s", color = 'C5', label=' Train MAE')

    plt.xlabel('Epochs')
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.savefig("{}/Training_AllKeysMarkers.png".format(save_dir))
    plt.clf()

    # Plot mae separately
    plt.plot(history.epoch, history.history['val_mae'], linestyle = '--', color = 'C0', label=' Validation MAE')
    plt.plot(history.epoch, history.history['mae'], color = 'C0', label=' Train MAE')


    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.xlim([0,max(history.epoch)])
    plt.savefig("{}/Training_MAE.png".format(save_dir))
                                                        
