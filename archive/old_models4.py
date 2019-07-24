def stacked_LSTM0(**kwargs):
    """A denser version of model_multi"""
    loss_fn = 'mse'
    if "custom_loss" in kwargs.keys(): loss_fn = losses[kwargs["custom_loss"]]

    model = keras.models.Sequential()
    model.add(Dense(216, input_shape=(36,)))
    model.add(Reshape(target_shape=(6,36)))
    model.add(LSTM(108, return_sequences=True))
    #model.add(TimeDistributed(Dense(108, activation='tanh')))
    model.add(LSTM(72, return_sequences=True))
    #model.add(TimeDistributed(Dense(72, activation='tanh')))
    model.add(LSTM(36, return_sequences=True))
    #model.add(TimeDistributed(Dense(36, activation='tanh')))
    model.add(LSTM(18, return_sequences=True))
    #model.add(TimeDistributed(Dense(18, activation='tanh')))
    model.add(LSTM(3, return_sequences=True))
    #model.add(TimeDistributed(Dense(3, activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(10e-5, decay=0.0)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model


def bidirectional_LSTM0(**kwargs):
    """A denser version of model_multi"""
    loss_fn = 'mse'
    if "custom_loss" in kwargs.keys(): loss_fn = losses[kwargs["custom_loss"]]

    config = {'size1': 800.0, 'size2': 54.0}
    model = keras.models.Sequential()
    model.add(Reshape(target_shape=(6,6), input_shape=(36,)))
    # Initially, due to typo, size1 = size2
    model.add(TimeDistributed(Dense(int(config['size1']), activation='tanh')))
    model.add(Bidirectional(LSTM(int(config['size2']), return_sequences=True)))
    model.add(TimeDistributed(Dense(27, activation='tanh')))
    model.add(TimeDistributed(Dense(9, activation='tanh')))
    model.add(TimeDistributed(Dense(3, activation='tanh')))

    optimizer = tf.keras.optimizers.Adam(10e-5, decay=0.)
    model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

    return model
