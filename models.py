from tensorflow import keras
from tensorflow.keras import layers

from features import *

n_features_input = 63
n_target_features = 5

def create_model():
    model = keras.Sequential()
    model.add(layers.Dense(n_features_input, activation='relu'))


   x_t_lep = TimeDistributed( Dense(100), input_shape=inshape_t_lep )(input_t_lep)
   x_t_lep = LSTM(  80, return_sequences=True )(x_t_lep)
   x_t_lep = LSTM(  80, return_sequences=True )(x_t_lep)
   x_t_lep = LSTM(  50, return_sequences=True )(x_t_lep)
   x_t_lep = LSTM(  50, return_sequences=False )(x_t_lep)
   x_t_lep = Dense(30)(x_t_lep)
   x_t_lep = Dense(20)(x_t_lep)
   #output_t_lep = Dense( n_target_features )(x_t_lep)

   x_t_had = TimeDistributed( Dense(100), input_shape=inshape_t_had )(input_t_had)
   x_t_had = LSTM(  80, return_sequences=True )(x_t_had)
   x_t_had = LSTM(  80, return_sequences=True )(x_t_had)
   x_t_had = LSTM(  50, return_sequences=True )(x_t_had)
   x_t_had = LSTM(  50, return_sequences=False )(x_t_had)
   x_t_had = Dense(30)(x_t_had)
   x_t_had = Dense(20)(x_t_had)
   #output_t_had = Dense( n_target_features )(x_t_had)

   x_ttbar = concatenate( [ x_t_lep, x_t_had ] )
   x_ttbar = Dense( 30 )(x_ttbar)
   x_ttbar = Dense( 20 )(x_ttbar)
   output  = Dense( n_target_features )(x_ttbar)

   model = Model(inputs=[input_t_lep, input_t_had], outputs=[output] )

   model.compile( optimizer='adam', loss='mean_squared_error' )

   return model


################


def create_model_rnn():
   inshape   = ( n_rows, n_cols )

   print "INFO: building model: input shape:", inshape

   input = Input( shape=inshape )
   dropout = 0.00

   x = TimeDistributed( Dense(100), input_shape=inshape )(input)

   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  80, recurrent_dropout=dropout, return_sequences=True )(x)

   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=True )(x)
   x = LSTM(  50, recurrent_dropout=dropout, return_sequences=False )(x)

   x = Dense(30)(x)
   x = Dense(20)(x)

   output = Dense( n_target_features )(x)

   model = Model(inputs=[input], outputs=[output] )

   model.compile( optimizer='adam', loss='mean_squared_error' )

   return model


def create_model_multi():
   print "INFO: building model: (W_lep,b_lep,t_lep,W_had,b_had,t_had)"

   shape_jets = (n_jets_per_event,n_features_per_jet)
   shape_lep  = (n_features_lepton,)

   input_jets = Input( shape=shape_jets, name='input_jets' )
   input_lep  = Input( shape=shape_lep,  name='input_lept' )

   x_W_lep = Dense(30)(input_lep)
   x_W_lep = Dense(30, activation="relu")(x_W_lep)
   x_W_lep = Dense(20, activation="relu")(x_W_lep)
   x_W_lep = Dense(10, activation="relu")(x_W_lep)
   x_W_lep = Dense(6)(x_W_lep)
   x_W_lep_out = Dense(3, name='W_lep_out')(x_W_lep)

#   x_b_had = TimeDistributed( Dense(100), input_shape=(5,6) )(input_jets)
#   x_b_had = LSTM( 80, return_sequences=True)(x_b_had)
#   x_b_had = LSTM( 50, return_sequences=True)(x_b_had)
#   x_b_had = LSTM( 30, return_sequences=False)(x_b_had)
#   x_b_had = Dense(30, activation="relu")(x_b_had)
#   x_b_had = Dense(20, activation="relu")(x_b_had)
#   x_b_had = Dense(10, activation="relu")(x_b_had)
#   x_b_had_out = Dense(3, name="b_had_out")(x_b_had)

#   x_b_lep = TimeDistributed( Dense(100), input_shape=(5,6) )(input_jets)
#   x_b_lep = LSTM( 80, return_sequences=True)(x_b_lep)
#   x_b_lep = LSTM( 50, return_sequences=True)(x_b_lep)
#   x_b_lep = LSTM( 30, return_sequences=False)(x_b_lep)
#   #x_b_lep = concatenate( [ x_b_lep, x_W_lep ] )
#   x_b_lep = Dense(30, activation="relu")(x_b_lep)
#   x_b_lep = Dense(20, activation="relu")(x_b_lep)
#   x_b_lep = Dense(10, activation="relu")(x_b_lep)
#   x_b_lep_out = Dense(3, name="b_lep_out")(x_b_lep)

   # shared layers
   x_jets = TimeDistributed( Dense(100), input_shape=(5,6) )(input_jets)
   x_jets = LSTM( 100, return_sequences=True)(x_jets)
   x_jets = LSTM(  80, return_sequences=True)(x_jets)
   x_jets = LSTM(  50, return_sequences=False)(x_jets)

   x_b_had = Dense(30, activation="tanh")(x_jets)
   x_b_had = Dense(20, activation="tanh")(x_b_had)
   x_b_had = Dense(10, activation="tanh")(x_b_had)
   x_b_had_out = Dense(3, name="b_had_out")(x_b_had)

   x_b_lep = Dense(30, activation="tanh")(x_jets)
   x_b_lep = Dense(20, activation="tanh")(x_b_lep)
   x_b_lep = Dense(10, activation="tanh")(x_b_lep)
   x_b_lep_out = Dense(3, name="b_lep_out")(x_b_lep)

   x_W_had = Dense(30, activation="tanh")(x_jets)
   x_W_had = Dense(20, activation="tanh")(x_W_had)
   x_W_had = Dense(10, activation="tanh")(x_W_had)
   x_W_had_out = Dense(3, name="W_had_out")(x_W_had)

#   x_W_had = TimeDistributed( Dense(100), input_shape=shape_jets )(input_jets)
#   x_W_had = LSTM(  100, return_sequences=True)(x_W_had)
#   x_W_had = LSTM(   80, return_sequences=True)(x_W_had)
#   x_W_had = LSTM(   50, return_sequences=False)(x_W_had)
#   x_W_had = Dense( 30, activation="tanh")(x_W_had)
#   x_W_had = Dense( 20, activation="tanh")(x_W_had)
#   x_W_had = Dense( 10, activation="tanh")(x_W_had)
#   x_W_had_out = Dense(3, name='W_had_out')(x_W_had) # (Px, Pz, Py) + mass constraint

   #x_t_had  = add( [ x_W_had_out, x_b_had_out ] )
   #x_t_had_out = Dense(3, name='t_had_out')(x_t_had)

   #x_t_lep  = add( [ x_W_lep_out, x_b_lep_out ] )
   #x_t_lep_out = Dense(3, name='t_lep_out')(x_t_lep)

   model = Model(inputs=[input_jets,
                         input_lep],
                 outputs=[x_W_lep_out, x_W_had_out,
                          x_b_lep_out, x_b_had_out ] )
                          #x_t_lep_out, x_t_had_out] )

   model.compile( optimizer='adam', loss='mean_squared_error' )

   return model
