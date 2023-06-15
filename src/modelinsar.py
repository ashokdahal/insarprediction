from tensorflow import keras
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from scipy.stats import genpareto
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import genpareto
#import tensorflow_probability.distributions as tfp
tfd = tfp.distributions
class insarpred():
    def __init__(self,modelparam):
        self.depth=modelparam['depth']
        self.infeatures=modelparam['infeatures']
        self.outfeatures=modelparam['outfeatures']
        self.units=modelparam['units']
        self.kernel_initializer=modelparam['kernel_initializer']
        self.bias_initializer=modelparam['bias_initializer']
        self.droupout=modelparam['droupout']
        self.batchnormalization=modelparam['batchnormalization']
        self.dropoutratio=modelparam['dropoutratio']
        self.lastactivation=modelparam['lastactivation']
        self.middleactivation=modelparam['middleactivation']
        self.lr=modelparam['lr']
        self.decay_steps=modelparam['decay_steps']
        self.decay_rate=modelparam['decay_rate']
        self.opt=tf.keras.optimizers.Adam

    def getAreaDensityModel(self):

        features_only=Input((self.infeatures))
        x=layers.Dense(units=self.units,activation='selu',name=f'DN_0',kernel_initializer=self.kernel_initializer,bias_initializer=self.bias_initializer)(features_only)
        for i in range(1,self.depth+1):
            x=layers.Dense(activation=None,units=self.units,name=f'DN_{str(i)}',kernel_initializer=self.kernel_initializer,bias_initializer=self.bias_initializer)(x)
            if self.batchnormalization:
                x= layers.BatchNormalization()(x)
            if self.droupout:
                x= layers.Dropout(self.dropoutratio)(x) 
            x=layers.LeakyReLU(alpha=0.02)(x)     
        out_areaDen=layers.Dense(units=self.outfeatures,activation=self.lastactivation,name='insar')(x)
        self.model = Model(inputs=features_only, outputs=out_areaDen)

    def getOptimizer(self,):
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.lr,decay_steps=self.decay_steps,decay_rate=self.decay_rate,staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    def preparemodel(self,weights=None):
        self.getAreaDensityModel()
        self.getOptimizer()
        self.model.compile(optimizer=self.optimizer, loss=keras.losses.MeanAbsoluteError(), metrics=[keras.metrics.MeanAbsolutePercentageError(),keras.metrics.MeanAbsoluteError()])