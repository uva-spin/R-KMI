import sys 
#8. 
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import RepeatedKFold, train_test_split
from tensorflow_addons.activations import tanhshrink
from tensorflow_addons.optimizers import AdamW
sys.path.append('../')
#from Formulation.BHDVCS_tf_modified import BHDVCStf
import matplotlib.pyplot as plt 
import time

import tensorflow as tf  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import math
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking 
from tensorflow.keras import backend as K

#from non_model_utils import FFs

# class TotalFLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(TotalFLayer, self).__init__(dtype='float32')
#         self.f=BHDVCStf()

#     def call(self, inputs):
#         return self.f.total_xs(inputs[:, 0:5], inputs[:, 5:9]) # QQ, x, t, phi, k, cff1, cff2, cff3, cff4
    
class TotalFLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TotalFLayer, self).__init__(dtype='float32')
        self.f=BHDVCStf()

    def call(self, inputs):
        return self.f.total_xs(inputs[:, 0:5], inputs[:, 5:9]) # QQ, x, t, phi, k, cff1, cff2, cff3, cff4

class CustomFLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomFLayer, self).__init__(dtype='float32')
        self.f=BHDVCStf()

    def call(self, kin, params, F_data, F_err):
        F_dnn = self.f.total_xs(kin, params) 
        self.add_loss( tf.reduce_mean(tf.square( (F_dnn - F_data) / (F_err) ) ) )
        return F_dnn
    
class CustomFLayer4(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomFLayer4, self).__init__(dtype='float32')
        self.f=BHDVCStf()

    def call(self, kin, params):
        F_dnn = self.f.total_xs(kin, params) 
        return F_dnn
     
class BHDVCStf(object): 

    def __init__(self):
        self.ALP_INV = tf.constant(137.0359998)  # 1 / Electromagnetic Fine Structure Constant
        self.PI = tf.constant(3.1415926535)
        self.RAD = tf.constant(self.PI / 180.)
        self.M = tf.constant(0.938272)  # Mass of the proton in GeV
        self.GeV2nb = tf.constant(.389379 * 1000000)  # Conversion from GeV to NanoBar
        self.M2 = tf.constant(0.938272 * 0.938272)  # Mass of the proton  squared in GeV

    @tf.function
    def SetKinematics(self, QQ, x, t, k):
        ee = 4. * self.M2 * x * x / QQ  # epsilon squared
        y = tf.sqrt(QQ) / (tf.sqrt(ee) * k)  # lepton energy fraction
        xi = x * (1. + t / 2. / QQ) / (2. - x + x * t / QQ);  # Generalized Bjorken variable
        Gamma = x * y * y / self.ALP_INV / self.ALP_INV / self.ALP_INV / self.PI / 8. / QQ / QQ / tf.sqrt(1. + ee)  # factor in front of the cross section, eq. (22)
        tmin = - QQ * (2. * (1. - x) * (1. - tf.sqrt(1. + ee)) + ee) / (4. * x * (1. - x) + ee)  # eq. (31)
        Ktilde_10 = tf.sqrt(tmin - t) * tf.sqrt((1. - x) * tf.sqrt(1. + ee) + ((t - tmin) * (ee + 4. * x * (1. - x)) / 4. / QQ)) * tf.sqrt(1. - y - y * y * ee / 4.) / tf.sqrt(1. - y + y * y * ee / 4.)  # K tilde from 2010 paper
        K = tf.sqrt(1. - y + y * y * ee / 4.) * Ktilde_10 / tf.sqrt(QQ)
        return ee, y, xi, Gamma, tmin, Ktilde_10, K

    @tf.function
    def BHLeptonPropagators(self, phi, QQ, x, t, ee, y, K):
        # KD 4-vector product (phi-dependent)
        KD = - QQ / (2. * y * (1. + ee)) * (1. + 2. * K * tf.cos(self.PI - (phi * self.RAD)) - t / QQ * (1. - x * (2. - y) + y * ee / 2.) + y * ee / 2.)  # eq. (29)

        # lepton BH propagators P1 and P2 (contaminating phi-dependence)
        P1 = 1. + 2. * KD / QQ
        P2 = t / QQ - 2. * KD / QQ
        return P1, P2

    @tf.function
    def BHUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K):

        # BH utorch.larized Fourier harmonics eqs. (35 - 37)
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K]]
        c0_BH = 8. * K * K * ((2. + 3. * ee) * (QQ / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (F1 + F2) * (F1 + F2)) + (2. - y) * (2. - y) * ((2. + ee) * (
                    (4. * x * x * self.M2 / t) * (1. + t / QQ) * (
                        1. + t / QQ)
                        + 4. * (1. - x) * (1. + x * t / QQ)) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 4. * x * x * (x + (1. - x + ee / 2.) * (1. - t / QQ) * (1. - t / QQ) - x * (1. - 2. * x) * t * t / (QQ * QQ)) * (F1 + F2) * (F1 + F2)) + 8. * (
                                 1. + ee) * (1. - y - ee * y * y / 4.) * (
                                 2. * ee * (1. - t / (4. * self.M2)) * (
                                     F1 * F1 - F2 * F2 * t / (4. * self.M2)) - x * x * (
                                             1. - t / QQ) * (1. - t / QQ) * (F1 + F2) * (F1 + F2))

        c1_BH = 8. * K * (2. - y) * (
                    (4. * x * x * self.M2 / t - 2. * x - ee) * (
                        F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * x * x * (
                                1. - (1. - 2. * x) * t / QQ) * (F1 + F2) * (F1 + F2))

        c2_BH = 8. * x * x * K * K * (
                    (4. * self.M2 / t) * (F1 * F1 - F2 * F2 * t / (4. * self.M2)) + 2. * (F1 + F2) * (
                        F1 + F2))

        # BH squared amplitude eq (25) divided by e^6
        Amp2_BH = 1. / (x * x * y * y * (1. + ee) * (
                    1. + ee) * t * P1 * P2) * (c0_BH + c1_BH * tf.cos(
            self.PI - (phi * self.RAD)) + c2_BH * tf.cos(2. * (self.PI - (phi * self.RAD))))

        Amp2_BH = self.GeV2nb * Amp2_BH  # convertion to nb

        return Gamma * Amp2_BH

    @tf.function
    def IUU(self, phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, twist, tmin, xi, Ktilde_10):
        phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10 = [tf.cast(i, np.float32) for i in [phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, tmin, xi, Ktilde_10]]
        # Get BH propagators and set the kinematics
        self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)

        # Get A_UU_I, B_UU_I and C_UU_I interference coefficients
        A_U_I, B_U_I, C_U_I = self.ABC_UU_I_10(phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10)

        # BH-DVCS interference squared amplitude
        I_10 = 1. / (x * y * y * y * t * P1 * P2) * (
                    A_U_I * (F1 * ReH - t / 4. / self.M2 * F2 * ReE) + B_U_I * (F1 + F2) * (
                        ReH + ReE) + C_U_I * (F1 + F2) * ReHtilde)

        I_10 = self.GeV2nb * I_10  # convertion to nb

        return Gamma * I_10

    @tf.function
    def ABC_UU_I_10(self, phi, twist, QQ, x, t, ee, y, K, tmin, xi, Ktilde_10):  # Get A_UU_I, B_UU_I and C_UU_I interference coefficients BKM10

        if twist == "t2":
            f = 0  # F_eff = 0 ( pure twist 2)
        if twist == "t3":
            f = - 2. * xi / (1. + xi)
        if twist == "t3ww":
            f = 2. / (1. + xi)

        # Interference coefficients  (BKM10 Appendix A.1)
        # n = 0 -----------------------------------------
        # helicity - conserving (F)
        C_110 = - 4. * (2. - y) * (1. + tf.sqrt(1 + ee)) / tf.pow((1. + ee), 2) * (
                    Ktilde_10 * Ktilde_10 * (2. - y) * (2. - y) / QQ / tf.sqrt(1 + ee)
                    + t / QQ * (1. - y - ee / 4. * y * y) * (2. - x) * (1. + (
                        2. * x * (2. - x + (tf.sqrt(
                    1. + ee) - 1.) / 2. + ee / 2. / x) * t / QQ + ee) / (2. - x) / (
                                                                                                                       1. + tf.sqrt(
                                                                                                                   1. + ee))))
        C_110_V = 8. * (2. - y) / tf.pow((1. + ee), 2) * x * t / QQ * (
                    (2. - y) * (2. - y) / tf.sqrt(1. + ee) * Ktilde_10 * Ktilde_10 / QQ
                    + (1. - y - ee / 4. * y * y) * (1. + tf.sqrt(1. + ee)) / 2. * (
                                1. + t / QQ) * (1. + (tf.sqrt(1. + ee) - 1. + 2. * x) / (
                        1. + tf.sqrt(1. + ee)) * t / QQ))
        C_110_A = 8. * (2. - y) / tf.pow((1. + ee), 2) * t / QQ * (
                    (2. - y) * (2. - y) / tf.sqrt(
                1. + ee) * Ktilde_10 * Ktilde_10 / QQ * (
                                1. + tf.sqrt(1. + ee) - 2. * x) / 2.
                    + (1. - y - ee / 4. * y * y) * ((1. + tf.sqrt(1. + ee)) / 2. * (
                        1. + tf.sqrt(1. + ee) - x + (
                            tf.sqrt(1. + ee) - 1. + x * (3. + tf.sqrt(1. + ee) - 2. * x) / (
                                1. + tf.sqrt(1. + ee)))
                        * t / QQ) - 2. * Ktilde_10 * Ktilde_10 / QQ))
        # helicity - changing (F_eff)
        C_010 = 12. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 ee + (2. - 6. * x - ee) / 3. * t / QQ)
        C_010_V = 24. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * x * t / QQ * (
                                   1. - (1. - 2. * x) * t / QQ)
        C_010_A = 4. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * t / QQ * (
                                   8. - 6. * x + 5. * ee) * (
                                   1. - t / QQ * ((2. - 12 * x * (1. - x) - ee)
                                                            / (8. - 6. * x + 5. * ee)))
        # n = 1 -----------------------------------------
        # helicity - conserving (F)
        C_111 = -16. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * ((1. + (1. - x) * (tf.sqrt(
            1 + ee) - 1.) / 2. / x + ee / 4. / x) * x * t / QQ - 3. * ee / 4.) - 4. * K * (
                                 2. - 2. * y + y * y + ee / 2. * y * y) * (
                                 1. + tf.sqrt(1 + ee) - ee) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 1. - (1. - 3. * x) * t / QQ + (
                                     1. - tf.sqrt(1 + ee) + 3. * ee) / (
                                             1. + tf.sqrt(1 + ee) - ee) * x * t / QQ)
        C_111_V = 16. * K / tf.pow(tf.sqrt(1. + ee), 5) * x * t / QQ * (
                    (2. - y) * (2. - y) * (1. - (1. - 2. * x) * t / QQ) + (
                        1. - y - ee / 4. * y * y)
                    * (1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (t - tmin) / QQ)
        C_111_A = -16. * K / tf.pow((1. + ee), 2) * t / QQ * (
                    (1. - y - ee / 4. * y * y) * (1. - (1. - 2. * x) * t / QQ + (
                        4. * x * (1. - x) + ee) / 4. / tf.sqrt(1. + ee) * (
                                                                                  t - tmin) / QQ)
                    - tf.pow((2. - y), 2) * (
                                1. - x / 2. + (1. + tf.sqrt(1. + ee) - 2. * x) / 4. * (
                                    1. - t / QQ) + (4. * x * (1. - x) + ee) / 2. / tf.sqrt(
                            1. + ee) * (t - tmin) / QQ))
        # helicity - changing (F_eff)
        C_011 = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            (1. + ee), 2) * (tf.pow((2. - y), 2) * (t - tmin) / QQ * (
                    1. - x + ((1. - x) * x + ee / 4.) / tf.sqrt(1. + ee) * (
                        t - tmin) / QQ)
                                  + (1. - y - ee / 4. * y * y) / tf.sqrt(1 + ee) * (
                                              1. - (1. - 2. * x) * t / QQ) * (
                                              ee - 2. * (1. + ee / 2. / x) * x * t / QQ))
        C_011_V = 16. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * x * t / QQ * (
                                   tf.pow(Ktilde_10 * (2. - y), 2) / QQ + tf.pow(
                               (1. - (1. - 2. * x) * t / QQ), 2) * (
                                               1. - y - ee / 4. * y * y))
        C_011_A = 8. * math.sqrt(2.) * tf.sqrt(1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * t / QQ * (
                                   tf.pow(Ktilde_10 * (2. - y), 2) * (1. - 2. * x) / QQ + (
                                       1. - (1. - 2. * x) * t / QQ)
                                   * (1. - y - ee / 4. * y * y) * (
                                               4. - 2. * x + 3. * ee + t / QQ * (
                                                   4. * x * (1. - x) + ee)))
        # n = 2 -----------------------------------------
        # helicity - conserving (F)
        C_112 = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                     2) * (
                                 2. * ee / tf.sqrt(1. + ee) / (1. + tf.sqrt(1. + ee)) * tf.pow(
                             Ktilde_10, 2) / QQ + x * t * (
                                             t - tmin) / QQ / QQ * (1. - x - (
                                     tf.sqrt(1. + ee) - 1.) / 2. + ee / 2. / x))
        C_112_V = 8. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                       2) * x * t / QQ * (
                                   4. * tf.pow(Ktilde_10, 2) / tf.sqrt(1. + ee) / QQ + (
                                       1. + tf.sqrt(1. + ee) - 2. * x) / 2. * (1. + t / QQ) * (
                                               t - tmin) / QQ)
        C_112_A = 4. * (2. - y) * (1. - y - ee / 4. * y * y) / tf.pow((1. + ee),
                                                                                                       2) * t / QQ * (
                                   4. * (1. - 2. * x) * tf.pow(Ktilde_10, 2) / tf.sqrt(
                               1. + ee) / QQ - (3. - tf.sqrt(
                               1. + ee) - 2. * x + ee / x) * x * (
                                               t - tmin) / QQ)
        # helicity - changing (F_eff)
        C_012 = -8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee), 5) * (
                                 1. + ee / 2.) * (
                                 1. + (1. + ee / 2. / x) / (1. + ee / 2.) * x * t / QQ)
        C_012_V = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                      5) * x * t / QQ * (
                                   1. - (1. - 2. * x) * t / QQ)
        C_012_A = 8. * math.sqrt(2.) * K * (2. - y) * tf.sqrt(
            1. - y - ee / 4. * y * y) / tf.pow((1. + ee), 2) * t / QQ * (
                                   1. - x + (t - tmin) / 2. / QQ * (
                                       4. * x * (1. - x) + ee) / tf.sqrt(1. + ee))
        # n = 3 -----------------------------------------
        # helicity - conserving (F)
        C_113 = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(tf.sqrt(1. + ee),
                                                                                               5) * (
                                 tf.sqrt(1. + ee) - 1.) * (
                                 (1. - x) * t / QQ + (tf.sqrt(1. + ee) - 1.) / 2. * (
                                     1. + t / QQ))
        C_113_V = -8. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * x * t / QQ * (tf.sqrt(1. + ee) - 1. + (
                    1. + tf.sqrt(1. + ee) - 2. * x) * t / QQ)
        C_113_A = 16. * K * (1. - y - ee / 4. * y * y) / tf.pow(
            tf.sqrt(1. + ee), 5) * t * (t - tmin) / QQ / QQ * (
                                   x * (1. - x) + ee / 4.)

        # A_U_I, B_U_I and C_U_I
        A_U_I = C_110 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
            QQ) * f * C_010 + (C_111 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
            QQ) * f * C_011) * tf.cos(self.PI - (phi * self.RAD)) + (
                                 C_112 + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                             QQ) * f * C_012) * tf.cos(
            2. * (self.PI - (phi * self.RAD))) + C_113 * tf.cos(3. * (self.PI - (phi * self.RAD)))
        B_U_I = xi / (1. + t / 2. / QQ) * (
                    C_110_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * C_010_V + (
                                C_111_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * C_011_V) * tf.cos(self.PI - (phi * self.RAD)) + (
                                C_112_V + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * C_012_V) * tf.cos(
                2. * (self.PI - (phi * self.RAD))) + C_113_V * tf.cos(3. * (self.PI - (phi * self.RAD))))
        C_U_I = xi / (1. + t / 2. / QQ) * (
                    C_110 + C_110_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                QQ) * f * (C_010 + C_010_A) + (
                                C_111 + C_111_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * (C_011 + C_011_A)) * tf.cos(self.PI - (phi * self.RAD)) + (
                                C_112 + C_112_A + math.sqrt(2.) / (2. - x) * Ktilde_10 / tf.sqrt(
                            QQ) * f * (C_012 + C_012_A)) * tf.cos(
                2. * (self.PI - (phi * self.RAD))) + (C_113 + C_113_A) * tf.cos(
                3. * (self.PI - (phi * self.RAD))))

        return A_U_I, B_U_I, C_U_I

    @tf.function
    def total_xs(self, kins, cffs,phi):
        ffs = FFs()
        # c0fit = dvcs

        k, QQ, x, t = tf.split(kins, num_or_size_splits=4 , axis=1)
     
        F1, F2 = ffs.F1_F2(t) # calculating F1 and F2 using passed data as opposed to passing in F1 and F2
        ReH, ReE, ReHtilde, c0fit = tf.split(cffs, num_or_size_splits=4, axis=1)  # output of network
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.SetKinematics(QQ, x, t, k)
        P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return f_pred
    @tf.function
    def total_xs2(self, kins, cffs, dvcs, ReH):
        ffs = FFs()
        c0fit = dvcs 

        k, QQ, x, t, phi = tf.split(kins, num_or_size_splits=5, axis=1)
     
        F1, F2 = ffs.F1_F2(t) # calculating F1 and F2 using passed data as opposed to passing in F1 and F2
        ReE, ReHtilde = tf.split(cffs, num_or_size_splits=2, axis=1)  # output of network
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.SetKinematics(QQ, x, t, k)
        P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return f_pred
    @tf.function
    def total_xs3(self, kins, cffs, dvcs, ReH, ReHtilde):
        ffs = FFs()
        c0fit = dvcs 

        k, QQ, x, t, phi = tf.split(kins, num_or_size_splits=5, axis=1)
     
        F1, F2 = ffs.F1_F2(t) # calculating F1 and F2 using passed data as opposed to passing in F1 and F2
        ReE  = tf.split(cffs, num_or_size_splits=1, axis=1)  # output of network
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.SetKinematics(QQ, x, t, k)
        P1, P2 = self.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return f_pred

class DvcsData(object):
    def __init__(self, df):
        self.df = df
        self.X = df.loc[:, ['phi_x', 'k', 'QQ', 'x_b', 't', 'F1', 'F2', 'dvcs']]
        self.XnoCFF = df.loc[:, ['phi_x', 'k']] # removed redundant data
        self.y = df.loc[:, 'F']
        self.Kinematics = df.loc[:, ['QQ', 'x_b', 't']] # Removed k from kinematics
        self.erry = df.loc[:, 'errF']

    def getSet(self, setNum, itemsInSet=36):
        pd.options.mode.chained_assignment = None
        subX = self.X.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1, :]
        subX['F'] = self.y.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
        subX['errF'] = self.erry.loc[setNum*itemsInSet:(setNum+1)*itemsInSet-1]
        pd.options.mode.chained_assignment = 'warn'
        return DvcsData(subX)

    def __len__(self):
        return len(self.X)

    def sampleY(self):
        return np.random.normal(self.y, self.erry)

    def sampleWeights(self):
        return 1/self.erry

    def getAllKins(self, itemsInSets=36):
        return self.Kinematics.iloc[np.array(range(len(self.df)//itemsInSets))*itemsInSets, :]

def F2VsPhi(dataframe,SetNum,xdat,cffs):
	f = BHDVCStf().total_xs
	TempFvalSilces=dataframe[dataframe["#Set"]==SetNum]
	TempFvals=TempFvalSilces["F"]
	TempFvals_sigma=TempFvalSilces["errF"]

	temp_phi=TempFvalSilces["phi_x"]
	plt.errorbar(temp_phi,TempFvals,TempFvals_sigma,fmt='.',color='blue',label="Data")
	plt.xlim(0,368)
	temp_unit=(np.max(TempFvals)-np.min(TempFvals))/len(TempFvals)
	plt.ylim(np.min(TempFvals)-temp_unit,np.max(TempFvals)+temp_unit)
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.legend(loc=4,fontsize=10,handlelength=3)
	plt.title("Local fit with data set #"+str(SetNum),fontsize=20)
	plt.plot(temp_phi, f(xdat,cffs), 'g--', label='fit')
	file_name = "plot_set_number_{}.png".format(SetNum)
	plt.savefig(file_name)

def cffs_from_globalModel(model, kinematics, numHL=1):
	'''
	:param model: the model from which the cffs should be predicted
	:param kinematics: the kinematics that should be used to predict
	:param numHL: the number of hidden layers:
	'''
	subModel = tf.keras.backend.function(model.layers[0].input, model.layers[numHL+2].output)
	return subModel(np.asarray(kinematics)[None, 0])[0]



############# Defined in October 2023 #############

class Models: 
    def tf_model1(self, data_length):
        initializer = tf.keras.initializers.HeNormal() 
        #### QQ, x_b, t, phi, k ####
        inputs = tf.keras.Input(shape=(5))
        QQ, x_b, t, phi, k = tf.split(inputs, num_or_size_splits=5, axis=1)
        kinematics = tf.keras.layers.concatenate([QQ, x_b, t], axis=1)
        x1 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(kinematics)
        x2 = tf.keras.layers.Dense(100, activation="tanh", kernel_initializer=initializer)(x1)
        outputs = tf.keras.layers.Dense(4, activation="linear", kernel_initializer=initializer)(x2)
        #### QQ, x_b, t, phi, k, cffs ####
        total_FInputs = tf.keras.layers.concatenate([inputs, outputs], axis=1)

        TotalF = TotalFLayer()(total_FInputs) # get rid of f1 and f2

        tfModel = tf.keras.Model(inputs=inputs, outputs = TotalF, name="tfmodel")

        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            0.0085, data_length, 0.96, staircase=False, name=None
        )

        tfModel.compile(
            optimizer = tf.keras.optimizers.Adam(lr),
            loss = tf.keras.losses.MeanSquaredError()
        )

        return tfModel
    
class F_calc:
    def __init__(self):
        self.module = BHDVCStf()
        self.ffs = FFs()

    def fn_1(self, kins, cffs):
        phi, QQ, x, t, k = kins
        F1, F2 = self.ffs.F1_F2(t)
        ReH, ReE, ReHtilde, c0fit = cffs
        ee, y, xi, Gamma, tmin, Ktilde_10, K = self.module.SetKinematics(QQ, x, t, k)
        P1, P2 = self.module.BHLeptonPropagators(phi, QQ, x, t, ee, y, K)
        xsbhuu = self.module.BHUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, Gamma, K)
        xsiuu = self.module.IUU(phi, F1, F2, P1, P2, QQ, x, t, ee, y, K, Gamma, ReH, ReE, ReHtilde, "t2", tmin, xi, Ktilde_10)
        f_pred = xsbhuu + xsiuu + c0fit
        return tf.get_static_value(f_pred)


class FFs:

    GM0 = 2.792847337
    M = 0.938272
    # Kelly's parametrization fit Parameters
    a1_GEp = -0.24
    b1_GEp = 10.98
    b2_GEp = 12.82
    b3_GEp = 21.97
    a1_GMp = 0.12
    b1_GMp = 10.97
    b2_GMp = 18.86
    b3_GMp = 6.55
     
    def tau(self, t):       
        tau = - t / 4. / FFs.M / FFs.M
        return tau

    def GEp(self, t): 
        GEp = ( 1. + FFs.a1_GEp * self.tau(t) )/( 1. + FFs.b1_GEp * self.tau(t) + FFs.b2_GEp * self.tau(t) * self.tau(t) + FFs.b3_GEp * self.tau(t) * self.tau(t) * self.tau(t) )
        return GEp

    def GMp(self, t):
        GMp = FFs.GM0 * ( 1. + FFs.a1_GMp * self.tau(t) )/( 1. + FFs.b1_GMp * self.tau(t) + FFs.b2_GMp * self.tau(t) * self.tau(t) + FFs.b3_GMp * self.tau(t) * self.tau(t) * self.tau(t) )
        return GMp

    def F2(self, t):
        f2 = ( self.GMp(t) - self.GEp(t) ) / ( 1. + self.tau(t) )
        return f2

    def F1(self, t): 
        f1 = ( self.GMp(t) - self.F2(t) )
        return f1

    def F1_F2(self, t):
        return self.F1(t), self.F2(t)  
    ################################################################################################################
tf.keras.utils.get_custom_objects().update({'tanhshrink': tanhshrink}) 

bkm10 = BHDVCStf()

'''GPD_MODEL = 'basic'
NUM_OF_REPLICAS = 3 
early_stop = True
replica = True
cross_validation = False # True #True 

datafile = 'shorted_pseudoo.csv' 

epochs = 500 '''   
#############################################################################################################
# get (pseudo)data file 
def get_data():
    df = pd.read_csv(datafile, dtype=np.float32)
    return df

# filtering the unique set values to prevent overfitting    
def filter_unique_sets(data):
    unique_sets = set()
    filtered_data = {key: [] for key in data.keys()}
    for i in range(len(data['set'])):
        if data['set'][i] not in unique_sets: 
            unique_sets.add(data['set'][i])
            for key in data.keys():
                filtered_data = {key: np.array(value) for key, value in filtered_data.items()}
    return filtered_data
def filter_unique_sets(df):
    # Drop duplicate rows based on the first column
    unique_rows = df.drop_duplicates(subset=df.columns[0])
    return unique_rows 
# Normalize QQ, xB, t
def normalize(QQ, xB, t):
    QQ_norm = -1 + 2 * (QQ / 10) 
    xB_norm = -1 + 2 * (xB / 0.8)
    t_norm = -1 + 2 * ((t + 2) / 2 )
    return QQ_norm, xB_norm, t_norm


def gen_replica(pseudo):
    F_rep = np.random.normal(loc=pseudo['F'], scale=abs(pseudo['errF']*pseudo['F'])) # added abs for runtime error: 'ValueError: scale < 0'
    errF_rep = pseudo['errF'] * F_rep
    #errF_rep = pseudo['varF'] * F_rep
    
    replica_dict = {'set': pseudo['set'], 
                    'k': pseudo['k'], 'QQ': pseudo['QQ'], 'xB': pseudo['xB'], 't': pseudo['t'],     
                    'phi': pseudo['phi'], 'F': F_rep,'errF': errF_rep}       
    return pd.DataFrame(replica_dict)

# Reduced chi2 custom Loss function (model predicted inside loss)
def rchi2_Loss(kin, pars, F_data, F_err, phi):
    #print('pars')
    #print(pars[0])
    kin = tf.cast(kin, pars.dtype)
    F_dnn = tf.reshape(bkm10.total_xs(kin, pars, phi), [-1])
    F_data = tf.cast(F_data, pars.dtype)
    F_err = tf.cast(F_err, pars.dtype) 
    loss = tf.reduce_mean(tf.square( (F_dnn - F_data) / (F_err) ) ) 
    return loss

def fit_replica(i, pseudoo, j):
    epochs = 500  
    # ----- prepare input data -----------  
    #pseduo = filter_unique_sets(pseudo)
    
    
    def build_model():    # smae as above, but longer as it got cut off, 5000 epochs
    # model 75, info path: /media/lily/Data/GPDs/ANN/KMI/tunning/kt_RS
        #j=25
        j= 27 
        model = tf.keras.Sequential([ 
                Masking(mask_value=0, input_shape=(3, )),
                tf.keras.layers.Dense(150, activation="sigmoid", input_shape=(3,)), 
                tf.keras.layers.Dense(150 *j, activation="tanhshrink"),tf.keras.layers.Dense(145 *j, activation="tanh"),
                tf.keras.layers.Dense(75 *j, activation="tanh"), tf.keras.layers.Dense(70 *j, activation="tanh"), 
                tf.keras.layers.Dense(60 *j, activation="tanh"), tf.keras.layers.Dense(51 *j, activation="tanh"),   
                tf.keras.layers.Dense(4, activation="linear") # ReH, ReE, ReHt, no dvcs dvcs 
            ])    
        return model   

    
    
    if replica:        
        data = gen_replica(pseudoo) # generate replica
    else:
        data = pseudoo  
    #print(data)
    pseudo = filter_unique_sets(data) 
    PHI = data['phi'].head(24).to_numpy()
    #pseudo
    #PHI
    FFF = data['F']#.iloc(2)
    FFF_error = data['errF']
    # FFF.iloc[0]
    F_store = np.zeros((24, 89))
    F_error_store = np.zeros((24, 89) ) 
    #F_first = np.zeros(89)
    for j in range(24): 
        for i in range(0,89):
            F_store[j, i] = FFF[j+i*24]
            F_error_store[j,i] = FFF_error[j+i*24]

    #kin = np.dstack((data['k'], data['QQ'] , data['xB'], data['t'] ))
    
    kin = np.dstack((pseudo['k'], pseudo['QQ'] , pseudo['xB'], pseudo['t'] ))
    kin = kin.reshape(kin.shape[1:]) # loss inputs
    #QQ_norm, xB_norm, t_norm = normalize(data['QQ'] , data['xB'], data['t']) 
    QQ_norm, xB_norm, t_norm = normalize(pseudo['QQ'] , pseudo['xB'], pseudo['t']) 
    kin3_norm = np.array([QQ_norm, xB_norm, t_norm]).transpose() # model inputs
    pseudo = filter_unique_sets(pseudoo)
    pars_true = np.array([pseudo['ReH'], pseudo['ReE'], pseudo['ReHtilde'], pseudo['dvcs']]).transpose()#, pseudo['dvcs']]).transpose() # true parameters
    #print(len(pars_true))
    #dvcs = np.array([pseudo['dvcs']]).transpose()
    F_train = []
    F_test = []
    Ferr_train = []
    Ferr_test = []
    #print(len(dvcs))
    # ---- split train and testing replica data samples ---- 
    if cross_validation:
        rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
        for train_index, test_index in rkf.split(kin):
            kin_train, kin_test, kin3_norm_train, kin3_norm_test = kin[train_index], kin[test_index], kin3_norm[train_index], kin3_norm[test_index]
            F_train, F_test = data['F'][train_index], data['F'][test_index]
            Ferr_train, Ferr_test = data['errF'][train_index], data['errF'][test_index]
    else:
        kin_train, kin_test, kin3_norm_train, kin3_norm_test = train_test_split(kin, kin3_norm, test_size=0.10, random_state=42)
        #dvcss, dvcsv = train_test_split(dvcs, test_size=0.10, random_state=42)
        for i in range(24):
            F_trainn, F_testt, Ferr_trainn, Ferr_testt = train_test_split(F_store[i], F_error_store[i], test_size=0.10, random_state=42)
            F_train += [F_trainn]
            F_test += [F_testt]
            Ferr_train += [Ferr_trainn]
            Ferr_test += [Ferr_testt]
        #print(F_train)
            #print(F_trainn)
        

        
        
        
    model = build_model()
    model.summary()
      
    # Instantiate an optimizer to train the model.
    optimizer = AdamW(learning_rate=0.0001, weight_decay=0.0001)
    rmse = tf.keras.metrics.RootMeanSquaredError()
    mape = tf.keras.metrics.MeanAbsolutePercentageError()
    ##########################################################



################################################
#think I am goona keep, train and test outputs the same, so just the value on first set of phi

    @tf.function 
    def train_step(loss_inputs, inputs, targets, weights):
        for ii in range(23, -1, -1):
            
            with tf.GradientTape() as tape:
                pars = model(inputs) 
                loss_value = rchi2_Loss(loss_inputs, pars, targets[ii], weights[ii], PHI[ii])
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value
        
    @tf.function
    def test_step(loss_inputs, inputs, targets, weights):
        pars = model(inputs)
        val_loss_value = rchi2_Loss(loss_inputs, pars, targets, weights, PHI[0])
        return val_loss_value
    
    # Functions to update the metrics
    # MAPE for a given parameter: accuracy = (100 - MAPE)
    @tf.function
    def metricWrapper(m, kin3_norm, pars_true):
        mape.reset_states()
        def mapeMetric():
            pars = model(kin3_norm)       
            mape.update_state(pars_true[:, m], pars[:, m])
            return tf.convert_to_tensor(mape.result(), np.float32)
        return mapeMetric()
    # F RMSE weighted over F_errors
    '''@tf.function
    def rmseMetric(kin, kin3_norm, pars_true, F_errors):    
        pars = model(kin3_norm)
        kin = tf.cast(kin, pars.dtype)        
        pars_true = tf.cast(pars_true, pars.dtype)
        F_dnn = tf.reshape(bkm10.total_xs(kin, pars, PHI[0], dvcs), [-1])
        F_true = tf.reshape(bkm10.total_xs(kin, pars_true, PHI[0], dvcs), [-1])
        weights = 1. / F_errors
        rmse.update_state(F_true, F_dnn, sample_weight = weights)
        return tf.convert_to_tensor(rmse.result(), np.float32)'''
 
    # Keep results for plotting
    train_loss_results = [] 
    val_loss_results = []
   # F_rmse_results = []
    total_mape_results = []
    ReH_mape_results = []
    ReE_mape_results = []
    ReHt_mape_results = []
    dvcs_mape_results = []
    predictions_results = []

    patience = 1000
    wait = 0
    best = float("inf")
    
    for epoch in range(epochs):
       
        loss_value = train_step(kin_train, kin3_norm_train, F_train, Ferr_train)
        val_loss_value = test_step(kin_test, kin3_norm_test, F_test, Ferr_test)

        # Update metrics    
       # F_rmse = rmseMetric(kin, kin3_norm, pars_true, pseudo['errF'])
        pars_mape = [metricWrapper(m, kin3_norm, pars_true).numpy()  for m in range(4)]
        total_mape = np.mean(pars_mape) 
         
        # End epoch
        train_loss_results.append(loss_value)
        val_loss_results.append(val_loss_value) 
       # F_rmse_results.append(F_rmse)
        total_mape_results.append(total_mape)
        ReH_mape_results.append(pars_mape[0])
        ReE_mape_results.append(pars_mape[1])
        ReHt_mape_results.append(pars_mape[2])
        dvcs_mape_results.append(pars_mape[3])
       # rmse.reset_states()
        mape.reset_states()
           
                
###################################################################################################################################
        if (epoch==(epochs-1)):
            predictionss = model.predict(kin3_norm)
            predictions_resultss = np.array(predictionss)
            resultss = "Epoch {:03d}: Loss: {:.3f} val_Loss: {:.3f}  ReH_mape: {:.5f} ReE_mape: {:.5f} ReHt_mape: {:.5f} dvcs_mape: {:.5f} total_mape: {:.5f}".format(epoch, loss_value, val_loss_value,  pars_mape[0], pars_mape[1], pars_mape[2],pars_mape[3],  total_mape)
            joined_array = np.concatenate((predictions_resultss, kin3_norm), axis=1)
            cff_results = pd.DataFrame(predictions_resultss, columns = ['ReH_results', 'ReE_results', 'ReHt_results', 'dvcs_results','QQ', 'x_B', 't'])
            
            cff_results.to_csv('phiidvcs_data.csv', mode='a', index=False)
            results_file_path = 'phiidvcs_data.py'

            with open(results_file_path, 'a') as file:
                file.write(resultss + '\n')
            predictionss = model.predict(kin3_norm)[:,0] 
            ReHH = np.array(predictionss)
            return ReHH
            
        
K.clear_session()        
        
        

        
firsttt = get_data()
pseudo = filter_unique_sets(firsttt) 



#kin = np.dstack((pseudo['k'], pseudo['QQ'] , pseudo['xB'], pseudo['t'], pseudo['phi']))
#kin = kin.reshape(kin.shape[1:]) # loss inputs

ddf = get_data()
kin_norepeat = filter_unique_sets(ddf)

'''kin = []
target_length = 24
for i in range(int(len(kin_norepeat))):
#for i in range(2):
    matching_rows = ddf[(ddf['k'] == kin_norepeat.iloc[i,2]) & 
                       (ddf['QQ'] == kin_norepeat.iloc[i,3]) & 
                       (ddf['xB'] == kin_norepeat.iloc[i,4]) & 
                       (ddf['t'] == kin_norepeat.iloc[i,5])] 
    Phi = matching_rows.iloc[:, 6].tolist()#.to_numpy() 
    #print(Phi)
    #phii+=[Phi]
    
    kinn = [ 
        np.pad(np.array([kin_norepeat.iloc[i,2]]), (0, target_length - 1), 'constant'),
        np.pad(np.array([kin_norepeat.iloc[i,3]]), (0, target_length - 1), 'constant'),
        np.pad(np.array([kin_norepeat.iloc[i,4]]), (0, target_length - 1), 'constant'),
        np.pad(np.array([kin_norepeat.iloc[i,5]]), (0, target_length - 1), 'constant'),
        np.pad(np.array(Phi), (0, target_length - len(Phi)), 'constant')
    ] 
   # print(kinn) 
    kin+=[kinn]''' 

#print('pseudo') 
##print(pseudo)
#print('kinematicsa') 
#print(kin)

#Model_number = PHI = first['phi'].head(24).to_numpy()
#pseudo
#first = get_data()
'''pseudo = filter_unique_sets(first) 
PHI = first['phi'].head(24).to_numpy()
#pseudo
PHI
FFF = first['F']#.iloc(2)
FFF_error = first['errF']
# FFF.iloc[0]
F_store = np.zeros((24, 89))
F_error_store = np.zeros((24, 89) ) 
#F_first = np.zeros(89)
for j in range(24): 
    for i in range(0,89):
        F_store[j, i] = FFF.iloc[j+i*24]
        F_error_store[j,i] = FFF_error[j+i*24]
'''
        
GPD_MODEL = 'basic' 
NUM_OF_REPLICAS = 3 
early_stop = True
replica = True
cross_validation = False # True #True 

datafile = 'shorted_pseudoo.csv' 
first = get_data()

def get_data(neww):
    df = pd.read_csv(neww, dtype=np.float32)
    return df
 
stor1 = []
for i in range(0, NUM_OF_REPLICAS): 
    j=27
    start = time.time() 
    stor1 += [fit_replica(i, first, j)] 
#ReH_mean = np.mean(np.array(stor1), axis = 0)


print('done')      