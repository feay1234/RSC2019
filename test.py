import os
os.environ['THEANO_FLAGS'] = "device=cuda,force_device=True,floatX=float32"
import theano
import keras
