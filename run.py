import tensorflow as tf
import json
from src import preparedata
from src import modelinsar
from src import traininsar

print(tf.__version__)
print(tf.config.list_physical_devices(
    device_type=None
))

params=json.load(open('params/params.json','r'))
dataset=preparedata.readInsarData(params['dataprepinargs'])
dataset.preparedata()

landslidehazard=modelinsar.insarpred(params['modelparam'])
landslidehazard.preparemodel()

traininsar.trainmodel(landslidehazard.model,dataset.X_train,dataset.Y_train,params['trainparam'])
