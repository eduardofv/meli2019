#!/usr/bin/env python
# coding: utf-8

# ## MercadoLibre Challenge 
# ### TensorflowHub embeddings

# In[ ]:


EXPERIMENT_NAME = "meli-TFH" 
EXPERIMENT_VERSION = "v3"
LOG_DIR = "../logs/TFH-3"


# ## Version log
# ### TFH-v2_3
# Viene de 2_2_2
# 
# ### TFH-v2_2
# 
# Ajustar hiperparámetros para buscar mejor score
# 
# ### TFH-v2_1
# 
# Usar normalización
# 
# ### TFH-v2_0
# 
# Nueva arquitectura, incorporarar info de lenguaje
# 
# Viene de meli-TFH-v1_1

# ## Development

# ### Initialize

# In[ ]:


#get_ipython().run_line_magic('matplotlib', 'inline')

import sys
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #disable gpu

import time
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score 

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, Dense, Dropout
from tensorflow.keras.layers import Concatenate, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.utils import to_categorical


# In[ ]:


print(f"TF Version: {tf.__version__}")
print(f"TF Hub Version: {hub.__version__}")
assert( tf.__version__ >= "2.0.0-beta0" ) #rc1
assert( hub.__version__ >= "0.3" ) #"0.7.0.dev"


# In[ ]:


#initialize env
#seeds to make reproducible
#todo: check reproducibility
np.random.seed(12347)
tf.random.set_seed(12347)

pd.options.display.max_rows = 10

#filenames and directories
DATASET_FN = "../data/train.csv"
ROWS_TO_LOAD = None#4000000 #None == all
SAVED_MODEL_DIR = "../saved_models/"

#TFHUB_EMB_MODEL = "https://tfhub.dev/google/universal-sentence-encoder/2" 
# download the module manually if network problems 
# like URLError: <urlopen error [Errno -3] Temporary failure in name resolution>
# check https://www.tensorflow.org/hub/common_issues 
#TFHUB_EMB_MODEL = "../tf_hub_cache/gnews-swivel-20dim-v1"
#TFHUB_EMB_MODEL = "../tf_hub_cache/nnlm-es-dim50" #
#TFHUB_EMB_MODEL = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
#TFHUB_EMB_MODEL = "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128/1"
#TFHUB_EMB_MODEL = "../tf_hub_cache/nnlm-es-dim128-with-normalization"
TFHUB_EMB_MODEL = "https://tfhub.dev/google/tf2-preview/nnlm-es-dim128-with-normalization/1"
TFHUB_EMB_MODEL_DIM = 128
LANGUAGE = "portuguese"#"spanish" #"portuguese" #None

MODEL_DENSE_UNITS = 2048
MODEL_DROPOUT_RATE = 0.5

#Training
TRAIN_EPOCHS = 20
TRAIN_INITIAL_EPOCH = 15
TRAIN_BATCH_SIZE = 8192

#set some parameters on how data will be used
# specify a small proportion to speed things while testing, 1.0 when running full training
DATASET_PROPORTION_TO_USE = 1.0
# how much data will reserve for test set (of the DS prop to use) (0.10)
TEST_SET_SPLIT = 0.004
# how much of the data will be used for validation (of the DS prop to use) (0.05)
VALIDATION_SET_SPLIT = 0.004


# ### Load and prepare datasets

# In[ ]:


df = pd.read_csv(DATASET_FN, nrows=ROWS_TO_LOAD)
if LANGUAGE is not None:
    df = df[df["language"]==LANGUAGE]
print(df)


# In[ ]:


output_dim = len(df["category"].unique())
print(output_dim)
print(list(df["category"][:3]))


# In[ ]:


cat_dict = dict(zip(df["category"].unique(), np.arange(output_dim)))
inverse_cat_dict = dict(zip(cat_dict.values(), cat_dict.keys()))
labels = df["category"].map(cat_dict)
#labels = to_categorical(df["category"].map(cat_dict))
labels[:3]

# In[ ]:


language = (df["language"]=='spanish').astype("int32")
language[:3]


# ### Split datasets

# In[ ]:


num_samples = len(df)
num_test_samples = int(num_samples * TEST_SET_SPLIT)
num_training_samples = num_samples - num_test_samples

training_set_data = df["title"].head(num_training_samples)
training_set_lang = language[:num_training_samples]
training_set_labels = labels[:num_training_samples]
test_set_data = df["title"].tail(num_test_samples)
test_set_lang = language[-num_test_samples:]
test_set_labels = labels[-num_test_samples:]

print(training_set_data.shape)
print(training_set_labels.shape)
print(test_set_data.shape)
print(test_set_labels.shape)


# ### Build Model

# In[ ]:


hub_layer = hub.KerasLayer(TFHUB_EMB_MODEL,
                    input_shape=[],
                    output_shape=[TFHUB_EMB_MODEL_DIM],
                    trainable=True, 
                    dtype=tf.string)

input_lang = Input(shape=(1,))

mod = Sequential()
mod.add(hub_layer)

fo = Dense(MODEL_DENSE_UNITS, activation="relu", name="DEN_1")(mod.output)
fo = Dropout(rate=MODEL_DROPOUT_RATE)(fo)
fo = Dense(MODEL_DENSE_UNITS, activation="relu", name="DEN_2")(fo)
fo = Dropout(rate=MODEL_DROPOUT_RATE)(fo)
fo = Concatenate()([fo, input_lang])
fo = Dense(output_dim, activation="softmax", name="DEN_OUT")(fo)
model = Model(inputs=[mod.input, input_lang], outputs=fo)

optimizer = Adam()
model.compile(optimizer=optimizer, 
              loss='sparse_categorical_crossentropy')

print(model.summary())


# ### Train and Test

# In[ ]:


#run params
runid = "%s-%s-%s"%(EXPERIMENT_NAME,
     EXPERIMENT_VERSION,
     time.strftime(time.strftime('%y%m%d_%H%M',time.localtime())))

#Create saved model dir     
directory = SAVED_MODEL_DIR+"/"+runid
if not os.path.exists(directory):
    os.makedirs(directory)
    
print("Python: "+str(sys.version))
print("Tensorflow version: "+tf.__version__)
print("Keras version: "+tf.keras.__version__)
print("RUNID: "+runid)


# In[ ]:


#Train from saved point
model = tf.keras.models.load_model("../saved_models/meli-TFH-v3-191001_0150/model.hdf5", 
                                            custom_objects={"KerasLayer":hub_layer})

print('Training...')
tensorboard = TensorBoard(log_dir=LOG_DIR+'/'+runid)
checkpoint = ModelCheckpoint(directory+"/model.hdf5", monitor='val_loss',
                             verbose=1, save_best_only=True, mode="min")

# train
t0 = time.time()
print("Start:"+time.strftime("%Y%m%d_%H%M",time.localtime()))
history = model.fit([training_set_data.array, training_set_lang.array], 
          training_set_labels,
          batch_size=TRAIN_BATCH_SIZE,
          epochs=TRAIN_EPOCHS,
          initial_epoch=TRAIN_INITIAL_EPOCH,
          validation_split=VALIDATION_SET_SPLIT,
          verbose=1,
          callbacks=[tensorboard, checkpoint])

tfin = time.time()
print("End:" + time.strftime("%Y%m%d_%H%M",time.localtime()))
print(tfin-t0)


# In[ ]:


fn = directory+"/history.pickle"
with open(fn, "wb")as fo:
    pickle.dump(history.history, fo, protocol=4)
print(f"Saved {fn}")


# In[ ]:

#Last Model (current)
#analysis_model = model

#Best Model saved
analysis_model = tf.keras.models.load_model(directory+"/model.hdf5", 
                                            custom_objects={"KerasLayer":hub_layer})

print("Predict:")
predictions = analysis_model.predict([test_set_data.array, 
                                      test_set_lang.array], verbose=1)

inverse_cat_dict = dict(zip(cat_dict.values(), cat_dict.keys()))
predicted_categories = [inverse_cat_dict[np.argmax(p)] for p in predictions]

real_categories = [inverse_cat_dict[p] for p in test_set_labels]

bac = balanced_accuracy_score(real_categories, predicted_categories)
print(f"Eval BAC={bac}")


# #### History graphs

# In[ ]:


#with open(directory+"/history.pickle", "rb") as f:
#    history = pickle.load(f)


# In[ ]:


history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

fn = directory+"/history.pdf"
plt.savefig(fn)
print(f"Saved {fn}")


# # Create Submission

# In[ ]:


submission_test_df = pd.read_csv("../data/test.csv")
#submission_test_df


# In[ ]:


sub_data = submission_test_df["title"]
sub_lang = (submission_test_df["language"]=='spanish').astype("int32")

sub_predictions = analysis_model.predict([sub_data.array, sub_lang.array], verbose=1)


# In[ ]:


sub_predicted_categories = [inverse_cat_dict[np.argmax(p)] for p in sub_predictions]
#sub_predicted_categories[:10]


# In[ ]:


submission = pd.DataFrame({"id":range(len(sub_predicted_categories)), 
                          "category":sub_predicted_categories})
#submission


# In[ ]:


fn = directory+"/submission-"+runid+".csv"
submission[["id","category"]].to_csv(fn, index=False)
print(f"Saved {fn}")

