print('Loading Modules')
from input_384 import load_images_from_folder_rgb, create_image_folder_dataset
import pandas as pd
import tensorflow as tf
from keras import layers
import keras
import os
from keras.regularizers import L1, L2
from transformers import ViTFeatureExtractor, DefaultDataCollator, TFViTForImageClassification, create_optimizer, TFViTModel
from keras.callbacks import TensorBoard as TensorboardCallback, EarlyStopping
import matplotlib.pyplot as plt
from scipy.spatial import procrustes
import numpy as np
import warnings
import sys 


warnings.filterwarnings('ignore')

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

num_train_epochs = int(sys.argv[1])
train_batch_size = int(sys.argv[2])
eval_batch_size = int(sys.argv[3])
learning_rate = float(sys.argv[4])
weight_decay_rate = float(sys.argv[5])
num_warmup_steps = int(sys.argv[6])
num_dense_layers = int(sys.argv[7])

print(str(num_train_epochs)+'\t'+str(train_batch_size)+'\t'+str(eval_batch_size)+'\t'+str(learning_rate)+'\t'+str(weight_decay_rate)+'\t'+str(num_warmup_steps)+'\t'+str(num_dense_layers))

print("Getting Input")

img_360_rgb = load_images_from_folder_rgb("360 Rocks")
img_120_rgb = load_images_from_folder_rgb("120 Rocks")

flattened_image_rgb = img_360_rgb.reshape(360, -1)

rocks360_ds = create_image_folder_dataset("360 Split 30")

img_class_labels = rocks360_ds.features["label"].names

print(img_class_labels)

rocks360_labels = img_class_labels

# we are also renaming our label col to labels to use `.to_tf_dataset` later

#model_id = "google/vit-base-patch16-224-in21k"

#model_id = 'google/vit-base-patch16-384'

model_id = 'google/vit-base-patch32-384'

output_dir=model_id.split("/")[1]

feature_extractor = ViTFeatureExtractor.from_pretrained(model_id)

# learn more about data augmentation here: https://www.tensorflow.org/tutorials/images/data_augmentation
data_augmentation = keras.Sequential(
    [
        layers.Resizing(feature_extractor.size, feature_extractor.size),
        layers.Rescaling(1./255),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# use keras image data augementation processing
def augmentation(examples):
    # print(examples["img"])
    examples["pixel_values"] = [data_augmentation(image) for image in examples["img"]]
    return examples


# basic processing (only resizing)
def process(examples):
    examples.update(feature_extractor(examples['img'], ))
    return examples

rocks360_ds = rocks360_ds.rename_column("label", "labels")

print("Processing Data")

processed_dataset = rocks360_ds.map(process, batched=True)

test_size=.3333333333333333333333

processed_dataset = processed_dataset.shuffle().train_test_split(test_size=test_size)

id2label = {str(i): label for i, label in enumerate(rocks360_labels)}
label2id = {v: k for k, v in id2label.items()}

fp16=True

# Train in mixed-precision float16
# Comment this line out if you're using a GPU that will not benefit from this
if fp16:
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

# Data collator that will dynamically pad the inputs received, as well as the labels.
data_collator = DefaultDataCollator(return_tensors="tf")

# converting our train dataset to tf.data.Dataset
tf_train_dataset = processed_dataset["train"].to_tf_dataset(
   columns=['pixel_values'],
   label_cols=["labels"],
   shuffle=True,
   batch_size=train_batch_size,
   collate_fn=data_collator)

# converting our test dataset to tf.data.Dataset
tf_eval_dataset = processed_dataset["test"].to_tf_dataset(
   columns=['pixel_values'],
   label_cols=["labels"],
   shuffle=True,
   batch_size=eval_batch_size,
   collate_fn=data_collator)

# create optimizer wight weigh decay
num_train_steps = len(tf_train_dataset) * num_train_epochs
optimizer, lr_schedule = create_optimizer(
    init_lr=learning_rate,
    num_train_steps=num_train_steps,
    weight_decay_rate=weight_decay_rate,
    num_warmup_steps=num_warmup_steps,
)

print('Model Initialization')

# @keras.saving.register_keras_serializable()
# load pre-trained ViT model
model = TFViTModel.from_pretrained(
        model_id,
        id2label=id2label,
        label2id=label2id
)

dense_layer = tf.keras.layers.Dense(8, activation='tanh',name='mds')

classification_layer = tf.keras.layers.Dense(30, activation='softmax',name='classification')

pixel_values = tf.keras.layers.Input(shape=(3,384,384), name='pixel_values', dtype='float32')

x = model.vit(pixel_values)[0]
x = x[:,0,:]
if num_dense_layers!=0:
    for n in (2**p for p in range(num_dense_layers)):
        x = tf.keras.layers.Dense(int(512/n), activation='tanh',name='dense_'+str(n))(x)
mds = dense_layer(x)
classifier = classification_layer(mds)

mds_model = tf.keras.Model(inputs=pixel_values, outputs=mds)
# print(mds_model.layers[-1].get_weights())

new_model = tf.keras.Model(inputs=pixel_values, outputs=classifier)
new_model.compile(optimizer=optimizer,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"))

new_model.summary()

callbacks=[]

callbacks.append(TensorboardCallback(log_dir=os.path.join(output_dir,"logs")))
callbacks.append(EarlyStopping(monitor="val_accuracy",patience=5,restore_best_weights=True,verbose=1))

print('Model Training')

train_results = new_model.fit(
    tf_train_dataset,
    validation_data=tf_eval_dataset,
    callbacks=callbacks,
    epochs=num_train_epochs,
    shuffle=True
)

# print(mds_model.layers[-1].get_weights())

mds_model.save('/N/slate/janandan/Models/mds_30_384_patch32_base_'+str(num_train_epochs)+'_'+str(train_batch_size)+'_'+str(eval_batch_size)+'_'+str(learning_rate)+'_'+str(weight_decay_rate)+'_'+str(num_warmup_steps)+'_'+str(num_dense_layers)+'_'+'.keras')

img_360_rgb = tf.transpose(img_360_rgb, perm=[0, 3, 1, 2])
img_120_rgb = tf.transpose(img_120_rgb, perm=[0, 3, 1, 2])

print('Model Prediction')

activations_train = mds_model.predict(img_360_rgb)

activations_val = mds_model.predict(img_120_rgb)
# print(activations_val)

with open('mds_360.txt','r') as f:
    arr = f.read().strip().split('\n')

mds_360 = []
for i in arr:
    mds_360.append(i.strip().split('  '))

mds_360 = np.array(mds_360)

# print(mds_360)

with open('mds_120.txt','r') as f:
    arr = f.read().strip().split('\n')

mds_120 = []
for i in arr:
    mds_120.append(i.strip().split(' '))

mds_120 = np.array(mds_120)
# print(mds_120)

print('Results')

mtx1, mtx2, disparity_360 = procrustes(mds_360, activations_train)
print("Disparity from MDS 360: "+str(disparity_360))

cnn_human_comp = pd.DataFrame()
cnn_human_comp['Dimension'] = range(1, mtx1.shape[1] + 1)

corr_coeff = np.zeros(mtx1.shape[1])
avg_train_coeff = 0
for i in range(mtx1.shape[1]):
    corr_coeff[i] = np.corrcoef(mtx1[:, i], mtx2[:, i])[0, 1]
    avg_train_coeff += corr_coeff[i]
cnn_human_comp["Train"] = corr_coeff
avg_train_coeff = avg_train_coeff/mtx1.shape[1]

mtx1, mtx2, disparity_120 = procrustes(mds_120, activations_val)
print("Disparity from MDS 120: "+str(disparity_120))

corr_coeff = np.zeros(mtx1.shape[1])
avg_test_coeff = 0
for i in range(mtx1.shape[1]):
    corr_coeff[i] = np.corrcoef(mtx1[:, i], mtx2[:, i])[0, 1]
    avg_test_coeff+= corr_coeff[i]
cnn_human_comp["Test"] = corr_coeff
avg_test_coeff = avg_test_coeff/mtx1.shape[1]

print(cnn_human_comp.set_index('Dimension'))

print('Average Train Coefficient = '+str(avg_train_coeff))

print('Average Test Coefficient = '+str(avg_test_coeff))

with open('30_rocks_results_384_patch32_base.txt','a') as f:
    f.write(str(num_train_epochs)+'\t\t'+str(train_batch_size)+'\t\t'+str(eval_batch_size)+'\t'+str(learning_rate)+'\t'+str(weight_decay_rate)+'\t\t'+str(num_warmup_steps)+'\t\t'+str(num_dense_layers)+'\t\t'+'{:.4f}\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\t\t{:.4f}\n'.format(train_results.history['accuracy'][-6],train_results.history['val_accuracy'][-6],disparity_360,disparity_120,avg_train_coeff,avg_test_coeff))

