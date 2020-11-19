# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import glob
import os
import sys
from PIL import Image

# %% [markdown]
# ## Load data

# %%
masks = glob.glob("**/whales/*.png")
orgs = list(map(lambda x: x.replace(".png", ".jpg"), masks))


# %%
imgs_list = []
masks_list = []
for image, mask in zip(orgs, masks):
    imgs_list.append(np.array(Image.open(image).resize((384,384))))
    masks_list.append(np.array(Image.open(mask).resize((384,384))))

imgs_np = np.asarray(imgs_list)
masks_np = np.asarray(masks_list)


# %%
print(imgs_np.shape, masks_np.shape)

# %% [markdown]
# ## Plot images + masks + overlay (mask over original)

# %%
from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=imgs_np, mask_imgs=masks_np, nm_img_to_plot=10, figsize=6)

# %% [markdown]
# ## Get data into correct shape, dtype and range (0.0-1.0)

# %%
print(imgs_np.max(), masks_np.max())


# %%
x = np.asarray(imgs_np, dtype=np.float32)/255
y = np.asarray(masks_np, dtype=np.float32)/255


# %%
print(x.max(), y.max())


# %%
print(x.shape, y.shape)


# %%
y = y.reshape(y.shape[0], y.shape[1], y.shape[2], 1)
print(x.shape, y.shape)

# %% [markdown]
# ##  Train/val split

# %%
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.9, random_state=0)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_val: ", x_val.shape)
print("y_val: ", y_val.shape)

# %% [markdown]
# ## Prepare train generator with data augmentation

# %%
from keras_unet.utils import get_augmented

train_gen = get_augmented(
    x_train, y_train, batch_size=2,
    data_gen_args = dict(
        rotation_range=5.,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=40,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='constant'
    ))


# %%
sample_batch = next(train_gen)
xx, yy = sample_batch
print(xx.shape, yy.shape)
from keras_unet.utils import plot_imgs

plot_imgs(org_imgs=xx, mask_imgs=yy, nm_img_to_plot=2, figsize=6)

# %% [markdown]
# ## Initialize network

# %%
from keras_unet.models import custom_unet

input_shape = x_train[0].shape

model = custom_unet(
    input_shape,
    filters=32,
    use_batch_norm=True,
    dropout=0.3,
    dropout_change_per_layer=0.0,
    num_layers=4
)


# %%
model.summary()

# %% [markdown]
# ## Compile + train

# %%
from tensorflow.keras.callbacks import ModelCheckpoint


model_filename = 'segm_model_v3.h5'
callback_checkpoint = ModelCheckpoint(
    model_filename, 
    verbose=1, 
    monitor='val_loss', 
    save_best_only=True,
)


# %%
from tensorflow.keras.optimizers import Adam, SGD
from keras_unet.metrics import iou, iou_thresholded
from keras_unet.losses import jaccard_distance

model.compile(
    optimizer=Adam(), 
    #optimizer=SGD(lr=0.01, momentum=0.99),
    loss='binary_crossentropy',
    #loss=jaccard_distance,
    metrics=[iou, iou_thresholded]
)


# %%
history = model.fit_generator(
    train_gen,
    steps_per_epoch=10,
    epochs=1,
    
    validation_data=(x_val, y_val),
    callbacks=[callback_checkpoint]
)

# %% [markdown]
# ## Plot training history

# %%
from keras_unet.utils import plot_segm_history

# plot_segm_history(history)

# %% [markdown]
# ## Plot original + ground truth + pred + overlay (pred on top of original)

# %%
from os import path
model.save(path.curdir)
model.load_weights(model_filename)
y_pred = model.predict(x_val)


# %%
from keras_unet.utils import plot_imgs
import cv2
for i,preds in enumerate(y_pred):
    prd = np.dstack([preds, preds, preds])
    prd = (prd * 255).astype(np.uint8)
    cv2.imwrite(f'{i}.png',prd)
plot_imgs(org_imgs=x_val, mask_imgs=y_val, pred_imgs=y_pred, nm_img_to_plot=10)


# %%



