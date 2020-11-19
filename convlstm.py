#@title Import { display-mode: "form" }
import numpy as np
import datetime
import tensorflow.compat.v2 as tf
import tensorboard
import os
import argparse
tf.enable_v2_behavior()

import tensorflow_datasets as tfds
import tensorflow_probability as tfp
np.random.seed(43)
tf.random.set_seed(43) 

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

import matplotlib.pyplot as plt
import sys


parser = argparse.ArgumentParser(description="ConvLSTM")
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--validation_split",type=float, default=0.05,help='validation split for traning')
parser.add_argument("--patch_size",type=int, default = 34, help="Size of Patches Need to be extracted from Images")
parser.add_argument("--timestep",type=int, default = 10, help = "#timestep use to create timestep")
parser.add_argument("--input_data",type=str,default = "noisy_data.npy",help="Path to input data")
parser.add_argument("--lr",type=float,default = 0.001,help="Learning rate")
parser.add_argument("--mode",type=int,default = 1,help="if mode==1, train from scratch, if mode==any_thing else use pretained model")
parser.add_argument("--initial_epoch",type=int,default=0,help = "When using a pretranined model, the total number of epochs pn which the model is already trained")

opt = parser.parse_args()


def create_patch_data(data,opt):
    patch_size     = opt.patch_size
    input_shape    = list(data.shape)
    Number_of_frames,timestep,W,H,Channels = input_shape
    assert W==H
    num_patches  = int(np.ceil(W/patch_size))
    pad = patch_size*num_patches - W 
    New_Number_of_frames = Number_of_frames*(num_patches**2)
    New_W = patch_size
    New_H = patch_size
    Patch_data = np.zeros((New_Number_of_frames,timestep,New_W,New_H,Channels))
    for i in range(num_patches):
        for j in range(num_patches):
            if(i==num_patches-1 and j==num_patches-1):
                #print((i*34,((i+1)*34-pad)),(j*34,(j+1)*34-pad),1.1)
                Patch_data[Number_of_frames*(j+num_patches*i):Number_of_frames*((j+num_patches*i)+1),:,:-pad,:-pad,:] = data[:,:,i*patch_size:(i+1)*patch_size-pad,j*patch_size:(j+1)*patch_size-pad,:]
            elif(i==num_patches-1):
                #print((i*34,((i+1)*34-pad)),(j*34,(j+1)*34),1.0)
                Patch_data[Number_of_frames*(j+num_patches*i):Number_of_frames*((j+num_patches*i)+1),:,:-pad,:,:] = data[:,:,i*patch_size:(i+1)*patch_size-pad,j*patch_size:(j+1)*patch_size,:]
            elif(j==num_patches-1):
                #print((i*34,((i+1)*34)),(j*34,(j+1)*34-pad),0.1)
                Patch_data[Number_of_frames*(j+num_patches*i):Number_of_frames*((j+num_patches*i)+1),:,:,:-pad,:] = data[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size-pad,:]
            else:
                #print((i*34,((i+1)*34)),(j*34,(j+1)*34),0.0)
                Patch_data[Number_of_frames*(j+num_patches*i):Number_of_frames*((j+num_patches*i)+1),:,:,:,:] = data[:,:,i*patch_size:(i+1)*patch_size,j*patch_size:(j+1)*patch_size,:]
    return Patch_data


def create_model(opt):
    
    patch_size = opt.patch_size
    seq = tfk.models.Sequential()
    seq.add(tfkl.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       input_shape=(None, patch_size, patch_size, 1),
                       padding='same', return_sequences=True))
    seq.add(tfkl.BatchNormalization())

    seq.add(tfkl.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(tfkl.BatchNormalization())

    seq.add(tfkl.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(tfkl.BatchNormalization())

    seq.add(tfkl.ConvLSTM2D(filters=40, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(tfkl.BatchNormalization())

    seq.add(tfkl.Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))
    
    lr = opt.lr
    optimizer = tfk.optimizers.Adadelta(learning_rate=lr)
    loss_fn = tfk.losses.MeanSquaredError(reduction='sum')
    seq.compile(loss=loss_fn, optimizer=optimizer)
    
    return seq


def train_new(noisy_movies,shifted_movies,opt,flag_save=True,flag_callback=True):

    log_dir = os.path.join(opt.outf,"fit",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    #checkpoint_path = "logs/training/cp-{epoch:04d}.ckpt"
    checkpoint_path = os.path.join(opt.outf,"training","cp-{epoch:04d}.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    # Create a callback that saves the model's weights every 5 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, 
        verbose=1, 
        save_weights_only=True,
        period=5)
    
    model = create_model(opt)
    model.save_weights(checkpoint_path.format(epoch=0))
    batch_size = opt.batchSize
    epochs = opt.epochs
    
    if flag_callback:
        
        model.fit(noisy_movies, shifted_movies, batch_size=batch_size,
                epochs=epochs, validation_split=0.05,
                callbacks=[tensorboard_callback,cp_callback]) #cp_callback
    
    else:    
        model.fit(noisy_movies, shifted_movies, batch_size=batch_size,
            epochs=epochs, validation_split=0.05)   
        
    if flag_save:
        model.save(os.path.join(opt.outf,"my_model")) 
    
    return model


def pretrain(noisy_movies,shifted_movies,opt,flag_save=True,flag_callback=True):
    
    #Load PRETRAINED MODEL    
    checkpoint_path = os.path.join(opt.outf,"training","cp.ckpt")
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model = create_model(opt)
    model.load_weights(latest)
   

    #define model param
    batch_size = opt.batchSize
    epochs = opt.epochs
    initial_epoch = opt.initial_epoch
    
    
    #training 
    if flag_callback:
        
        log_dir = os.path.join(opt.outf,"fit",datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        checkpoint_path = os.path.join(opt.outf,"training","cp-{epoch:04d}.ckpt")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            verbose=1, 
            save_weights_only=True,
        period=5)
        
        model.fit(noisy_movies, shifted_movies, batch_size=batch_size,
                epochs=epochs, validation_split=0.05,
                callbacks=[tensorboard_callback,cp_callback],initial_epoch = initial_epoch) #cp_callback
    
    else:  
        
        model.fit(noisy_movies, shifted_movies, batch_size=batch_size,
            epochs=epochs, validation_split=0.05,initial_epoch = initial_epoch)   
        
    
    if flag_save:
        
        model.save(os.path.join(opt.outf,"my_model")) 
    
    return model


def main(opt):
    #Create Training Data
    data = np.load(opt.input_data)
    data = np.moveaxis(data,0,1)
    data = data[:,:,:,:,np.newaxis]
    print(data.shape)
    timestep = opt.timestep
    #patch_size = opt.patch_size
    Patch_data = create_patch_data(data,opt)
    noisy_movies   = Patch_data[:,:timestep,:,:,:]
    shifted_movies = Patch_data[:,1:timestep+1,:,:,:]
    print(noisy_movies.shape,shifted_movies.shape)

    mode = opt.mode
    if (mode==1):
        Model = train_new(noisy_movies,shifted_movies,opt,True,True)
    else:
        Model = pretrain(noisy_movies,shifted_movies,opt,True,True)
        

if __name__ == "__main__":
    main(opt)
