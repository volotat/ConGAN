from keras.layers import Input, Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt
import keras
import math
import os

import numpy as np


EXAMPLES = 100
LATENT_SPACE = 20
INPUT_SPACE = 48
CHANNELS = 3

INPUT_IMG_SIZE = 200
OUTPUT_IMG_SIZE = 64

DATASET_FILE = 'faces.png'


POSITION_INPUTS = 2
position_input = Input(shape = (POSITION_INPUTS,))
img_ident_input = Input(shape = (EXAMPLES,)) 
latent_input = Input(shape = (LATENT_SPACE,))


class ConGAN():
    def __init__(self):
        optimizer = Adam(0.0004, 0.5, clipnorm = 1)
        opt_small = Adam(0.0002, 0.5, clipnorm = 1) 
        
        inputs_real = [position_input, img_ident_input]
        inputs_fake = [position_input, latent_input]

        #main pieces
        if (not os.path.isfile('generator.h5')):
            img_ident_layer = Dense(LATENT_SPACE, activation='tanh')(img_ident_input) 
            self.ident = Model(img_ident_input, img_ident_layer, name = 'IDENT')
            #plot_model(self.ident, to_file='ident.png', show_shapes=True)
            
            self.generator = self.build_generator()
            #plot_model(self.generator, to_file='generator.png', show_shapes=True)
            
            self.discriminator = self.build_discriminator()
            #plot_model(self.discriminator, to_file='discriminator.png', show_shapes=True)
        else:
            self.discriminator = load_model('discriminator.h5')
            self.generator = load_model('generator.h5')
            self.ident = load_model('ident.h5')
        
        
        
        self.ident.trainable = True
        self.generator.trainable = True
        self.generator.compile(loss='mse', optimizer=optimizer)
        self.discriminator.trainable = False
        
        self.generator_real_t = self.generator([position_input, self.ident([img_ident_input])])[0] #Train ident -> pixel as normal model
        self.generator_real = Model(inputs_real, self.generator_real_t, name = 'generator_real')
        self.generator_real.compile(loss='mse', optimizer=optimizer)
        #plot_model(self.generator_real, to_file='generator_real.png', show_shapes=True)

        self.generator_fake_t = self.discriminator(self.generator(inputs_fake)[1])   #Train noise -> 1 on discriminator
        self.generator_fake = Model(inputs_fake, self.generator_fake_t, name = 'generator_fake')
        self.generator_fake.compile(loss='binary_crossentropy', optimizer=opt_small)
        #plot_model(self.generator_fake, to_file='generator_fake.png', show_shapes=True)
        
        
        
        
        self.ident.trainable = False
        self.generator.trainable = False
        self.discriminator.trainable = True

        self.discriminator_real_t = self.discriminator(self.generator([position_input, self.ident([img_ident_input])])[1])   #Train discriminator assign ident -> 1
        self.discriminator_real = Model(inputs_real, self.discriminator_real_t, name = 'discriminator_real')
        self.discriminator_real.compile(loss='binary_crossentropy', optimizer=opt_small)
        #plot_model(self.discriminator_real, to_file='discriminator_real.png', show_shapes=True)

        
        self.discriminator_fake_t = self.discriminator(self.generator(inputs_fake)[1])   #Train discriminator assign noise -> 0
        self.discriminator_fake = Model(inputs_fake, self.discriminator_fake_t, name = 'discriminator_fake')
        self.discriminator_fake.compile(loss='binary_crossentropy', optimizer=opt_small)
        #plot_model(self.discriminator_fake, to_file='discriminator_fake.png', show_shapes=True)
        

    # Do not use Batch Normalization anywhere, it will be harmful for discriminator ability 
    # to distinguish good and bad samples, and as a result it will break the generator
    
    def build_generator(self):
        position_layer = Dense(INPUT_SPACE)(position_input) 
        position_layer = LeakyReLU(alpha=0.2)(position_layer) 
        
        #Head layer contains understanding of object we want to create
        head_layer = keras.layers.concatenate([position_layer, latent_input])
        
        head_layer = Dense(256)(head_layer) 
        head_layer = LeakyReLU(alpha=0.2)(head_layer) 
        head_layer = Dense(512)(head_layer) 
        head_layer = LeakyReLU(alpha=0.2)(head_layer) 
        
        head_layer = Dense(1024)(head_layer) 
        head_layer = LeakyReLU(alpha=0.2)(head_layer) 
        head_layer = Dense(1024)(head_layer) 
        head_layer = LeakyReLU(alpha=0.2)(head_layer) 
        head_layer = Dense(1024)(head_layer) 
        head_layer = LeakyReLU(alpha=0.2)(head_layer) 
        head_layer = Dense(1024)(head_layer) 
        head_layer = LeakyReLU(alpha=0.2)(head_layer) 
        
        
        #Draw part predict color in concrete spot
        draw_layer = head_layer
        draw_layer = Dense(256)(draw_layer)
        draw_layer = LeakyReLU(alpha=0.2)(draw_layer)
        draw_layer = Dense(32)(draw_layer)
        draw_layer = LeakyReLU(alpha=0.2)(draw_layer)
        
        draw_layer = Dense(CHANNELS, activation= 'linear')(draw_layer)

        layer_cnc = keras.layers.concatenate([head_layer, position_layer, draw_layer]) 
        return Model([position_input, latent_input], [draw_layer, layer_cnc], name = 'GENERATOR')
        
    def build_discriminator(self):
        input = Input(shape = (1075,))
        
        layer = input
        layer = Dense(512)(layer) 
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(128)(layer) 
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(32)(layer) 
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(8)(layer) 
        layer = LeakyReLU(alpha=0.2)(layer)
        layer = Dense(1, activation='sigmoid')(layer) 
        
        return Model(input, layer, name = 'DISCRIMINATOR')

        
    def choose_rnd_data(self, batch_size):
        idx = np.random.randint(0, self.train_chan.shape[0], batch_size)
            
        imgs_grid = self.train_grid[idx] 
        imgs_nois = self.noise_for_true[idx]
        imgs_chan = self.train_chan[idx] 
        
        return imgs_grid, imgs_nois, imgs_chan
    
    def train(self, epochs, batch_size=128, save_interval=50):
        part_batch = int(batch_size / 2)
        
        zeros = np.zeros((batch_size, 1))
        ones = np.ones((batch_size, 1))
        stack = np.vstack((zeros[:part_batch], ones[:part_batch]))
        
        main_loss = 0
        for epoch in range(epochs):
        
            imgs_grid, imgs_nois, imgs_chan = self.choose_rnd_data(batch_size)
            main_loss += self.generator_real.train_on_batch([imgs_grid, imgs_nois], imgs_chan)
           
            imgs_grid, imgs_nois, imgs_chan = self.choose_rnd_data(batch_size)
            imgs_nois_rnd = np.random.normal(0,1,(batch_size, LATENT_SPACE))
            imgs_nois_rnd[part_batch:] = self.ident.predict(imgs_nois[part_batch:])
            
            self.discriminator_fake.train_on_batch([imgs_grid, imgs_nois_rnd], stack)
            
            
            imgs_grid, imgs_nois, imgs_chan = self.choose_rnd_data(batch_size)
            imgs_nois_rnd = np.random.normal(0,1,(batch_size, LATENT_SPACE))
            self.generator_fake.train_on_batch([imgs_grid, imgs_nois_rnd], ones)
            
            if epoch % 50 == 49:
                print (epoch + 1, main_loss / 50.)
                main_loss = 0
            
            if epoch % save_interval == save_interval - 1:
                self.save_imgs(epoch + 1)
                self.save_models(epoch + 1)

                
    def save_models(self, epoch):
        path = 'saved_model/'+str(epoch)
        self.ident.save(path+'_ident.h5')
        self.generator.save(path+'_generator.h5')
        self.discriminator.save(path+'_discriminator.h5')
        
    def create_grid(self, half_size):
        X,Y = np.mgrid[-half_size:half_size,-half_size:half_size] + 0.5
        grid = np.vstack((X.flatten(), Y.flatten())).T / half_size
        
        # This part of code is highly unnecessary, but might be helpful if you have some 
        # kind of rotation diversity in your data
        
        '''
        #adding ability to construct rotation dependency
        ref_points = np.array([[-1,-1], [1, 1], [-1, 1], [1, -1]])
        sz = grid.shape[0]
        add = np.empty((sz, 0))
        for ref in ref_points:
            grid_ = grid - ref
            r = np.linalg.norm(grid_, axis = 1)
            phi = np.arctan2(grid_[:,1], grid_[:,0]) 
            add = np.concatenate((add, r.reshape(sz,1), np.sin(phi).reshape(sz,1), np.cos(phi).reshape(sz,1)), axis = 1) 
        
        grid = np.concatenate((grid, add), axis = 1)
        '''
        return grid
    
    
    def save_imgs(self, epoch):
        out_size = OUTPUT_IMG_SIZE #Size of single image in output image set
        out_half_size = math.floor(out_size / 2)

        grid = self.create_grid(out_half_size)
        

        im_out = Image.new("RGB", (out_size * 3, out_size * 3))
        fnt = ImageFont.truetype("arial.ttf", 10)
        

        for i in range(9):
            
            if i<3:
                c_noise = np.eye(EXAMPLES)[np.random.choice(EXAMPLES, 1)]
                c_noise = c_noise.reshape(1, EXAMPLES)
                c_noise = np.repeat(c_noise, grid.shape[0], axis=0)
            
                predicted = self.generator_real.predict([grid, c_noise])
                val = self.discriminator_real.predict([grid, c_noise])
            else:
                c_noise = np.random.normal(0,1,(LATENT_SPACE))
                
                c_noise = c_noise.reshape(1, LATENT_SPACE)
                c_noise = np.repeat(c_noise, grid.shape[0], axis=0)
                
                predicted = self.generator.predict([grid, c_noise])[0]
                val = self.discriminator_fake.predict([grid, c_noise])
            
            predicted = np.clip(predicted *  255., 0, 255) 
            predicted = (predicted).astype(np.uint8).reshape(out_size, out_size, CHANNELS) 
            im = Image.fromarray(predicted)
            
            
            val = np.average(val)
            d = ImageDraw.Draw(im)
            d.text((10,10), "{:.2f}".format(val) , font=fnt, fill=(255,255,255,255))  
            d.text((11,11), "{:.2f}".format(val) , font=fnt, fill=(0,0,0,255))  
            
            im_out.paste(im, (i % 3 * out_size, math.floor(i / 3) * out_size))
            print ('Img: ', i + 1)
            
        im_out.save("images/out_%d.png" % epoch)
        

    def prepare_data(self, image_container, size = 200):	
        half_size = math.floor(size / 2)
        
        im_arr = np.zeros((EXAMPLES, size, size, CHANNELS))
        im_set = Image.open(image_container).convert('RGB')
        print ('size: ', im_set.size)
        im_set = im_set.resize(np.array(im_set.size))
        wh = (im_set.size[0] / size, im_set.size[1] / size)
        
        for i in range(EXAMPLES):
            w = i % wh[0] * size
            h = math.floor (i / wh[0]) * size
            im = im_set.crop((w, h, w + size, h + size))	
            im_arr[i] = np.array(im).reshape(size, size, CHANNELS) / 255.

        im_arr = im_arr.reshape(EXAMPLES * size * size, CHANNELS)
        print ('im_arr shape:', im_arr.shape)	

        grid = self.create_grid(half_size)
        print ('grid shape:', grid.shape)	
        
        grid_ = grid.reshape(1, (half_size * 2) ** 2, POSITION_INPUTS)
        grid_ = np.repeat(grid_, EXAMPLES, axis=0).reshape(EXAMPLES * size * size, POSITION_INPUTS)
        print ('grid_ shape:', grid_.shape)	
        
        lat_arr = np.zeros((EXAMPLES, EXAMPLES))
        pnt = np.arange(EXAMPLES)
        lat_arr[pnt, pnt] = 1 #one hot matrix for representing pictures		
        
        lat_arr = lat_arr.reshape(EXAMPLES, 1, EXAMPLES)
        noise = np.repeat(lat_arr, size * size, axis=1).reshape(EXAMPLES * size * size, EXAMPLES)
        print ('noise shape:', noise.shape)	
        
        self.noise_for_true = noise
        self.train_chan = im_arr
        self.train_grid = grid_

	

if __name__ == '__main__':
    gan = ConGAN()
    gan.prepare_data(DATASET_FILE, size = INPUT_IMG_SIZE)
    gan.train(epochs=100000, batch_size=2 ** 10, save_interval=500)
