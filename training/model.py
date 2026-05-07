import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Cropping2D, ZeroPadding2D, BatchNormalization, Activation

def conv2d_block(input_tensor, n_filters, kernel_size=3):
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               padding='same', use_bias=False)(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), 
               padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def build_unet(input_shape=(257, 256, 1)):
    inputs = Input(shape=input_shape)

    c1 = conv2d_block(inputs, 32)
    p1 = MaxPooling2D((2,2))(c1)    
    
    c2 = conv2d_block(p1, 64)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = conv2d_block(p2, 128)
    p3 = MaxPooling2D((2,2))(c3)

    c4 = conv2d_block(p3, 256)

    u5 = UpSampling2D((2,2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = conv2d_block(u5, 128)

    u6 = UpSampling2D((2,2))(c5)

    u6 = Cropping2D(cropping=((0,1), (0,0)))(u6) if u6.shape[1] != c2.shape[1] else u6
    u6 = Concatenate()([u6, c2])
    c6 = conv2d_block(u6, 64)
    
    u7 = UpSampling2D((2,2))(c6)
    u7 = ZeroPadding2D(padding=((0, 1), (0, 0)))(u7)
    u7 = Concatenate()([u7, c1])
    c7 = conv2d_block(u7, 32)
    

    outputs = Conv2D(1, (1,1), activation='sigmoid')(c7)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model

if __name__ == "__main__":
    model = build_unet()
    model.summary()