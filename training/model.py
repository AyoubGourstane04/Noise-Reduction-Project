import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Cropping2D, ZeroPadding2D

def build_unet(input_shape=(257, 256, 1)):
    inputs = tf.keras.Input(shape=input_shape)
    
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)    
    
    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(64, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(128, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    
        
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(256, (3,3), activation='relu', padding='same')(c4)
    
    u5 = UpSampling2D((2,2))(c4)
    u5 = Concatenate()([u5, c3])
    c5 = Conv2D(128, (3,3), activation='relu', padding='same')(u5)
    c5 = Conv2D(128, (3,3), activation='relu', padding='same')(c5)

    u6 = UpSampling2D((2,2))(c5)
    u6 = Cropping2D(cropping=((0,1), (0,0)))(u6) if u6.shape[1] != c2.shape[1] else u6
    u6 = Concatenate()([u6, c2])
    c6 = Conv2D(64, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3,3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2,2))(c6)
    u7 = ZeroPadding2D(padding=((0, 1), (0, 0)))(u7)
    u7 = Concatenate()([u7, c1])
    c7 = Conv2D(32, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3,3), activation='relu', padding='same')(c7)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c7)
    
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model


if __name__ == "__main__":
    model = build_unet()
    model.summary()