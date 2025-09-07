import tensorflow as tf
from tensorflow.keras  import layers, Model, Input

def create_rba_fe_model(input_shape, num_classes):
    """创建RBA-FE模型"""
    inputs = Input(shape=input_shape)
    
    def tcnn_block(x, filters, kernel_size):
        x = layers.Conv2D(filters, kernel_size, padding='same')(x) 
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    conv_outputs = []
    for i in range(5):
        x = tcnn_block(inputs, 64, (3,3))  
        x = tcnn_block(x, 64, (3,3))
        x = layers.GlobalAveragePooling2D()(x)
        conv_outputs.append(x) 
        
    x = layers.Concatenate()(conv_outputs) 
    
    x = layers.Reshape((1,-1))(x)
    
    x = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
    x = layers.LayerNormalization()(x)
    
    x = ARSLIFBiLSTM(128)(x) 
    x = layers.Dropout(0.5)(x)
      
    x = layers.GlobalAveragePooling1D()(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    
    return model