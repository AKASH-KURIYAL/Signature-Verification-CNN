from tensorflow.keras import layers, models ,Input
from tensorflow.keras.optimizers import Adam


def create_cnn_model(input_shape,num_individuals):
    input_img = Input(shape=input_shape)    #Input Layer of shape(128,128,1)


    #Hidden Layers
    x = layers.Conv2D(32,(3,3),activation='relu')(input_img)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(64,(3,3),activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(128,(3,3),activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Conv2D(256,(3,3),activation='relu')(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Flatten()(x)     #flatten the layer from 2D to 1D

    #fully conncected neurons
    x = layers.Dense(256,activation='relu')(x)
    
    #prevents overfitting by randomly setting 50% of feature to 0
    x = layers.Dropout(0.5)(x)


    output_real_forged = layers.Dense(1,activation='sigmoid',name='real_forged')(x)
    output_individual = layers.Dense(num_individuals,activation='softmax',name='individual')(x)

    model = models.Model(inputs = input_img,outputs=[output_real_forged,output_individual])
    
    optimizer = Adam(learning_rate = 0.001)
    model.compile(optimizer=optimizer,
                  loss={'real_forged':'binary_crossentropy','individual':'categorical_crossentropy'},
                  metrics={'real_forged':'accuracy','individual':'accuracy'})
    
    return model
