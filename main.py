from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense



def classifier():
    cls = Sequential()
    cls.add(Conv2D(32 , (3,3) , input_shape = (64 , 64 ,3) , activation = 'relu'))
    cls.add(MaxPooling2D(pool_size=(2,2))) #used to reduct the chances of overfitting
    
    cls.add(Conv2D(32 , (3, 3), activation = 'relu'))
    cls.add(MaxPooling2D(pool_size=(2,2))) #used to reduct the chances of overfitting
    
    cls.add(Conv2D(32 , (3, 3), activation = 'relu'))
    cls.add(Flatten())
    
    cls.add(Dense(units = 128 , activation = 'relu'))
    cls.add(Dense(units = 1 , activation = 'sigmoid'))
    
    cls.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
    return cls


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True
                                   )
test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('/home/victorh/Git/cnn-facial-recognition/dataset/training_set',
                                                 target_size = (64 , 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary',
                                                 )

test_set = test_datagen.flow_from_directory('/home/victorh/Git/building-cnn/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

result = classifier().fit_generator(training_set,
                            steps_per_epoch = 1000,
                            epochs = 25,
                            validation_data = test_set,
                            validation_steps = 100,
                            workers = 0
                            )

print("History of Losses and acurracy :{}".format(result.history))

'''
o dataset não tem uma variação boa de imagens, que foram
tiradas em tempo e espaco muito proximos um do outro (frames)
o que ocasionou um overffting. Ela estava acertando 100% do test set,
embora o test set tivesse suas diferenças
'''