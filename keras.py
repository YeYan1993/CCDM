# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 10:05:29 2018

@author: yuyangyang
"""
import datetime
import keras 
from keras.preprocessing.image import ImageDataGenerator 
from keras_tqdm import TQDMNotebookCallback 
from keras.models import Sequential 
from keras.layers import Dense 
from keras.layers import Dropout 
from keras.layers import Flatten 
from keras.constraints import maxnorm 
from keras.optimizers import SGD 
from keras.layers.convolutional import Conv2D 
from keras.layers.convolutional import MaxPooling2D 
from keras.utils import np_utils 
from keras.callbacks import Callback


start_1 = datetime.datetime.now()
#创建两个 ImageDataGenerator 对象
batch_size = 32 
 
train_datagen = ImageDataGenerator(rescale=1/255., #归一化
shear_range=0.2, 
zoom_range=0.2, 
horizontal_flip=True 
) 
val_datagen = ImageDataGenerator(rescale=1/255.) 

#基于前面的两个对象，我们接着创建两个文件生成器
train_generator = train_datagen.flow_from_directory( 
'E:\\image project\\dogvscat\\dog_cat_svm\\data\\train\\',                      #E:\\image project\\dogvscat\\dog_vs_cat_optimize\\data\\train_4000\\
#E:\\image project\\dogvscat\\dog_cat_svm\\data\\train_20000\\
target_size=(208, 208), 
batch_size=batch_size, 
class_mode='categorical') 
 
validation_generator = val_datagen.flow_from_directory( 
'E:\\image project\\dogvscat\\dog_cat_svm\\data\\validation\\', 
target_size=(208, 208), 
batch_size=batch_size, 
class_mode='categorical') 
#


 
##Found 20000 images belonging to 2 classes. 
##Found 5000 images belonging to 2 classes. 
 
model = Sequential() 
 
model.add(Conv2D(32, (3, 3), input_shape=(208, 208, 3), 
                 padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
 
model.add(Conv2D(64, (3, 3), activation='relu', padding='same')) 
model.add(MaxPooling2D(pool_size=(2, 2))) 
 
model.add(Dropout(0.25)) 
model.add(Flatten()) 
model.add(Dense(64, activation='relu')) 
model.add(Dropout(0.5)) 
model.add(Dense(2, activation='softmax')) 

epochs = 50 
lrate = 0.01 
decay = lrate/epochs 
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False) 
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy']) 

model.summary()
'''
自动调整学习速率
keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1, 
    patience=10, 
    verbose=0, 
    mode='auto', 
    epsilon=0.0001, 
    cooldown=0, 
    min_lr=0
)

1. monitor：被监测的量
2. factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
3. patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
4. mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
5. epsilon：阈值，用来确定是否进入检测值的“平原区”
6. cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
7. min_lr：学习率的下限
'''
#定义了两个将在训练时调用的回调函数 (callback function)
## Callback for loss logging per epoch 
class LossHistory(Callback): 
    def on_train_begin(self, logs={}): 
        self.losses = [] 
        self.val_losses = [] 
 
    def on_epoch_end(self, batch, logs={}): 
        self.losses.append(logs.get('loss')) 
        self.val_losses.append(logs.get('val_loss')) 
 
history = LossHistory() 
 
'''## Callback for early stopping the training 
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
min_delta=0, 
patience=2, 
verbose=0, mode='auto')
1. monitor：需要监视的量
2. patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），
则经过patience个epoch后停止训练。
3. verbose：信息展示模式
4. mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。
在max模式下，当检测值不再上升则停止训练。
'''

#训练过程
n= 25000
ratio = 0.2
fitted_model = model.fit_generator( train_generator, 
                                   steps_per_epoch= int(n * (1-ratio)) // batch_size, 
                                    epochs=50, 
                                    validation_data=validation_generator, 
                                    validation_steps= int(n * ratio) // batch_size, 
                                    callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True),  history], #callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True), early_stopping, history], 
                                    verbose=1) 
                                
# Save the weights                            
#model.save_weights('E:\\2018.3\\Image_classification\\keras_20000_0319.h5')
#保存神经网络的结构与训练好的参数
json_string = model.to_json()#等价于 json_string = model.get_config()  
open('E:\\2018.3\\Image_classification\\0320_model_architecture.json','w').write(json_string)    
model.save_weights('E:\\2018.3\\Image_classification\\0320_model_weights.h5') 
end_1 = datetime.datetime.now()
print('训练时间： '+str(end_1-start_1))

'''
score = model.evaluate(X_train, Y_train, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])
'''
