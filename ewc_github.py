# -*- coding: utf-8 -*-
"""
Method of computing Fisher information taken from: "Progress & Compress: A scalable framework for continual learning".
Other references: "Three scenarios for continual learning" ;
"New Insights and Perspectives on the Natural Gradient
Method";
"Limitations of the Empirical Fisher Approximation For Natural Gradient Descent"
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from load_process_mnist_data_flattened_permuted import *



input_vec=tf.keras.Input(shape=input_shape)
tmp=layers.Dense(1000, activation="relu", name='fc1')(input_vec)
tmp2=layers.Dense(1000, activation="relu", name='fc1')(tmp)
tmp3=layers.Dense(num_classes, name='fc2')(tmp)
model= tf.keras.Model(inputs=input_vec, outputs=tmp3)

model0=tf.keras.models.clone_model(model)
model1=tf.keras.models.clone_model(model)
model2=tf.keras.models.clone_model(model)
model3=tf.keras.models.clone_model(model)
model4=tf.keras.models.clone_model(model)
model5=tf.keras.models.clone_model(model)
model6=tf.keras.models.clone_model(model)
model7=tf.keras.models.clone_model(model)
model8=tf.keras.models.clone_model(model)
model9=tf.keras.models.clone_model(model)


class ewc:
    def __init__(self, model):
        #self.lambda_term=lambda_term
        self.model =model
        self.weights_old= self.model.trainable_weights
        self.reg_terms=tf.Variable(0, dtype=tf.float32)
        self.num_samples=500
        self.fisher=[]
        self.weights_old=[]
        self.gamma=.9

        
    def update_weights(self):
        self.weights_old=self.model.trainable_weights

    def update_fisher(self, x):
        F=[]
        grads=[]
        num=min(self.num_samples, x.shape[0])
        indices = tf.range(start=0, limit=num, dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        for i in range(num):
            print('Fisher sample {} out of {}'.format(i, num) )
            with tf.GradientTape() as tape:
                x_sample=x[shuffled_indices[i]]
                x_sample=tf.expand_dims(x_sample,0)
                mod_output = model(x_sample, training=True)
                log_probs=tf.math.log(tf.nn.softmax(mod_output))
            grads= tape.jacobian(log_probs, model.trainable_weights)
            if len(F)==0:
                F=grads
            else:
                F=[ F[i]  +grads[i]**2 for i in range(0,len(grads))  ]
          
        F=[F[i]/num for i in range(len(F))]
        #F=[F[i]**2 for i in range(len(F))]
       
        
        
        if len(self.fisher)==0:
            self.fisher=F
        else:
            self.fisher= [self.gamma* self.fisher[i] + F[i] for i in range(len(F))]
            

    def compute_loss(self, x, y, mod_output, lambda_term):
        if lambda_term !=0:
            for i in range(len(self.fisher)):
                self.reg_terms.assign_add(tf.cast(self.reg_terms+tf.math.reduce_sum( tf.math.multiply(self.fisher[i],  (self.weights_old[i]- self.model.trainable_weights[i]) )**2), tf.float32))
            return (ce(y, mod_output) + tf.cast(lambda_term*self.reg_terms, tf.float32))
        else:
            return ce(y, mod_output)


##################################
ewc_object=ewc(model)
acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()
optimizer = keras.optimizers.SGD(learning_rate=1*1e-3)
ce=keras.losses.CategoricalCrossentropy(from_logits=True)



@tf.function
def train_step(model, x, y,lambda_term):
    with tf.GradientTape() as tape:
        mod_output = model(x, training=True)
        loss_value =  ewc_object.compute_loss(x,y, mod_output, lambda_term)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    acc_metric.update_state(y, mod_output)
    ewc_object.update_weights()
    return loss_value

@tf.function
def test_step(model, x, y):
    val = model(x, training=False)
    val_acc_metric.update_state(y, val)


def train(model,epochs,current_train_dataset, current_val_dataset,lambda_term=0):
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
         
        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(current_train_dataset):
            loss_value = train_step(model,x_batch_train, y_batch_train, lambda_term)
    
            # Log every 200 batches.
            if step % 200 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * batch_size))
    
        # Display metrics at the end of each epoch.
        train_acc = acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))
    
        # Reset training metrics at the end of each epoch
        acc_metric.reset_states()
    
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in current_val_dataset:
            test_step(model,x_batch_val, y_batch_val)
    
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print("Validation acc: %.4f" % (float(val_acc),))
        print("Time taken: %.2fs" % (time.time() - start_time))


def correctly_classified(some_model, thedataset_x, thedataset_y):
    good_set=set()
    for i in range(0, len(thedataset_x)):
        el=thedataset_x[i]
        el=np.expand_dims(el, 0)
        y=tf.math.argmax(some_model(el),axis=1) #for one-hot endoced data
        y=y.numpy()[0]  #for one-hot endoced data
        if y==tf.math.argmax(thedataset_y[i]): # y==y_test[i]
            good_set.add(i)
    return good_set



lambda_term=1000

#learn class 0
train(model,5, train_dataset_0 , val_dataset_0)
model0.layers[0].set_weights(model.layers[0].get_weights())
model0.layers[1].set_weights(model.layers[1].get_weights())
model0.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_0)
len(correctly_classified(model, x_val_0, y_val))/len(x_val_0)


#learn class 1
train(model,3, train_dataset_1 , val_dataset_1,lambda_term)
model1.layers[0].set_weights(model.layers[0].get_weights())
model1.layers[1].set_weights(model.layers[1].get_weights())
model1.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_1)
len(correctly_classified(model, x_val_0, y_val))/len(x_val_0)


#learn class 2
train(model,5, train_dataset_2 , val_dataset_2,lambda_term)
model2.layers[0].set_weights(model.layers[0].get_weights())
model2.layers[1].set_weights(model.layers[1].get_weights())
model2.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_2)



#learn class 3
train(model,4, train_dataset_3 , val_dataset_3,lambda_term)
model3.layers[0].set_weights(model.layers[0].get_weights())
model3.layers[1].set_weights(model.layers[1].get_weights())
model3.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_3)


#learn class 4
train(model,4,train_dataset_4 , val_dataset_4,lambda_term)
model4.layers[0].set_weights(model.layers[0].get_weights())
model4.layers[1].set_weights(model.layers[1].get_weights())
model4.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_4)


#learn class 5
train(model,5, train_dataset_5 , val_dataset_5,lambda_term)
model5.layers[0].set_weights(model.layers[0].get_weights())
model5.layers[1].set_weights(model.layers[1].get_weights())
model5.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_5)


#learn class 6
train(model,4, train_dataset_6 , val_dataset_6,lambda_term)
model6.layers[0].set_weights(model.layers[0].get_weights())
model6.layers[1].set_weights(model.layers[1].get_weights())
model6.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_6)


#learn class 7
train(model,5, train_dataset_7 , val_dataset_7,lambda_term)
model7.layers[0].set_weights(model.layers[0].get_weights())
model7.layers[1].set_weights(model.layers[1].get_weights())
model7.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_7)

#learn class 8
train(model,5, train_dataset_8 , val_dataset_8,lambda_term)
model8.layers[0].set_weights(model.layers[0].get_weights())
model8.layers[1].set_weights(model.layers[1].get_weights())
model8.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_8)


#learn class 9
train(model,4, train_dataset_9 , val_dataset_9,lambda_term)
model9.layers[0].set_weights(model.layers[0].get_weights())
model9.layers[1].set_weights(model.layers[1].get_weights())
model9.layers[2].set_weights(model.layers[2].get_weights())
ewc_object.update_fisher(x_train_9)




#model9
total=len(correctly_classified(model9, x_val_0, y_val))
total=total+len(correctly_classified(model9, x_val_1, y_val))
total=total+len(correctly_classified(model9, x_val_2, y_val))
total=total+len(correctly_classified(model9, x_val_3, y_val))
total=total+len(correctly_classified(model9, x_val_4, y_val))
total=total+len(correctly_classified(model9, x_val_5, y_val))
total=total+len(correctly_classified(model9, x_val_6, y_val))
total=total+len(correctly_classified(model9, x_val_7, y_val))
total=total+len(correctly_classified(model9, x_val_8, y_val))
total=total+len(correctly_classified(model9, x_val_9, y_val))
avg_acc=total/(10*len(x_val))


print(avg_acc)