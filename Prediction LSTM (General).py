import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs

#Import all needed files and combine them into a dataframe
files=glob.glob('Data/Data*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index = True)

#Transform all data into the LSTM's usable form
a=df.A
a=np.asarray(a)
print(a.shape)
a=np.split(a,len(files))
a_data=np.asarray(a).T
print(a_data.shape)

b=df.B
b=np.asarray(b)
b=np.split(b,len(files))
b_data=np.asarray(b).T

c=df.C
c=np.asarray(c)
c=np.split(c,len(files))
c_data=np.asarray(b).T

#plot the data (Just A values for now)
initial_plot=input('Would you like to see the inital data graphed: ')
if initial_plot in ['y','Yes','yes','Y']:
    plt.figure(figsize=(12,7), frameon=False, facecolor='brown', edgecolor='blue')
    plt.subplot(3,1,1)
    plt.title('A Data')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.plot(a_data,label='A Data')
    plt.legend()
    plt.subplot(3,1,2)
    plt.title('B Data')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.plot(b_data,label='B Data')
    plt.legend()
    plt.subplot(3,1,3)
    plt.title('C Data')
    plt.xlabel('Time Step')
    plt.ylabel('Values')
    plt.plot(c_data,label='C Data')
    plt.legend()
    plt.show()
else:
    print('THE GRAPHS WILL BE SKIPPED')
    
 #Create Window definition
def window_data(data, window_size):
    X = []
    y = []
    
    i = 0
    while (i + window_size) <= len(data) - 1:
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        
        i += 1
    assert len(X) ==  len(y)
    return X, y

epoch_input=int(input('How many epochs would you like to run? ' ))
epochs = epoch_input
batch_size = 5
LSTM_letter=input('What column data would you like to predict [A/B/C]: ')

#define all parts of the LSTM
def LSTM_cell(hidden_layer_size, batch_size,number_of_layers, dropout=True, dropout_rate=0.5):
    layer = tf.contrib.rnn.LSTMCell(hidden_layer_size)
    if dropout:
        layer = tf.contrib.rnn.DropoutWrapper(layer, output_keep_prob=dropout_rate)    
    cell = tf.contrib.rnn.MultiRNNCell([layer]*number_of_layers)
    init_state = cell.zero_state(batch_size, tf.float32)
    return cell, init_state
def output_layer(lstm_output, in_size, out_size):
    x = lstm_output[:, -1, :]
    print(x)
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.05), name='output_layer_weights')
    bias = tf.Variable(tf.zeros([out_size]), name='output_layer_bias')
    output = tf.matmul(x, weights) + bias
    return output
def opt_loss(logits, targets, learning_rate, grad_clip_margin):
    losses = []
    for i in range(targets.get_shape()[0]):
        losses.append([(tf.pow(logits[i] - targets[i],2))])    
    loss = tf.reduce_sum(losses)/(2*batch_size)
    #Cliping the gradient loss
    gradients = tf.gradients(loss, tf.trainable_variables())
    clipper_, _ = tf.clip_by_global_norm(gradients, grad_clip_margin)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train_optimizer = optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
    return loss, train_optimizer
class ExpPredictionRNN(object):
    def __init__(self, learning_rate=0.0005 , batch_size=5, hidden_layer_size=2000, number_of_layers=1, 
                 dropout=True, dropout_rate=0.5, number_of_classes=3, gradient_clip_margin=4, window_size=7):   
        self.inputs = tf.placeholder(tf.float32, [batch_size, window_size, 3], name='input_data')
        self.targets = tf.placeholder(tf.float32, [batch_size, 3], name='targets')
        cell, init_state = LSTM_cell(hidden_layer_size, batch_size, number_of_layers, dropout, dropout_rate)
        outputs, states = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=init_state)
        self.logits = output_layer(outputs, hidden_layer_size, number_of_classes)
        self.loss, self.opt = opt_loss(self.logits, self.targets, learning_rate, gradient_clip_margin)

tf.reset_default_graph()
model = ExpPredictionRNN()
session =  tf.Session()
session.run(tf.global_variables_initializer())


if LSTM_letter in ['A','a']:
#shape the data and create test and train variables
    """REMEMBER TO CHANGE TO PREDICT THE FUTURE """
    Xa,ya = window_data(a_data,7)
    Xa_train  = np.array(Xa[:673])
    ya_train = np.array(ya[:673])
    Xa_test = np.array(Xa[673:])
    ya_test = np.array(ya[673:])
    print("X_train size: {}".format(Xa_train.shape))
    print("y_train size: {}".format(ya_train.shape))
    print("X_test size: {}".format(Xa_test.shape))
    print("y_test size: {}".format(ya_test.shape))

    for i in range(epochs):
        traind_scores = []
        ii = 0
        epoch_loss = []
        while(ii + batch_size) <= len(Xa_train):
            Xa_batch = Xa_train[ii:ii+batch_size]
            ya_batch = ya_train[ii:ii+batch_size]        
            o, c, _ = session.run([model.logits, model.loss, model.opt], feed_dict={model.inputs:Xa_batch, model.targets:ya_batch})        
            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
        if (i % 1) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
        
    sup =[]
    for i in range(len(traind_scores)):
        for j in range(len(traind_scores[i])):
            sup.append(traind_scores[i][j])  
    tests = []
    i = 0
    while i+batch_size <= len(Xa_test):    
        o = session.run([model.logits], feed_dict={model.inputs:Xa_test[i:i+batch_size]})
        i += batch_size
        tests.append(o)
    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j]) 
    test_results = []
    for i in range(1314):
        if i >= 674:
            test_results.append(tests_new[i-994])
        else:
            test_results.append([None,None,None])  
            
# make a prediction
    ynew = model.predict(5)
          
    
    test_results=np.asarray(test_results)
    sup=np.asarray(sup)
    a1=a_data[:,0]
    a2=a_data[:,1]
    a3=a_data[:,2]
    a1_tr=test_results[:,0]
    a2_tr=test_results[:,1]
    a3_tr=test_results[:,2]
    sup_a1=sup[:,0]
    sup_a2=sup[:,1]
    sup_a3=sup[:,2]
    plt.figure(figsize=(16,7))
    plt.subplot(3,1,1)
    plt.title('A1 Data vs Predicton')
    plt.plot(a1,label='Original')
    plt.plot(sup_a1,label='Training')
    plt.plot(a1_tr, label=' Prediction')
    plt.legend()
    plt.subplot(3,1,2)
    plt.title('A2 Data vs Prediction')
    plt.plot(a2,label='Original')
    plt.plot(sup_a2,label='Testing')
    plt.plot(a2_tr,label='Prediction')
    plt.legend()
    plt.subplot(3,1,3)
    plt.title('A3 Data vs Prediction')
    plt.plot(a3,label='Original')
    plt.plot(sup_a3,label='Testing')
    plt.plot(a3_tr,label='Prediction')
    plt.legend()
    plt.savefig('epoch_{}_batchsize_{}_learnrate_.0005_hiddensize_2000_a.png'.format(epochs,batch_size))
    plt.show()
    
if LSTM_letter in ['B','b']:
    Xb,yb = window_data(b_data,7)
    Xb_train  = np.array(Xb[:673])
    yb_train = np.array(yb[:673])
    Xb_test = np.array(Xb[673:])
    yb_test = np.array(yb[673:])
    pred = np.empty((500,7,3))
    print("X_train size: {}".format(Xb_train.shape))
    print("y_train size: {}".format(yb_train.shape))
    print("X_test size: {}".format(Xb_test.shape))
    print("y_test size: {}".format(yb_test.shape))

    for i in range(epochs):
        traind_scores = []
        ii = 0
        epoch_loss = []
        while(ii + batch_size) <= len(Xb_train):
            Xb_batch = Xb_train[ii:ii+batch_size]
            yb_batch = yb_train[ii:ii+batch_size]        
            o, c, _ = session.run([model.logits, model.loss, model.opt], feed_dict={model.inputs:Xb_batch, model.targets:yb_batch})        
            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
        if (i % 1) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
    sup =[]
    for i in range(len(traind_scores)):
        for j in range(len(traind_scores[i])):
            sup.append(traind_scores[i][j])  
    tests = []
    i = 0
    while i+batch_size <= len(Xb_test):    
        o = session.run([model.logits], feed_dict={model.inputs:Xb_test[i:i+batch_size]})
        i += batch_size
        tests.append(o)
    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j]) 
    test_results = []
    for i in range(994):
        if i >= 674:
            test_results.append(tests_new[i-994])
        else:
            test_results.append([None,None,None])   
    predict=[]
    i=0
    while i+batch_size <= len(pred):
        o=session.run([model.logits], feed_dict={model.inputs:pred[i:i+batch_size]})
        i += batch_size
        predict.append(o)
    predict_new=[]
    for i in range(len(predict)):
        for j in range(len(predict[i][0])):
            predict_new.append(predict[i][0][j])
            
    plt.figure(figsize=(32,14))
    plt.plot(predict_new)

    test_results=np.asarray(test_results)
    sup=np.asarray(sup)
    b1=b_data[:,0]
    b2=b_data[:,1]
    b3=b_data[:,2]
    b1_t=test_results[:,0]
    b2_t=test_results[:,1]
    b3_t=test_results[:,2]
    sup_b1=sup[:,0]
    sup_b2=sup[:,1]
    sup_b3=sup[:,2]
    plt.figure(figsize=(16,7))
    plt.subplot(3,1,1)
    plt.title('B1 Data vs Predicton')
    plt.plot(b1,label='Original')
    plt.plot(sup_b1,label='Training')
    plt.plot(b1_t, label=' Prediction')
    plt.legend()
    plt.subplot(3,1,2)
    plt.title('B2 Data vs Prediction')
    plt.plot(b2,label='Original')
    plt.plot(sup_b2,label='Testing')
    plt.plot(b2_t,label='Prediction')
    plt.legend()
    plt.subplot(3,1,3)
    plt.title('B3 Data vs Prediction')
    plt.plot(b3,label='Original')
    plt.plot(sup_b3,label='Testing')
    plt.plot(b3_t,label='Prediction')
    plt.legend()
    plt.savefig('epoch_{}_batchsize_{}_learnrate_.0005_hiddensize_2000_b.png'.format(epochs,batch_size))
    plt.show()
    
if LSTM_letter in ['C','c']:
    Xc,yc = window_data(c_data,7)
    Xc_train  = np.array(Xc[:670])
    yc_train = np.array(yc[:670])
    Xc_test = np.array(Xc[670:])
    yc_test = np.array(yc[670:])
    print("X_train size: {}".format(Xc_train.shape))
    print("y_train size: {}".format(yc_train.shape))
    print("X_test size: {}".format(Xc_test.shape))
    print("y_test size: {}".format(yc_test.shape))

    for i in range(epochs):
        traind_scores = []
        ii = 0
        epoch_loss = []
        while(ii + batch_size) <= len(Xc_train):
            Xc_batch = Xc_train[ii:ii+batch_size]
            yc_batch = yc_train[ii:ii+batch_size]        
            o, c, _ = session.run([model.logits, model.loss, model.opt], feed_dict={model.inputs:Xc_batch, model.targets:yc_batch})        
            epoch_loss.append(c)
            traind_scores.append(o)
            ii += batch_size
        if (i % 1) == 0:
            print('Epoch {}/{}'.format(i, epochs), ' Current loss: {}'.format(np.mean(epoch_loss)))
        
    sup =[]
    for i in range(len(traind_scores)):
        for j in range(len(traind_scores[i])):
            sup.append(traind_scores[i][j])  
    tests = []
    i = 0
    while i+batch_size <= len(Xc_test):    
        o = session.run([model.logits], feed_dict={model.inputs:Xc_test[i:i+batch_size]})
        i += batch_size
        tests.append(o)
    tests_new = []
    for i in range(len(tests)):
        for j in range(len(tests[i][0])):
            tests_new.append(tests[i][0][j]) 
    test_results = []
    for i in range(1310):
        if i >= 671:
            test_results.append(tests_new[i-990])
        else:
            test_results.append([None,None,None])   
    
    test_results=np.asarray(test_results)
    sup=np.asarray(sup)
    c1=c_data[:,0]
    c2=c_data[:,1]
    c3=c_data[:,2]
    c1_t=test_results[:,0]
    c2_t=test_results[:,1]
    c3_t=test_results[:,2]
    sup_c1=sup[:,0]
    sup_c2=sup[:,1]
    sup_c3=sup[:,2]
    plt.figure(figsize=(16,7))
    plt.subplot(3,1,1)
    plt.title('C1 Data vs Predicton')
    plt.plot(c1,label='Original')
    plt.plot(sup_c1,label='Training')
    plt.plot(c1_t, label=' Prediction')
    plt.legend()
    plt.subplot(3,1,2)
    plt.title('C2 Data vs Prediction')
    plt.plot(c2,label='Original')
    plt.plot(sup_c2,label='Testing')
    plt.plot(c2_t,label='Prediction')
    plt.legend()
    plt.subplot(3,1,3)
    plt.title('C3 Data vs Prediction')
    plt.plot(c3,label='Original')
    plt.plot(sup_c3,label='Testing')
    plt.plot(c3_t,label='Prediction')
    plt.legend()
    plt.savefig('epoch_{}_batchsize_{}_learnrate_.0005_hiddensize_2000_c.png'.format(epochs,batch_size))
    plt.show()
session.close()
