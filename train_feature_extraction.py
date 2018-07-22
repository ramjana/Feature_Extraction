import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
from sklearn.utils import shuffle
import time
from sklearn.preprocessing import LabelBinarizer

# TODO: Load traffic signs data.
with open('./train.p', mode='rb') as f:
	datasets = pickle.load(f)

X_train, y_train = datasets['features'], datasets['labels']
X_train,X_test,y_train,y_test  = train_test_split(X_train,y_train,test_size=0.75, random_state=83229) 
X_train,X_test,y_train,y_test  = train_test_split(X_train,y_train,test_size=0.25, random_state=83229) 

Label_binarizer = LabelBinarizer()
y_train_ohc = Label_binarizer.fit_transform(y_train)
y_test_ohc  = Label_binarizer.fit_transform(y_test)

nb_classes = 43
BATCH_SIZE=128
learning_rate = 0.001
sigma=0.01
mu=0
epochs=10

x = tf.placeholder(tf.float32,[None,32,32,3])
y = tf.placeholder(tf.float32,[None,43])
x_resized = tf.image.resize_images(x,[227,227])

fc7 = AlexNet(x_resized, feature_extract=True)
fc7 = tf.stop_gradient(fc7)
shape = (fc7.get_shape().as_list()[-1],nb_classes)
fc8_wt = tf.Variable(tf.truncated_normal(shape=shape,stddev=sigma,mean=mu))
fc8_b = tf.Variable(tf.constant(0.05,shape=[nb_classes]))

logits  = tf.add(tf.matmul(fc7,fc8_wt),fc8_b)


cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y)
loss = tf.reduce_mean(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
prediction = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

def eval_model(X_data,y_data):
    num_examples = len(X_data)
    total_acc = 0
    total_loss =0
    for offset in range(0,num_examples,BATCH_SIZE):
	    batch_feature, batch_label = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
	   l,accu= sess.run([loss,accuracy], feed_dict = {x: batch_feature, y: batch_label})
	    total_loss += l*batch_feature.shape[0]
	    total_acc += accu*batch_feature.shape[0]
	    
    return total_accuracy/num_examples, total_loss/num_examples 	    

test_loss = 0
test_accu = 0
with tf.Session() as sess:

	sess.run(init)

	for epoch_itr in range(epochs):
	    X_train, y_train = shuffle(X_train,y_train)
	    t0 = time.time()
	    for  data_offset in range(0,X_train.shape[0],BATCH_SIZE):
		    batch_feature, batch_label = X_train[data_offset:data_offset+BATCH_SIZE], y_train_ohc[data_offset:data_offset+BATCH_SIZE]
		    sess.run(optimizer, feed_dict={x: batch_feature, y: batch_label})
            		    
	    test_accu, test_loss = eval_model(X_test,y_test_ohc)

	    print("Epoch",epoch_itr+1)
	    print("Time: %.3f seconds" %(time.time() - t0))
	    print("Validation Accuracy=",test_accu)
	    print("Validation loss    =", test_loss)
	    print("")

