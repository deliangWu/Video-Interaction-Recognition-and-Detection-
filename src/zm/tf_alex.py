# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorflow.python.framework import ops
import rtime

rtime.Wdatetime('time.txt','start')
X_train,y_train,X_test,y_test=tl.files.load_cifar10_dataset(shape=(-1,32,32,3),plotable=False)
X_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(y_train, dtype=np.int32)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int32)
num_labels=12
# label=y_train.reshape(-1,)
alpha=np.array([0.01],dtype=np.float32)
m=np.array([2*num_labels],dtype=np.float32)

batch_size=200
test_batch_size=100

def loss_func(x):
    n=x.shape[0]
    loss=np.zeros(n)
    for i in range(x.shape[0]-1):
        loss_1=np.sum((x[i,:]-x[(i+1):,:])**2,axis=1)
        loss_2=abs(np.sign(abs(label[i]-label[(i+1):])))
        loss_1_1=np.array([0 if bi<0 else bi for bi in m-loss_1])
        loss[i]=np.sum((1-loss_2)*loss_1+loss_2*loss_1_1)
    return (np.sum(loss)/(n*(n-1))+alpha*np.sum(abs(abs(x)-1))/(n-1)).astype(np.float32)


#loss0=b1-b2
#loss1= b1-b2 l2norm^2
#loss2=y
def loss_grad(x):

    loss_grad=np.zeros(x.shape)
    #y=np.zeros([x.shape[0],x.shape[0]])
    #for i in range(x.shape[0]):
        #y[i,:]=abs(np.sign(abs(label[i]-label)))

    for i in range(x.shape[0]):
        loss_0=x[i,:]-x
        loss_1=np.sum(loss_0**2,axis=1)
        loss_2=abs(np.sign(abs(label[i]-label)))
        yi=loss_2.reshape(-1,1)
        term1=(1-yi)*loss_0
        term1=np.sum(term1,axis=0)
        term2=np.array([(-yi[i]*loss_0[i,:]) if loss_1[i]<m else np.zeros(num_labels) for i in range(x.shape[0])])
        term2=np.sum(term2,axis=0)
        #loss_1_1=np.array([-1 if bi<m else 0 for bi in loss_1])
        x_=np.array([1 if (-1<=bi<=0)or(bi>=1) else -1 for bi in x[i,:]])
        #loss_grad[i,:]=(x.shape[0])*alpha*x_+np.sum((1-loss_2.reshape(-1,1))*loss_0+loss_2.reshape(-1,1)*loss_0*loss_1_1.reshape(-1,1),axis=0)
        loss_grad[i, :] = (x.shape[0]) * alpha * x_ + term1+term2
    return loss_grad.astype(np.float32)


def tf_d_spiky(x, name=None):
    with ops.op_scope( name, 'd_spiky',[x]) as name:
        y=tf.py_func(loss_grad,[x],tf.float32,stateful=False,name=name)
        return y


def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGard' + str(np.random.randint(0, 1e8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc': rnd_name}):
        return tf.py_func(func, inp, Tout, stateful, name)


def loss_gradient(op, grad):
    x=op.inputs[0]
    n_gr = tf_d_spiky(x)
    return grad * n_gr




def tf_loss(x, name=None):
    with ops.op_scope( name, 'spiky',[x]) as name:
        y = py_func(loss_func, [x], tf.float32, name=name,grad=loss_gradient)
        return y[0]

def get_chunk_random(samples,labels,chunkSize,times=1):
    # if len(samples) != len(labels):
    #     raise Exception('Length of samples and labels must equal')
    i=0
    step=int(len(labels)//chunkSize*times)
    while i<=step:
        indexes=np.random.randint(0,len(labels),chunkSize)
        yield i,samples[indexes,:,:,:],labels[indexes]
        i+=1



train_samples=tf.placeholder(dtype=tf.float32,shape=(batch_size,32,32,3),name='train_samples')
train_labels=tf.placeholder(dtype=tf.int64,shape=(batch_size,),name='train_labels')
test_samples=tf.placeholder(dtype=tf.float32,shape=(test_batch_size,32,32,3),name='test_samples')
test_labels=tf.placeholder(dtype=tf.int64,shape=(test_batch_size,),name='test_labels')

def define_model(train_samples,name):
    network=tl.layers.InputLayer(inputs=train_samples,name=name+'input_layers')
    network=tl.layers.Conv2dLayer(network,shape=[11,11,3,96],strides=[1,4,4,1],padding='SAME', \
                                  W_init=tf.contrib.layers.xavier_initializer(),name=name+'conv1')
    network.outputs=tf.nn.relu(network.outputs,name=name+'relu1')
    network.outputs=tf.nn.lrn(network.outputs,depth_radius=2,bias=1.0,alpha=2e-5,beta=0.75,name=name+'norm1')
    network=tl.layers.PoolLayer(network,ksize=[1,3,3,1],strides=[1,2,2,1],pool=tf.nn.max_pool,name=name+'max_pool1')
    network=tl.layers.Conv2dLayer(network,shape=[5,5,96,256],strides=[1,1,1,1],padding='SAME', \
                                  W_init=tf.contrib.layers.xavier_initializer(),name=name+'conv2')
    network.outputs=tf.nn.relu(network.outputs,name=name+'relu2')
    network.outputs=tf.nn.lrn(network.outputs,depth_radius=2,bias=1.0,alpha=2e-5,beta=0.75,name=name+'norm2')
    network=tl.layers.PoolLayer(network,ksize=[1,3,3,1],strides=[1,2,2,1],pool=tf.nn.max_pool,name=name+'max_pool2')
    network=tl.layers.Conv2dLayer(network,shape=[3,3,256,384],strides=[1,1,1,1],padding='SAME', \
                                  W_init=tf.contrib.layers.xavier_initializer(),name=name+'conv3')
    network.outputs=tf.nn.relu(network.outputs,name=name+'relu3')
    network=tl.layers.Conv2dLayer(network,shape=[3,3,384,384],strides=[1,1,1,1],padding='SAME', \
                                  W_init=tf.contrib.layers.xavier_initializer(),name=name+'conv4')
    network.outputs=tf.nn.relu(network.outputs,name=name+'relu4')
    network=tl.layers.Conv2dLayer(network,shape=[3,3,384,256],strides=[1,1,1,1],padding='SAME', \
                                  W_init=tf.contrib.layers.xavier_initializer(),name=name+'conv5')
    network.outputs=tf.nn.relu(network.outputs,name=name+'relu5')
    network=tl.layers.PoolLayer(network,ksize=[1,3,3,1],strides=[1,2,2,1],pool=tf.nn.max_pool,name=name+'max_pool5')
    network = tl.layers.FlattenLayer(network, name=name+'flatten_layer')
    network=tl.layers.DenseLayer(network,n_units=500,W_init=tf.contrib.layers.xavier_initializer(),name=name+'fc6')
    network.outputs=tf.nn.relu(network.outputs,name=name+'relu6')
    network=tl.layers.DenseLayer(network,n_units=500,W_init=tf.contrib.layers.xavier_initializer(),name=name+'fc7')
    network.outputs=tf.nn.relu(network.outputs,name=name+'relu7')
    network=tl.layers.DenseLayer(network,n_units=12,W_init=tf.contrib.layers.xavier_initializer(), name=name+'fc8')
    return network.outputs


y=define_model(train_samples,name='train')
loss=tf_loss(y,name='loss')
learning_rate = tf.train.exponential_decay(0.001,tf.Variable(0)*batch_size,20000,0.6,staircase=True)
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(loss)
#optimizer=tf.train.GradientDescentOptimizer(1e-4).minimize(loss)


train_predictions=y
test_predictions=define_model(test_samples,name='test')
losslist=[]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Start Training')
    with tf.device('/gpu0:'):
        for i, samples, label in get_chunk_random(X_train, y_train, batch_size,times=500):
            _, loss_1, predictions = sess.run(
                        [optimizer, loss, train_predictions],
                        feed_dict={train_samples: samples})
            if i % 100 == 0:
                print('Minibatch loss at step {}: {}'.format(i, loss_1))
                losslist.append(loss_1)
        test_prediction=sess.run(test_predictions,feed_dict={test_samples:test_samples})
    
        # for i, samples,_ in get_chunk_random(test_samples, test_labels, test_batch_size,times=1):
        #     test_predictions_1 = sess.run(
        #                 test_predictions,
        #                 feed_dict={test_samples: samples})
        n_test=y_test.shape[0]
        precision_test=y_test.reshape(-1,1)-y_test
        precision_predict=np.zeros((n_test,n_test))
        for i in range(n_test):
            precision_predict[i,:]=np.linalg.norm(test_prediction[i,:].reshape(1,-1)-test_prediction,axis=1)
        precision_predict[np.where(precision_predict<=2)]=0
        precision_predict[np.where(precision_predict>0)]=1
        precision_test[np.where(precision_test>0)]=1
        ##1代表i和j不相似
        error=np.sum(abs(precision_predict-precision_test))/(n_test*(n_test-1))
        print('mAP:'+error)
        losslist.append(error)
        np.savetxt("losslist_2.txt",losslist,newline='\n')
        rtime.Wdatetime('time.txt','end')
