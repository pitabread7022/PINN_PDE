
"""
@author: Maziar Raissi
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib
from latex import build_pdf
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
from plotting import newfig, savefig
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.random.seed(1234)
tf.set_random_seed(1234)
class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_p, p, X_f, layers, lb, ub, alpha):
        
        self.lb = lb
        self.ub = ub 
    
        self.X_p = X_p[:,0:1]
        self.t_p = X_p[:,1:2]
        
        self.x_f = X_f[:,0:1]
        self.t_f = X_f[:,1:2]
        
        self.p = p
        
        self.layers = layers
        self.alpha= alpha
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True)) ##tf.ConfigProto
        
        self.X_p_tf = tf.placeholder(tf.float32, shape=[None, self.X_p.shape[1]])
        self.t_p_tf = tf.placeholder(tf.float32, shape=[None, self.t_p.shape[1]])        
        self.p_tf = tf.placeholder(tf.float32, shape=[None, self.p.shape[1]])
        
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])        
                
        self.p_pred = self.net_p(self.X_p_tf, self.t_p_tf) 
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)         
        
        self.loss = tf.reduce_mean(tf.square(self.p_tf - self.p_pred)) +  tf.reduce_mean(tf.square(self.f_pred))
               
                
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

                
    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32) ##??
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0 
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))   ##Activation Function
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_p(self, x, t):
        p = self.neural_net(tf.concat([x,t],1), self.weights, self.biases)
        return p
    
#Acoustic Equation

    def net_f(self, x,t):
        p = self.net_p(x,t)
        p_t = tf.gradients(p, t)[0]
        p_x = tf.gradients(p, x)[0]
        p_tt = tf.gradients(p_t, t)[0]
        p_xx = tf.gradients(p_x, x)[0]
#         f = u_t + u*u_x - self.nu*u_xx
        c0=1
        f = (c0**2)*p_xx - p_tt - self.alpha*p_t       # f = u_t + u*u_x - self.nu*u_xx

        return f
    def callback(self, loss):
        print('Loss:', loss)
        
    def train(self):
        
        tf_dict = {self.X_p_tf: self.X_p, self.t_p_tf: self.t_p, self.p_tf: self.p,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}
                                                                                                                          
        self.optimizer.minimize(self.sess, 
                                feed_dict = tf_dict,         
                                fetches = [self.loss], 
                                loss_callback = self.callback)        
                                    
    
    def predict(self, X_star):
                
        p_star = self.sess.run(self.p_pred, {self.X_p_tf: X_star[:,0:1], self.t_p_tf: X_star[:,1:2]})  
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:,0:1], self.t_f_tf: X_star[:,1:2]})
               
        return p_star, f_star

if __name__ == "__main__": 
     
#     nu= 0.01/np.pi
    alpha1= 0
    noise = 0.0        

    N_p = 100
    N_f = 10000
    layers = [2, 10, 10,  1]
    
    data = scipy.io.loadmat('Training_Dataset.mat')
    
    t = data['t'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = np.real(data['p']).T
    
    X, T = np.meshgrid(x,t)
    
    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    p_star = Exact.flatten()[:,None]              

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)    
        
    xx1 = np.hstack((X[0:1,:].T, T[0:1,:].T))
    pp1 = Exact[0:1,:].T
    xx2 = np.hstack((X[:,0:1], T[:,0:1]))
    pp2 = Exact[:,0:1]
    xx3 = np.hstack((X[:,-1:], T[:,-1:]))
    pp3 = Exact[:,-1:]
    
    X_p_train = np.vstack([xx1, xx2, xx3])
    X_f_train = lb + (ub-lb)*lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_p_train))
    p_train = np.vstack([pp1, pp2, pp3])
    
    idx = np.random.choice(X_p_train.shape[0], N_p, replace=False)
    X_p_train = X_p_train[idx, :]
    p_train = p_train[idx,:]
        
    model = PhysicsInformedNN(X_p_train, p_train, X_f_train, layers, lb, ub, alpha1)
    
    start_time = time.time()                
    model.train()
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
    
    p_pred, f_pred = model.predict(X_star)
            
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)
    print('Error p: %e' % (error_p))                     

    
    P_pred = griddata(X_star, p_pred.flatten(), (X, T), method='cubic')
    Error = np.abs(Exact - P_pred)

######################################################################
    ############################# Plotting ###############################
    ######################################################################    
    
fig, ax = newfig(1.0, 1.1)
ax.axis('off')
    
    ####### Row 0: u(t,x) ##################    
gs0 = gridspec.GridSpec(1, 2)
gs0.update(top=1-0.06, bottom=1-1/3, left=0.15, right=0.85, wspace=0)
ax = plt.subplot(gs0[:, :])
    
h = ax.imshow(P_pred.T, interpolation='nearest', cmap='rainbow', 
    extent=[t.min(), t.max(), x.min(), x.max()], 
    origin='lower', aspect='auto')
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
#fig.colorbar(h, cax=cax)
    
ax.plot(X_p_train[:,1], X_p_train[:,0], 'kx', label = 'Data (%d points)' % (p_train.shape[0]), markersize = 4, clip_on = False)
    
line = np.linspace(x.min(), x.max(), 2)[:,None]
ax.plot(t[1000]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[2000]*np.ones((2,1)), line, 'w-', linewidth = 1)
ax.plot(t[3000]*np.ones((2,1)), line, 'w-', linewidth = 1)    
    
ax.set_xlabel('$t$')
ax.set_ylabel('$x$')
ax.legend(frameon=False, loc = 'best')
ax.set_title('$p(t,x)$', fontsize = 10)
    
    ####### Row 1: u(t,x) slices ##################    
gs1 = gridspec.GridSpec(1, 3)
gs1.update(top=1-1/3, bottom=0, left=0.1, right=0.9, wspace=0.5)
    
ax = plt.subplot(gs1[0, 0])
ax.plot(x,Exact[1000,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,P_pred[1000,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(t,x)$')    
ax.set_title('$t = 1$', fontsize = 10)
ax.axis('square')
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-1.1,1.1])
    
ax = plt.subplot(gs1[0, 1])
ax.plot(x,Exact[2000,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,P_pred[2000,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(t,x)$')
ax.axis('square')
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-1.1,1.1])
ax.set_title('$t = 2$', fontsize = 10)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
   
ax = plt.subplot(gs1[0, 2])
ax.plot(x,Exact[3000,:], 'b-', linewidth = 2, label = 'Exact')       
ax.plot(x,P_pred[3000,:], 'r--', linewidth = 2, label = 'Prediction')
ax.set_xlabel('$x$')
ax.set_ylabel('$p(t,x)$')
ax.axis('square')
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-1.1,1.1])    
ax.set_title('$t = 3$', fontsize = 10)
    
savefig('N10_L/Wave_L2_N10_a0')


x_val=data['x']
t_val=data['t']
np.savetxt('N10_L/P_t5_L2_N10_a0',P_pred)
np.savetxt('N10_L/x_t5_L2_N10_a0',x_val)
np.savetxt('N10_L/t_t5_L2_N10_a0',t_val)