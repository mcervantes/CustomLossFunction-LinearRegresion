import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

'''
This programs uses gradient descent to fit a linear model with a noise componet to a dataset.
The noise distribution is a cauchy distribution
Inputs: path to the dataset and
inputs for the gradient descent: alpha (learning rate), num_iters (number of iterations for gradient descent)
'''

def LossFunction(X,y, theta,k):
   '''
   Compute Loss for linear regression assuming that the noise follows a Cauchy distribution
   The Input parameters are: X_{i} = {1, x_{i}}, y_{i}. theta and k.
   theta and k are the initialized parameters.
   '''
   J=0
   numerator = np.pi*(np.square(k*abs(x)) + np.square((np.dot(X,theta)-y)))
   denominator = k*abs(x)
   J = np.sum(np.log((numerator/denominator)))

   return J	
  


def gradientDescent(X,y, theta,k, alpha, num_iters):
   '''
   Performs gradient descent to learn theta and k,  updates theta and k
   by taking num_iters gradient steps with learning rate alpha.
   In addition save the cost J in every iteration and returns the vector.
   theta, k, J_history = gradientDescent(X, y, theta, alpha, num_iters)
   '''	
   m= len(y) 
   J_history = []

   for n_iter in range(num_iters):
      
      #calculate the partial derivate of the loss function with respect to theta
      #by parts for better readibility -> kpartial = kpartialf1*kpartialf2

      res = np.dot(X,theta)-y
      theta_partialf1 =np.pi/(np.square(res) + np.square(k*abs(x)))
      theta_partial = np.dot((2*theta_partialf1*res).T, X)

      #theta update 
      theta = theta -(alpha/m)*theta_partial.T

      #calculate the partial derivate of the loss function with respect to k
      #by parts for better readibility -> kpartial = kpartialf1*kpartialf2

      k_partialf1 = k*abs(x)*theta_partialf1
      k_partialf2 = abs(x) - np.square(res)/(abs(x)*k*k)
      k_partial = k_partialf1*k_partialf2

      #k update
      k = k - (alpha/m)*np.sum(k_partial)

      J_history.append(LossFunction(X,y,theta,k))
      
   return theta,k, J_history



def linearmodel(x, a,b):
   '''
   Linear model without the noise component. To calculate predictions.
   '''
   return a*x +b

    
def coeffd(ypred, X, y, theta, k):
  
   '''
   Returns a custom coefficient of the determination 
   ''' 
   sresModel = LossFunction(X,y,theta, k)
   svartot = np.sum(np.square(y - np.mean(y)))
   
   return 1- sresModel/svartot



#Dataset path
filename = 'data_1_5.csv'
path = "../" + filename

#Some gradient descent settings
number_iters = 80000
alpha= 0.0005


#initializes  the parameters theta and k for the gradientDescent
#algorithm
theta = np.zeros((2,1))
k=0.9

#extracts the data from the files and prepares it to be used
xy = pd.read_csv(path)
x = xy.ix[:,0]
y = xy.ix[:,1]
ones = np.ones((len(y),1))
x = x.reshape((len(x),1))
y = y.reshape((len(y),1))
X = np.column_stack((ones, x))
 
#runs the gradientDescent
thetapred, kpred, J_history = gradientDescent(X, y, theta,k, alpha, number_iters)


#plot the fit along with the data
xx= np.linspace(-20, 20, 1000)
plt.scatter(x, y)
plt.plot(x, (thetapred[0][0]) + thetapred[1][0]*x)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#print the values of the parametes and the value of the coefficient of determination
ypred = linearmodel(xy.ix[:,0], thetapred[1][0], thetapred[0][0] )
print('b, intercept', thetapred[0][0])
print('a, slope', thetapred[1][0])
print('k, noise distribution parameter', kpred)
print("R^2", coeffd(ypred, X, y, thetapred, kpred))


#plot the  value of loss functions as a function of the
#number of interations

plt.plot(J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Loss function')
plt.show()

