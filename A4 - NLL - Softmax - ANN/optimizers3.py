# optimizers3.py

import numpy as np
import copy
import math
import sys  # for sys.float_info.epsilon

######################################################################
## class Optimizers()
######################################################################

class Optimizers():

    def __init__(self, all_weights):
        '''all_weights is a vector of all of a neural networks weights concatenated into a one-dimensional vector'''
        
        self.all_weights = all_weights
        self.error_trace = []
        self.best_val_error = None
        self.best_epoch = None
        self.best_weights = None 

    def sgd(self, Xtrain, Ttrain, Xval, Tval,
            error_f, gradient_f,
            n_epochs=100, learning_rate=0.001, momentum=0, 
            error_convert_f=None, error_convert_name='MSE', verbose=True):
        '''
        Xtrain : two-dimensional numpy array 
            number of training samples  by  number of input components
        Ttrain : two-dimensional numpy array
            number of training samples  by  number of output components
        Xvalidate : two-dimensional numpy array 
            number of validation samples  by  number of input components
        Tvalidate : two-dimensional numpy array
            number of validationg samples  by  number of output components
        error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
        gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error
                    with respect to each weight.
        n_epochs : int
            Number of passes to take through all samples
        learning_rate : float
            Controls the step size of each update, only for sgd and adam
        momentum : float
            Controls amount of previous weight update to add to current weight update, only for sgd
        error_convert_f: function that converts the standardized error from error_f to original T units
        error_convert_name: used when printing progress updates
        verbose: if True print progress occasionally
        '''

        if self.error_trace == []:
            # not initialized yet
            self.w_update = 0
 
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            # Assume to be standardized already
            error = error_f(Xtrain, Ttrain)
            grad = gradient_f(Xtrain, Ttrain)

            val_error = error_f(Xval, Tval)

                
            self.w_update = learning_rate * grad + momentum * self.w_update
            # Update all weights using -= to modify their values in-place.
            self.all_weights -= self.w_update

            if self.best_val_error is None or val_error < self.best_val_error:
                self.best_val_error = val_error
                self.best_epoch = epoch + 1
                self.best_weights = self.all_weights.copy()

            if error_convert_f:
                error = error_convert_f(error)
                val_error = error_convert_f(val_error)

            self.error_trace.append([error, val_error])

            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):
                print(f'SGD: Epoch {epoch + 1} {error_convert_name}= Train {error:.5f} Validate {val_error:.5f}')

        self.all_weights[:] = self.best_weights

        return self.error_trace

    ######################################################################
    #### adam
    ######################################################################

    def adam(self, Xtrain, Ttrain, Xval, Tval,
             error_f, gradient_f,
             n_epochs=100, learning_rate=0.001, momentum=None,
             error_convert_f=None, error_convert_name='MSE', verbose=True):
        '''
        Xtrain : two-dimensional numpy array 
            number of training samples  by  number of input components
        Ttrain : two-dimensional numpy array
            number of training samples  by  number of output components
        Xvalidate : two-dimensional numpy array 
            number of validation samples  by  number of input components
        Tvalidate : two-dimensional numpy array
            number of validationg samples  by  number of output components
        error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
        gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error
                    with respect to each weight.
        n_epochs : int
            Number of passes to take through all samples
        learning_rate : float
            Controls the step size of each update, only for sgd and adam
        error_convert_f: function that converts the standardized error from error_f to original T units
        error_convert_name: used when printing progress updates
        verbose: if True print progress occasionally
        '''

        if self.error_trace == []:
            shape = self.all_weights.shape
            self.mt = np.zeros(shape)
            self.vt = np.zeros(shape)
                
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.beta1t = 1
            self.beta2t = 1

        epsilon = 1e-8
        epochs_per_print = n_epochs // 10

        for epoch in range(n_epochs):

            error = error_f(Xtrain, Ttrain)
            grad = gradient_f(Xtrain, Ttrain)

            val_error = error_f(Xval, Tval)

            self.mt[:] = self.beta1 * self.mt + (1 - self.beta1) * grad
            self.vt[:] = self.beta2 * self.vt + (1 - self.beta2) * grad * grad
            self.beta1t *= self.beta1
            self.beta2t *= self.beta2

            m_hat = self.mt / (1 - self.beta1t)
            v_hat = self.vt / (1 - self.beta2t)

            # Update all weights using -= to modify their values in-place.
            self.all_weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    
            if self.best_val_error is None or val_error < self.best_val_error:
                self.best_val_error = val_error
                self.best_epoch = epoch + 1
                self.best_weights = self.all_weights.copy()
                
            if error_convert_f:
                error = error_convert_f(error)
                val_error = error_convert_f(val_error)

            self.error_trace.append([error, val_error])

            if verbose and ((epoch + 1) % max(1, epochs_per_print) == 0):
                print(f'Adam: Epoch {epoch + 1} {error_convert_name}= Train {error:.5f} Validate {val_error:.5f}')

        self.all_weights[:] = self.best_weights

        return self.error_trace

    ######################################################################
    #### scg
    ######################################################################

    def scg(self, Xtrain, Ttrain, Xval, Tval,
            error_f, gradient_f,
            n_epochs=100, learning_rate=None, momentum=None,
            error_convert_f=None, error_convert_name='MSE', verbose=True):
        '''
        Xtrain : two-dimensional numpy array 
            number of training samples  by  number of input components
        Ttrain : two-dimensional numpy array
            number of training samples  by  number of output components
        Xvalidate : two-dimensional numpy array 
            number of validation samples  by  number of input components
        Tvalidate : two-dimensional numpy array
            number of validationg samples  by  number of output components
        error_f: function that requires X and T as arguments (given in fargs) and returns mean squared error.
        gradient_f: function that requires X and T as arguments (in fargs) and returns gradient of mean squared error
                    with respect to each weight.
        n_epochs : int
            Number of passes to take through all samples
        error_convert_f: function that converts the standardized error from error_f to original T units
        error_convert_name: used when printing progress updates
        verbose: if True print progress occasionally
        '''

        if self.error_trace == []:
            shape = self.all_weights.shape
            self.w_new = np.zeros(shape)
            self.w_temp = np.zeros(shape)
            self.g_new = np.zeros(shape)
            self.g_old = np.zeros(shape)
            self.g_smallstep = np.zeros(shape)
            self.search_dir = np.zeros(shape)

        sigma0 = 1.0e-6
        val_fold = error_f(Xval, Tval)
        fold = error_f(Xtrain, Ttrain)
        error = fold
        val_error = val_fold
        self.g_new[:] = gradient_f(Xtrain, Ttrain)
        self.g_old[:] = copy.deepcopy(self.g_new)
        self.search_dir[:] = -self.g_new
        success = True				# Force calculation of directional derivs.
        nsuccess = 0				# nsuccess counts number of successes.
        beta = 1.0e-6				# Initial scale parameter. Lambda in Moeller.
        betamin = 1.0e-15 			# Lower bound on scale.
        betamax = 1.0e20			# Upper bound on scale.
        nvars = len(self.all_weights)
        epoch = 1				# j counts number of epochs

        # Main optimization loop.
        while epoch <= n_epochs:

            # Calculate first and second directional derivatives.
            if success:
                mu = self.search_dir @ self.g_new
                if mu >= 0:
                    self.search_dir[:] = - self.g_new
                    mu = self.search_dir.T @ self.g_new
                kappa = self.search_dir.T @ self.search_dir
                if math.isnan(kappa):
                    print('kappa', kappa)

                if kappa < sys.float_info.epsilon:
                    return self.error_trace

                sigma = sigma0 / math.sqrt(kappa)

                self.w_temp[:] = self.all_weights
                self.all_weights += sigma * self.search_dir
                error_f(Xtrain, Ttrain)
                self.g_smallstep[:] = gradient_f(Xtrain, Ttrain)
                self.all_weights[:] = self.w_temp

                theta = self.search_dir @ (self.g_smallstep - self.g_new) / sigma
                if math.isnan(theta):
                    print('theta', theta, 'sigma', sigma, 'search_dir[0]', self.search_dir[0], 'g_smallstep[0]', self.g_smallstep[0])

            ## Increase effective curvature and evaluate step size alpha.

            delta = theta + beta * kappa
            # if math.isnan(scalarv(delta)):
            if math.isnan(delta):
                print('delta is NaN', 'theta', theta, 'beta', beta, 'kappa', kappa)
            elif delta <= 0:
                delta = beta * kappa
                beta = beta - theta / kappa

            if delta == 0:
                success = False
                fnow = fold
                val_fnow = val_fold
            else:
                alpha = -mu / delta
                ## Calculate the comparison ratio Delta
                self.w_temp[:] = self.all_weights
                self.all_weights += alpha * self.search_dir
                val_fnew = error_f(Xval, Tval)
                fnew = error_f(Xtrain, Ttrain)
                Delta = 2 * (fnew - fold) / (alpha * mu)
                if not math.isnan(Delta) and Delta  >= 0:
                    success = True
                    nsuccess += 1
                    fnow = fnew
                    val_fnow = val_fnew
                else:
                    success = False
                    fnow = fold
                    val_fnow = val_fold
                    self.all_weights[:] = self.w_temp

            epochsPerPrint = math.ceil(n_epochs/10)
            if verbose and epoch % max(1, epochsPerPrint) == 0:
                print(f'SCG: Epoch {epoch} {error_convert_name}= Train {error:.5f} Validate {val_error:.5f}')


            if success:

                fold = fnew
                val_fold = val_fnew
                self.g_old[:] = self.g_new
                self.g_new[:] = gradient_f(Xtrain, Ttrain)

                # If the gradient is zero then we are done.
                gg = self.g_new @ self.g_new  # dot(gradnew, gradnew)
                if gg == 0:
                    return self.error_trace

            if math.isnan(Delta) or Delta < 0.25:
                beta = min(4.0 * beta, betamax)
            elif Delta > 0.75:
                beta = max(0.5 * beta, betamin)

            # Update search direction using Polak-Ribiere formula, or re-start
            # in direction of negative gradient after nparams steps.
            if nsuccess == nvars:
                self.search_dir[:] = -self.g_new
                nsuccess = 0
            elif success:
                gamma = (self.g_old - self.g_new) @ (self.g_new / mu)
                #self.search_dir[:] = gamma * self.search_dir - self.g_new
                self.search_dir *= gamma
                self.search_dir -= self.g_new

            epoch += 1

            # If we get here, then we haven't terminated in the given number of
            # epochs.

            if self.best_val_error is None or val_fnow < self.best_val_error:
                self.best_val_error = val_fnow
                self.best_epoch = epoch + 1
                self.best_weights = self.all_weights.copy()

            if error_convert_f:
                val_error = error_convert_f(val_fnow)
                error = error_convert_f(fnow)
            else:
                error = fnow
                val_error = val_fnow
            self.error_trace.append([error, val_error])

        self.all_weights[:] = self.best_weights

        return self.error_trace
