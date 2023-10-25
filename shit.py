import numpy as np
import scipy as scp
import sklearn as skl
import sklearn.datasets
import sklearn.naive_bayes

data = scp.io.loadmat('ExamData3D.mat')

# print(data)

iris = skl.datasets.load_iris(return_X_y=False)

# print(iris)


xtrain = np.swapaxes(np.array(data['X_train']), 0, 1)
xtest = np.swapaxes(np.array(data['X_test']), 0, 1)
ytrain = np.array([fucked_value - 1 for fucked_value in data['Y_train'][0]])
ytest = np.array([fucked_value - 1 for fucked_value in data['Y_test'][0]])

# np.apply_along_axis(lambda x: x 1, 0, ytest)

gnb = skl.naive_bayes.GaussianNB()
gnb.fit(xtrain, ytrain)
params = gnb.get_params()
print(f'Parameters are : {params}')
print(f'Class count is : {gnb.class_count_}')
print(f'With labels : {gnb.classes_}')
print(f'And probabilities : {gnb.class_prior_}')
print(f'Abs add. value to variances : {gnb.epsilon_}')
print(f'Mean of each feature per class got as : {gnb.theta_}')
print(f'Variance of each feature per class : {gnb.var_}')

score = round(gnb.score(xtest, ytest), 5)
print(f'Percent accuracy : {100.0 * score} %')
p_error = 1.00 - score
n_errors = p_error * np.shape(xtest)[0]
print(f'Number of errors : {n_errors}')

# print(xtrain)
# print(xtest)
# print(ytrain)
# print(ytest)


class MLClassifier:
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,  )
        '''
        # no. of variables / dimension
        self.d = x.shape[1]
        # no. of classes; assumes labels to be integers from 0 to nclasses-1
        self.nclasses = len(set(y))
        # self.nclasses = 0
        # list of means; mu_list[i] is mean vector for label i
        self.mu_list = []
        # list of inverse covariance matrices;
        # sigma_list[i] is inverse covariance matrix for label i
        # for efficiency reasons we store only the inverses
        self.sigma_inv_list = []
        # list of scalars in front of e^...
        self.scalars = []
        n = x.shape[0]
        for i in range(self.nclasses):
            # subset of obesrvations for label i
            cls_x = np.array([x[j] for j in range(n) if y[j] == i])
            # print(cls_x)
            if cls_x.__len__() == 0:
                continue
            # self.nclasses += 1
            mu = np.mean(cls_x, axis=0)
            # if mu == np.nan:
            #     mu = 0.0
            # rowvar = False, this is to use columns as variables
            # instead of rows
            sigma = np.cov(cls_x, rowvar=False)
            # if sigma == np.nan:
            #     sigma = 0.0
            # print(sigma)
            # eigs = np.linalg.eigvals(sigma)
            # print(eigs)
            if np.sum(np.linalg.eigvals(sigma) <= 0) != 0:
                # if at least one eigenvalue is <= 0 show warning
                print(f'Warning! Covariance matrix for label {cls_x} is not positive definite!\n')
            sigma_inv = np.linalg.inv(sigma)
            scalar = 1/np.sqrt(((2*np.pi)**self.d)*np.linalg.det(sigma))
            self.mu_list.append(mu)
            self.sigma_inv_list.append(sigma_inv)
            self.scalars.append(scalar)

    def _class_likelihood(self, x: np.ndarray, cls: int) -> float:
        '''
        x - numpy array of shape (d,)
        cls - class label
        Returns: likelihood of x under the assumption that class label is cls
        '''
        # print(f'Index {cls}')
        mu = self.mu_list[cls]
        sigma_inv = self.sigma_inv_list[cls]
        scalar = self.scalars[cls]
        # d = self.d
        exp = (-1/2)*np.dot(np.matmul(x-mu, sigma_inv), x-mu)
        return scalar * (np.e**exp)

    def predict(self, x: np.ndarray) -> int:
        '''
        x - numpy array of shape (d,)
        Returns: predicted label
        '''
        # print(range(self.nclasses))
        # print(self.mu_list.__len__())
        # print(self.sigma_inv_list.__len__())
        # print(self.scalars.__len__())
        likelihoods = [self._class_likelihood(x, i) for i in range(self.mu_list.__len__())]
        return np.argmax(likelihoods)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        x - numpy array of shape (n, d); n = #observations; d = #variables
        y - numpy array of shape (n,)
        Returns: accuracy of predictions
        '''
        n = x.shape[0]
        predicted_y = np.array([self.predict(x[i]) for i in range(n)])
        n_correct = np.sum(predicted_y == y)
        return n_correct/n

mlc = MLClassifier()
print(f'Shape of xtrain {np.shape(xtrain)}, should be (n, d)')
print(f'Shape of ytrain {np.shape(ytrain)}, should be (n,  )')
mlc.fit(xtrain, ytrain)

print(f'Shape of xtest {np.shape(xtest)}, should be (n, d)')
print(f'Shape of ytest {np.shape(ytest)}, should be (n,  )')
sc = mlc.score(xtest, ytest)
print(f'Final score is : {sc}')
