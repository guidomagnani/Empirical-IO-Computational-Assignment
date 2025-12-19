import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import time


# import optimize2 as opt # import modified optimize.minimize_nelder-mead ; alternatively change line 569 from 'and' to 'or' on original optimize.py

class BLP:
    def __init__(self, data, theta2w, mtol, niter, root_dir='../docs/Python_codes/', method='nelder-mead'):  # define data instances

        self.niter = niter
        self.mtol = mtol
        self.theta2w = theta2w
        self.x1 = data.x1
        self.x2 = data.x2
        self.s_jt = data.s_jt
        self.v = data.v
        self.cdindex = data.cdindex  # consider asarray
        self.cdid = data.cdid
        self.cdid_demogr = data.cdid_demogr
        self.vfull = data.v.loc[self.cdid_demogr, :]
        dfull = data.demogr.loc[self.cdid_demogr, :]
        dfull.columns = pd.RangeIndex(len(dfull.columns))
        dfull.index = pd.RangeIndex(len(dfull.index))
        self.dfull = dfull
        self.demogr = data.demogr
        self.IV = data.IV
        self.K1 = self.x1.shape[1]
        self.K2 = self.x2.shape[1]
        self.ns = data.ns  # number of simulated "indviduals" per market
        self.D = int(self.dfull.shape[1] / self.ns)  # number of demographic variables
        self.T = np.asarray(self.cdindex).shape[0]  # number of markets = (# of cities)*(# of quarters)
        self.TJ = self.x1.shape[0]  # number of observations
        self.J = int(self.TJ / self.T)  # average number of observation per market

        self.invA = np.linalg.inv(self.IV.T @ self.IV)

        self.root_dir = root_dir
        self.method = method

        # mid2 = np.mat(self.x1.T)*np.mat(self.IV)*self.invA*np.mat(self.IV.T)*np.mat(self.x1)
        # invmid2=np.linalg.inv(mid2)

        # Calculating s0 -outside good shares
        temp = self.s_jt.cumsum()
        sum1 = temp[self.cdindex]
        sum1[1:np.shape(sum1)[0]] = np.diff(sum1)
        outshr = (1 - sum1[np.asarray(self.cdid, dtype=int)])
        outshr = np.abs(outshr)
        outshr = np.reshape(outshr, (np.shape(outshr)[0], 1))

        # find initial guess of coefficients by running simple logit regression
        y = np.log(self.s_jt) - np.log(outshr)
        mid = self.x1.T @ self.IV @ self.invA @ self.IV.T
        t = np.linalg.inv(mid @ self.x1) @ mid @ y

        # mean utility - initial
        self.mvalold = self.x1 @ (t)
        self.mvalold = np.exp(self.mvalold).values.reshape(np.shape(self.mvalold)[0], 1)
        self.oldt2 = np.zeros(np.shape(theta2w))  # initial old2
        self.gmmvalold = 0
        self.gmmdiff = 1
        self.gmmresid = np.ones((6024, 1))  # @@@@@@ initialization
        self.theta1 = pd.DataFrame()

    # transforms inputed guess of theta2w into an ndarray type
    def init_theta(self, theta2w):
        theta2w = theta2w.reshape(self.K2, 1 + self.D)  # #rows: # x2 variables
        self.theti, self.thetj = list(np.where(theta2w != 0))  # columns: #demographic variables + 1 (constant)
        self.theta2 = theta2w[np.where(theta2w != 0)]
        return self.theta2

    # mean utility function
    def mufunc(self, theta2w):
        mu = np.zeros((self.TJ, self.ns))
        for i in range(self.ns):
            v_i = np.array(self.vfull.loc[:, np.arange(i, self.K2 * self.ns, self.ns)])
            d_i = np.array(self.dfull.loc[:, np.arange(i, self.D * self.ns, self.ns)])
            temp = d_i @ theta2w[:, 1:(self.D + 1)].T
            mu[:, i] = (np.multiply(self.x2, v_i) @ theta2w[:, 0]) + np.multiply(self.x2, temp) @ np.ones((self.K2))

        return mu

    def ind_sh(self, expmu):
        eg = np.multiply(expmu, np.kron(np.ones((1, self.ns)), self.mvalold))
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)
        denom = denom1[np.asarray(self.cdid, dtype=int), :]

        return np.multiply(eg, denom)

    def mktsh(self, expmu):
        # compute the market share for each product
        temp = self.ind_sh(expmu).T
        f = (sum(temp) / float(self.ns)).T
        f = f.reshape(self.x1.shape[0], 1)
        return f

    def meanval(self, theta2):
        # phased tolerance to speed up computation
        #        if self.gmmdiff < 1e-15:
        #            self.mtol = 1e-15
        #        elif self.gmmdiff < 1e-3:
        #            self.mtol = 1e-4
        #        else:
        #            self.mtol = 1e-2

        if np.ndarray.max(np.absolute(theta2 - self.oldt2)) < 0.01:
            tol = self.mtol
            flag = 0
        else:
            tol = self.mtol  # when theta2 (output of minimize) is still larger than 0.01, flag=1 => pass it to theta old2, that will be used in next iteration
            flag = 1
        norm = 1
        avgnorm = 1
        i = 0
        theta2w = np.zeros((self.K2, self.D + 1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = theta2[ind]
        u = self.mufunc(theta2w)
        u_safe = np.clip(u, -700, 700)
        expmu = np.exp(u_safe)

        while (norm > self.mtol) & (i < self.niter):

            pred_s_jt = self.mktsh(expmu)
            self.mval = np.multiply(self.mvalold, self.s_jt) / pred_s_jt
            t = np.abs(self.mval - self.mvalold)
            norm = np.max(t)
            avgnorm = np.mean(t)
            self.mvalold = self.mval
            i += 1

            if (norm > self.mtol) & (i > self.niter - 1):
                print('Max number of ' + str(self.niter) + ' iterations reached, final norm =', norm)

        # print(['# of iterations for delta convergence: ', i])

        if (flag == 1) & (sum(np.isnan(self.mval))) == 0:
            self.mvalold = self.mval
            self.oldt2 = theta2
        return np.log(self.mval)

    # calculates the jacobian of the
    def jacob(self, theta2):
        cdindex = np.asarray(self.cdindex, dtype=int)
        theta2w = np.zeros((self.K2, self.D + 1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = self.theta2[
                ind]  # build theta2w (from array to ndarray) to plug in ind_sh
        u = self.mufunc(theta2w)
        u_safe = np.clip(u, -700, 700)
        expmu = np.exp(u_safe)

        shares = self.ind_sh(expmu)
        f1 = np.zeros((np.asarray(self.cdid).shape[0], self.K2 * (self.D + 1)))
        # calculate derivative of shares with respect to the first column of theta2w (variable that are not interacted with demogr var, sigmas)
        for i in range(self.K2):
            xv = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1, self.ns)),
                             self.v.loc[self.cdid, self.ns * i:self.ns * (i + 1) - 1])
            temp = np.cumsum(np.multiply(xv, shares), 0).values
            sum1 = temp[cdindex, :]
            sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
            f1[:, i] = np.mean((np.multiply(shares, xv - sum1[np.asarray(self.cdid, dtype=int), :])),
                               1)  # mean over columns

        for j in range(self.D):
            d = self.demogr.loc[self.cdid, self.ns * (j) + 1:self.ns * (j + 1)]
            temp1 = np.zeros((np.asarray(self.cdid).shape[0], self.K2))
            for i in range(self.K2):
                xd = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1, self.ns)), d)
                temp = np.cumsum(np.multiply(xd, shares), 0).values
                sum1 = temp[cdindex, :]
                sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
                temp1[:, i] = np.mean((np.multiply(shares, xd - sum1[np.asarray(self.cdid, dtype=int), :])), 1)
            f1[:, self.K2 * (j + 1):self.K2 * (j + 2)] = temp1

        self.rel = self.theti + self.thetj * (max(self.theti) + 1)
        f = np.zeros((np.shape(self.cdid)[0], self.rel.shape[0]))
        n = 0

        for i in range(np.shape(self.cdindex)[0]):
            temp = shares[n:(self.cdindex[i] + 1), :]
            H1 = temp @ temp.T
            H = (np.diag(np.array(sum(temp.T)).flatten()) - H1) / self.ns
            f[n:(cdindex[i] + 1), :] = np.linalg.inv(H) @ f1[n:(cdindex[i] + 1), self.rel]
            n = cdindex[i] + 1
        return f

    # compute GMM objective function
    def gmmobj(self, theta2):
        # print(theta2)
        delta = self.meanval(theta2)
        self.theta2 = theta2
        # the following deals with cases where the min algorithm drifts into region where the objective is not defined
        if max(np.isnan(delta)) == 1:
            f = np.ndarray((1, 1), buffer=np.array([1e+10, 1e+10]))
        else:
            temp1 = self.x1.T @ self.IV
            temp2 = delta.T @ self.IV
            self.theta1 = np.linalg.inv(temp1 @ self.invA @ temp1.T) @ temp1 @ self.invA @ temp2.T
            self.gmmresid = delta - self.x1 @ self.theta1
            temp1 = self.gmmresid.T @ (self.IV)
            f = temp1 @ self.invA @ temp1.T
            f = f.values
            if np.shape(f) > (1, 1):
                temp = self.jacob(theta2).T
                df = 2 * temp @ self.IV @ self.invA @ self.IV.T @ self.gmmresid
        #print('fval:', f[0, 0])  # value of the gmm objective function

        # to be used in meanval to phase tolerance based on gmm value
        self.gmmvalnew = f[0, 0]
        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew

        return (f[0, 0])

    # calculates the gradient of the objective function based on jacobian
    def gradobj(self, theta2):
        temp = self.jacob(theta2).T
        df = 2 * temp @ self.IV @ self.invA @ self.IV.T @ self.gmmresid
        df = df.values  ##@@
        print('this is the gradient ' + str(df))

        return df

    # WARNING!!!!!!!! In some versions of scipy there can be a problem with xatol in optimize.minimize_neldermead. IF so, change optimize.py codeline 569: replace 'and' to 'or'OR use modified script optimize2.py

    # maximizes the objective function
    def iterate_optimization(self, opt_func, param_vec, jac, options, bounds=None):
        res = minimize(opt_func, param_vec, method=self.method, options=options, bounds=bounds)
        ##res = opt._minimize_neldermead(opt_func, param_vec,disp=True,maxiter=100,fatol=0.0001,xatol=0.0001)
        # res = minimize(opt_func, param_vec, method='BFGS', bounds=None, callback=None, options=options)
        return res

    # calculates the variance-covariance matrix of the estimates

    def varcov(self, theta2):
        Z = self.IV.shape[1]
        temp = self.jacob(theta2)
        a = np.concatenate((self.x1.values, temp), 1).T @ self.IV.values
        IVres = np.multiply(self.IV.values, self.gmmresid.values @ np.ones((1, Z)))
        b = IVres.T @ (IVres)
        if np.linalg.det(a @ np.asarray(self.invA) @ a.T) != 0:
            inv_aAa = np.linalg.inv(a @ np.asarray(self.invA) @ a.T)
        else:
            print('Error: singular matrix in covariate function ; forced result')  # if jacobian has row of zeros
            inv_aAa = np.linalg.lstsq(a @ np.asarray(self.invA) @ a.T)
        f = inv_aAa @ a @ np.asarray(self.invA) @ b @ np.asarray(self.invA) @ a.T @ inv_aAa
        return f

        # computes estimates of the coefficients and its standard errors; outputs a histogram and a csv file with estimates

    def results(self, theta2):
        self.theta1 = self.theta1.values
        var = self.varcov(self.theta2)
        se = np.sqrt(var.diagonal())
        t = se.shape[0] - self.theta2.shape[0]

        print('Object vcov dimensions: ' + str(np.shape(var)))
        print('Object se dimensions: ' + str(np.shape(se)))

        print('Object theta2w dimensions:     ' + str(np.shape(self.theta2)))
        print('Object t dimensions:     ' + str(np.shape(t)))

        self.fval_results = theta2.fun
        print('Optimum value of the GMM objective function: ' + str(self.fval_results))

        # histogram of the coefficients
        alfa_i = []
        alfa_i2 = []
        for i in range(0, self.T):
            data_market = np.reshape(self.demogr.loc[i, 0:self.ns * self.D].values, (self.ns, self.D))
            v_market = np.reshape(self.v.loc[i, 0:self.ns - 1].values,
                                  (self.ns, 1))  # caution: v is created to have 1000 columns
            alfa_i2.extend(np.add(data_market @ (self.theta2[1:3]).T, self.theta2[0] * v_market[:, 0]))
            alfa_i.extend(data_market @ (self.theta2[1:3].T))
        # alfa_i=(self.theta1[1]+alfa_i2)/100
        alfa_i = (self.theta1[1] + alfa_i2) / 100
        h = plt.figure()
        plt.hist(alfa_i, bins=25, range=(np.min(alfa_i), np.max(alfa_i)))
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of the price coefficient -2007')
        h.savefig('elasticities.png')
        # compute the elasticities
        # reshape the alfas
        alfa_i_reshaped = np.reshape(alfa_i, (
        self.ns, len(self.cdindex))).T  # changed from cdindex (136) to cdid (6024) to match matlab size
        alfa_i_reshaped = alfa_i_reshaped[np.asarray(self.cdid, dtype=int), :]

        # convert theta2 into ndarray to plug in ind_sh
        theta = np.zeros((self.K2, self.D + 1))
        for ind in range(len(self.theti)):
            theta[self.theti[ind], self.thetj[ind]] = self.theta2[ind]
            #######
        u = self.mufunc(theta)
        u_safe = np.clip(u, -700, 700)
        expmu = np.exp(u_safe)
        f = self.ind_sh(expmu)
        mval = (self.x1 @ self.theta1)

        # from ind_sh function: here mvalold = gmmresid +mval
        u = self.mufunc(theta)
        u_safe = np.clip(u, -700, 700)
        exp_u = np.exp(u_safe)
        eg = np.multiply(exp_u, np.kron(np.ones((1, self.ns)), (mval + self.gmmresid)))
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)
        denom = denom1[np.asarray(self.cdid, dtype=int), :]
        f22 = np.multiply(eg, denom)
        ######
        f2 = np.sum(f22.T, 0) / self.ns
        f2 = f2.T
        error = self.s_jt - f2

        # table with parameters + export to csv file

        self.theta1_results = pd.DataFrame(
            {'Theta1': self.theta1.reshape(self.K1, ), 'Std.Error_theta1': se[0:-self.theta2.shape[0]]},
            columns=(['Theta1', 'Std.Error_theta1']))
        self.theta2_results = pd.DataFrame(
            {'Theta2': self.theta2.reshape(self.theta2.shape[0], ), 'Std.Error_theta2': se[-self.theta2.shape[0]:]},
            columns=(['Theta2', 'Std.Error_theta2']))
        self.theta1_results.to_csv(self.root_dir + "theta1.csv")
        self.theta2_results.to_csv(self.root_dir + "theta2.csv")
