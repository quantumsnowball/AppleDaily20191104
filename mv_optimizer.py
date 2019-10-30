# from ipdb import set_trace as br
# import sys, IPython; sys.excepthook = IPython.core.ultratb.ColorTB(call_pdb=True)
import db, ib, util, os, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.optimize import minimize

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-s', '--start', type=str, default='20160101', help='The start date of all time series')
parser.add_argument('-e', '--end', type=str, default=None, help='The end date of all time series')
parser.add_argument('--minW', type=float, default=0.00, help='min weight for each holdings')
parser.add_argument('--maxW', type=float, default=0.20, help='max weight for each holdings')
parser.add_argument('-rf', '--riskfree', type=float, default=0.025, help='Risk-free rate')
sargs = parser.parse_args()

class Optimizer:
    def get(self, ticker, start, end):
        df = pd.read_csv(f'resources/{ticker}.csv', index_col=0, parse_dates=True)
        df = df.loc[start:end]
        return df
    
    def get_close(self, ticker, *args, **kwargs):
        return self.get(ticker, *args, **kwargs)['Close']

    def porf_mu(self, mus, weights):
        return np.dot(mus, weights)
    
    def porf_sigma(self, weights, covm):
        return np.sqrt(np.dot(np.dot(weights,covm), weights.T))

    def porf_sharpe(self, weights, mus, covm, verbose=False):
        pMu, pSigma = self.porf_mu(mus, weights), self.porf_sigma(weights, covm)
        sharpe = (pMu-self.riskfree)/pSigma
        return {'sharpe':sharpe, 'mu':pMu, 'sigma':pSigma} if verbose else sharpe

    def __init__(self, start=None, end=None, riskfree=None, minW=None, maxW=None):
        resources = [os.path.basename(x).replace('.csv','') for x in os.listdir('resources')]
        ref = self.get_close('0005.HK', start, end)
        tss = {t:self.get_close(t, start, end) for t in resources}
        tss = {t:ts for t,ts in tss.items() if len(ts)>0.95*len(ref)}
        tss = pd.concat(tss, axis=1)
        lnrs = np.log(tss).diff()
        mus = lnrs.mean()*252
        sigmas = lnrs.std()*np.sqrt(252)
        covm = lnrs.cov()*252
        
        self.mus, self.sigmas, self.covm = mus, sigmas, covm
        self.tickers = covm.columns.tolist()
        self.start, self.end = start, end
        self.minW, self.maxW = minW, maxW
        self.riskfree = riskfree

    def optimize(self):
        weights = np.random.random(len(self.tickers))
        weights /= weights.sum()
        
        # find min risk solution
        print('Calculating minRisk portfolio ... ', end='')
        def weights_sums_to_1(inputs):
            return 1.0-np.sum(inputs)
        sol_minRisk = minimize(self.porf_sigma, weights, args=(self.covm, ),
                            bounds=[(self.minW, self.maxW) for _ in range(len(self.tickers))],
                            constraints=({'type':'eq', 'fun': weights_sums_to_1}, ))
        if sol_minRisk["success"]:
            print(f'result:{sol_minRisk["success"]}')
        else: raise Exception('Could not find minRisk solution.')

        # find max sharpe solution
        print('Calculating maxSharpe portfolio ... ', end='')
        sol_maxSharpe = minimize(lambda *args: -self.porf_sharpe(*args), weights, args=(self.mus, self.covm, ),
                            bounds=[(self.minW, self.maxW) for _ in range(len(self.tickers))],
                            constraints=({'type':'eq', 'fun': weights_sums_to_1}, ))
        if sol_maxSharpe["success"]:
            print(f'result:{sol_minRisk["success"]}')
        else: raise Exception('Could not find maxSharpe solution.')
        
        self.solutions = {n:{'solution':sol, 'metrics':self.porf_sharpe(sol['x'], self.mus, self.covm, verbose=True)
                            } for n,sol in {'minRisk':sol_minRisk, 'maxSharpe':sol_maxSharpe}.items()}

    def plot(self):
        fig, ax = plt.subplots(1,1, figsize=(15,8))
        ax.axhline(0, c='k'); ax.axvline(0, c='k')
        ax.set_xlim(-0.05,self.sigmas.max())
        ax.set_ylim(self.mus.min(), self.mus.max())
        title = f'({self.start}-{self.end}), size:{len(self.tickers)}'
        ax.set_title(title)

        # plot individual stocks
        for ticker, mu, sigma in zip(self.tickers, self.mus, self.sigmas):
            ax.scatter(sigma, mu, c='gray')
            ax.annotate(ticker, (sigma, mu), c='gray')
        
        # print solution and metrics
        for name,sol,color,marker in zip(['P_min_risk','P_max_sharpe'], 
                                        [self.solutions['minRisk'], self.solutions['maxSharpe']], 
                                        ['r','g'], 
                                        ['x','X']):
            mt = sol['metrics']
            pf = pd.Series(sol['solution']['x'], index=self.tickers).sort_values(ascending=False)
            pf = pf[abs(pf)>0.00001]
            ax.scatter(mt['sigma'], mt['mu'], c=color, marker=marker, s=200)
            text = f'{name}\nmu:{mt["mu"]:.2%}, sd:{mt["sigma"]:.2%}\n#:{mt["sharpe"]:.2%}\n---------------------------\n'
            for t,w in pf.items():
                text += f'{t} {w:8.2%}\n'
            ax.annotate(text, (mt['sigma'], mt['mu']), horizontalalignment='right', verticalalignment='bottom', c='b')

        plt.show(block=True)        

def main():
    app = Optimizer(
        start=sargs.start, end=sargs.end,
        riskfree=sargs.riskfree,
        minW=sargs.minW, maxW=sargs.maxW,
    )
    app.optimize()
    app.plot()

if __name__ == '__main__':
    main()