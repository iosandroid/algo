import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as mpl
from sklearn.pipeline import Pipeline

def clfHyperFit(feat, lbl, t1, pip_clf, param_grid, cv = 3, bagging = [0, None, 1.], n_jobs = -1, pctEmbargo = 0, **fit_params):

	if set(lbl.values) == {0, 1}:
		scoring = 'f1' # f1 for meta-labeling
	else:
		scoring = 'neg_log_loss' # symmetric towards all cases

	# 1) hyperparameter search, on train data

	inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged

	gs = GridSearchCV(estimator = pipe_clf, n_jobs = n_jobs, iid = False)
	gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline

	# 2) fit validated model on the entirely of the data

	if bagging[1] > 0:
		gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps), n_estimators = int(bagging[0]), max_samples = float(bagging[1]), max_features = float(bagging[2]), n_jobs = n_jobs)
		gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
		gs = Pipeline(['bag', gs])

	return gs

def clfHyperFit0(feat, lbl, t1, pip_clf, param_grid, cv = 3, bagging = [0, None, 1.], rndSearchIter = 0, n_jobs = -1, pctEmbargo = 0, **fit_params):

	if set(lbl.values) == {0, 1}:
		scoring = 'f1' # f1 for meta-labeling
	else:
		scoring = 'neg_log_loss' # symmetric towards all cases

	# 1) hyperparameter search, on train data

	inner_cv = PurgedKFold(n_splits = cv, t1 = t1, pctEmbargo = pctEmbargo) # purged

	if rndSearchIter == 0:
		gs = GridSearchCV(estimator = pipe_clf, n_jobs = n_jobs, iid = False)
	else:
		gs = RandomizedSearchCV(estimator = pipe_clf, param_distributions = param_grid, scoring = scoring, cv = inner_cv, n_jobs = n_jobs, iid = False, n_iter = rndSearchIter)

	gs = gs.fit(feat, lbl, **fit_params).best_estimator_ # pipeline

	# 2) fit validated model on the entirely of the data

	if bagging[1] > 0:
		gs = BaggingClassifier(base_estimator = MyPipeline(gs.steps), n_estimators = int(bagging[0]), max_samples = float(bagging[1]), max_features = float(bagging[2]), n_jobs = n_jobs)
		gs = gs.fit(feat, lbl, sample_weight = fit_params[gs.base_estimator.steps[-1][0] + '__sample_weight'])
		gs = Pipeline(['bag', gs])

	return gs

class MyPipeline(Pipeline):

	def fit(self, X, y, sample_weight = None, **fit_params):
		
		if sample_weight is not None:
			fit_params[self.steps[-1][0] + '__sample_weight'] = sample_weight

		return super(MyPipeline, self).fit(X, y, **fit_params)


from scipy.stats import rv_continuous, kstest

class LogUniform_gen(rv_continuous):

	#random numbers log-uniformly distributed between 1 and e

	def _cdf(self, x):
		return np.log(x / self.a) / np.log(self.b / self.a)


def logUniform(a = 1, b = np.exp(1)):
	return LogUniform_gen(a = a, b = b, name = 'logUniform')


a, b, size = 1E-3, 1E3, 10000
vals = logUniform(a = a, b = b).rvs(size = size)

print kstest(rvs = np.log(vals), cdf = 'uniform', args = (np.log(a), np.log(b/a)), N = size)
print pd.Series(vals).describe()

mpl.subplot(121)
pd.Series(np.log(vals)).hist()

mpl.subplot(122)
pd.Series(vals).hist()

mpl.show()

