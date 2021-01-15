import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from yellowbrick.regressor import CooksDistance


from typing import Any


class DiagnosticPlots:

    def __init__(self, lin_model: Any):
        self.lin_model = lin_model
        self._influence_summary = None
        self._cooks_d_threshold = None

    @property
    def influence_summary(self):
        if self._influence_summary is None:
            influence = self.lin_model.get_influence()
            self._influence_summary = influence.summary_frame()
        return self._influence_summary

    @property
    def cooks_d_threshold(self):
        if self._cooks_d_threshold is None:
            self._cooks_d_threshold = 4/(len(self.lin_model.fittedvalues) - len(self.lin_model.params)-1)
        return self._cooks_d_threshold

    def linearity_res_vs_fitted(self):
        temp_data = pd.DataFrame(dict(fitted_values=self.lin_model.fittedvalues,
                                      residual=self.lin_model.resid))
        graph = sns.lmplot(x='fitted_values', y='residual',
                           data=temp_data, lowess=True, height=6, aspect=1.5,
                           line_kws={'color':'red'})
        graph.axes[0][0].axhline(0, color='black', ls='--')

    def normality_QQ_and_kernal(self, fig_size=(10, 5)):
        fig, ax = plt.subplots(1, 2, figsize=fig_size)
        sm.qqplot(self.lin_model.resid, dist=stats.t, fit=True, line='45', ax=ax[0])
        sns.distplot(self.lin_model.get_influence().resid_studentized_internal)

    def homoskedasticity(self):
        sqrt_std_residual = np.sqrt(np.abs(self.lin_model.get_influence().resid_studentized_internal))
        temp_data = pd.DataFrame(dict(fitted_values=self.lin_model.fittedvalues,
                                      sqrt_std_residual=sqrt_std_residual))
        graph = sns.lmplot(x='fitted_values', y='sqrt_std_residual',
                           data=temp_data, lowess=True, height=6, aspect=1.5,
                           line_kws={'color': 'red'})
        graph.axes[0][0].set_title('Scale-Location')
        graph.axes[0][0].set_ylabel('$\sqrt{|Standardized Residuals|}$')

    def outliers_std_res_vs_leverage(self, fig_size=(15, 10)):
        fig, ax = plt.subplots(figsize=fig_size)
        sm.graphics.influence_plot(self.lin_model, alpha=0.05, ax=ax, criterion="cooks")

    def outlier_cooks_distance(self, fig_size=(20, 7)):
        fig, ax = plt.subplots(figsize=fig_size)
        _inf_summary = self.influence_summary.reset_index()
        sns.scatterplot(x=_inf_summary.index, y=_inf_summary.cooks_d, marker='x')
        graph = sns.scatterplot(x=_inf_summary[_inf_summary.cooks_d > self.cooks_d_threshold].index,
                                y=_inf_summary[_inf_summary.cooks_d > self.cooks_d_threshold].cooks_d,
                                marker='x', color='red')
        graph.axhline(0, color='black')
        graph.axhline(self.cooks_d_threshold, color='red')

    def cooks_d_outliers(self):
        df = self.influence_summary[self.influence_summary.cooks_d > self.cooks_d_threshold][['cooks_d']]
        df = df[~df.index.duplicated(keep='first')]
        return df.style.bar(subset=['cooks_d'], color='Red')
