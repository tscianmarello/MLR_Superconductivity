#This file contains the different regression models using the final_dataset.csv.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mlxtend.plotting import scatterplotmatrix
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, ElasticNet, MultiTaskLasso, \
    MultiTaskLassoCV, MultiTaskElasticNetCV, MultiTaskElasticNet
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xg
from xgboost import XGBRegressor
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as statm
import statsmodels.stats.api as sms
from scipy.stats import yeojohnson
from scipy.special import lambertw
from sklearn.preprocessing import QuantileTransformer

########################################################################################################################
#Each of the following are the transformations necessary to see the results of the models. Just comment out the
#transformation you want to see in the models.

class MLR_models():

    #This is the Multi-Linear Regression Model.
    def multi_linear_regression(self, X_train, y_train, X_test, y_test):
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_train_pred = lr.predict(X_train)
        y_test_pred = lr.predict(X_test)


        #Columns associated with the dataset
        data.columns = ["wtd_range_atomic_mass", "range_ElectronAffinity", "range_FusionHeat", "gmean_ThermalConductivity",
                        "wtd_entropy_ThermalConductivity", "range_Valence", "wtd_range_Valence", "B", "O", "Si", "S", "Ca", "V",
                        "Fe", "Co", "Ni", "Cu", "Ge", "Se", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "In", "Sn", "Sb", "Ba",
                        "La", "Lu", "Os", "Ir", "Pt", "Tl", "Bi", "critical_temp"]

        X = data.iloc[:, 0:37]
        #Non-transformed critical_temp
        y = data.iloc[:, 37:38]
        print(lr.score(X_train, y_train))
        print(lr.score(X_test, y_test))
        print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
                mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    # RidgeCV from sklearn used to determine best alpha for Ridge Regression.
    def select_ridge(self, X, y):
        ridge_alphas = RidgeCV(
            alphas=[0.00001, .0001, .001, .01, .025, .05, .075, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0,
                    6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13, 14, 15, 16, 17, 18, 19, 20, 50, 75, 80, 90, 95, 100,
                    107, 107.5, 107.6, 107.7, 107.8, 107.9, 108, 108.05, 108.06, 108.07, 108.08, 108.09, 108.1,
                    108.11, 108.12, 108.13, 108.14, 108.15, 108.2, 108.3, 108.4, 108.5, 109, 109.5, 110, 114,
                    115, 116, 116.1, 116.2, 116.3, 116.4, 116.5, 116.6, 116.7, 116.8, 116.9, 117, 117.5, 118, 119,
                    120, 125, 130, 135, 136, 137, 138, 138.5, 139, 139.1, 139.2, 139.3, 139.4, 139.4, 139.5,
                    139.6, 139.7, 139.8, 139.9, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152,
                    152.1, 152.2, 152.3, 152.4, 152.5, 152.6, 152.7, 152.8, 152.9, 153, 153.1, 153.2, 153.3,
                    153.4, 153.5, 153.6, 153.7, 153.8, 153.9, 154, 155, 156, 157, 158, 159, 160, 170, 175, 176,
                    177, 178, 179, 179.1, 179.2, 179.3, 179.4, 179.5, 179.6, 179.7, 179.8, 179.9, 180, 180.1,
                    180.2, 180.3, 180.4, 180.5, 180.6, 180.7, 180.8, 180.9, 181, 182, 183, 184, 185, 190, 195,
                    195.1, 195.2, 195.3, 195.4, 195.5, 195.6, 195.7, 195.8, 195.9, 196, 196.1, 196.2, 196.3,
                    196.4, 196.5, 196.6, 196.7, 196.8, 196.9, 197, 198, 199, 200, 201, 202, 205, 210, 211, 212,
                    212.1, 212.2, 212.3, 212.4, 212.5, 212.51, 212.52, 212.53, 212.54, 212.55, 212.56, 212.57,
                    212.58, 212.59, 212.6, 212.61, 212.62, 212.63, 212.64, 212.65, 212.66, 212.67, 212.68, 212.69,
                    212.7, 212.8, 212.9, 213, 213.5, 214, 215, 216, 217, 218, 219, 220, 230, 240, 260, 300, 400,
                    500])

        sel_alpha = ridge_alphas.fit(X, y)
        sel_alpha.alpha_
        print(sel_alpha.alpha_)


    #Ridge Regression Model
    def ridge_model(self, X_train, y_train, X_test, y_test):
        ridge_model = Ridge(alpha=1520)

        ridge_model.fit(X_train, y_train)

        y_pred_train = ridge_model.predict(X_train)
        y_pred_test = ridge_model.predict(X_test)

        #Scoring the model
        print(ridge_model.score(X_train, y_train))
        print(ridge_model.score(X_test, y_test))
        print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_pred_train),
                mean_squared_error(y_test,y_pred_test)))
        print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)))

    # LassoCV from sklearn used to determine best alpha for Lasso Regression.
    def select_lasso(self, X, y):
        lasso_alphas = LassoCV(
            alphas=[0.00001, .0001, .001, .006, .007, .008, .009, .005, .004, .003, .002, .09, .08, .01,
                    .011, .012, .013, .014, .015, .016, .017, .018, .019, .02, .021, .022, .023, .024, .025,
                    .026, .027, .028, .029, .03, .04, .045, .046, .047, .048, .049, .05, .051, .052, .053,
                    .054, .055, .06, .07, .075, .08, .09, .091, .99, .1, .11, .13, .14, .15, .16, .17, .18,
                    .19, .2, .3, .4, .45, .46, .47, .48, .49, .5, .51, .52, .53, .54, .55, .6, .753, .754,
                    .7545, .755, .756, .76, .765, .77, .78, .79, .8, .9, 1.0, 1.2, 1.25, 1.5, 1.75, 2.0, 5,
                    10, 25, 30, 31, 32, 33, 34, 34.1, 34.2, 34.3, 34.4, 34.5, 34.6, 34.7, 34.8, 34.9, 35,
                    35.1, 35.2, 35.3, 35.4, 35.5, 35.6, 35.7, 35.8, 35.9, 36, 37, 38, 39, 40, 45, 50, 55, 60,
                    65, 70, 80, 90, 100])

        sel_alpha = lasso_alphas.fit(X, y)
        sel_alpha.alpha_
        print(sel_alpha.alpha_)

    # Lasso Regression Model
    def lasso_model(self, X_train, y_train, X_test, y_test):

        lasso_model = Lasso(alpha=.753)

        lasso_model.fit(X_train, y_train)

        y_train_pred = lasso_model.predict(X_train)
        y_test_pred = lasso_model.predict(X_test)

        # Scoring the model
        print(lasso_model.score(X_train, y_train))
        print(lasso_model.score(X_test, y_test))
        print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
                                                   mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    # Bayesian-Ridge Regression Model
    def bay_ridge_model(self, X_train, y_train, X_test, y_test):

        bay_ridge_model = BayesianRidge(alpha_1=1, alpha_2=1, lambda_1=900, lambda_2=100)

        bay_ridge_model.fit(X_train, y_train)

        y_train_pred = bay_ridge_model.predict(X_train)
        y_test_pred = bay_ridge_model.predict(X_test)

        # Scoring the model
        print(bay_ridge_model.score(X_train, y_train))
        print(bay_ridge_model.score(X_test, y_test))
        print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
                                                   mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    # ElasticNetCV from sklearn used to determine best alpha for Elastic-Net Regression.
    def select_elastic(self, X, y):

        encv_alphas = ElasticNetCV(
            alphas=[0.00001, .0001, .001, .002, .003, .004, .005, .006, .007, .008, .009, .099, .01,
                    .011, .012, .013, .014, .015, .016, .017, .018, .019, .02, .025, .026, .027, .028,
                    .029, .03, .031, .032, .033, .034, .035, .036, .037, .038, .039, .04, .041, .042,
                    .043, .044, .045, .05, .06, .07, .071, .072, .073, .074, .075, .076, .077, .078,
                    .079, .08, .09, .1, .11, .12, .13, .14, .15, .16, .17, .18, .19, .2, .21, .225, .23,
                    .24, .245, .246, .247, .248, .249, .25, .251, .252, .253, .254, .255, .26, .27, .275,
                    .3, .35, .4, .45, .46, .47, .48, .481, .482, .483, .484, .485, .486, .487, .488,
                    .489, .49, .491, .492, .493, .494, .495, .496, .497, .498, .499, .5, .51, .511, .512,
                    .513, .514, .515, .516, .517, .518, .519, .52, .525, .53, .54, .55, .6, .75, .752,
                    .7527, .7528, .7529, .753, .7531, .754, .7545, .755, .756, .76, .765, .77, .78, .79,
                    .8, .9, 1.0, 1.2, 1.25, 1.5, 1.75, 2.0])

        sel_alpha = encv_alphas.fit(X, y)
        sel_alpha.alpha_
        print(sel_alpha.alpha_)

    # Elastic-Net Regression Model
    def elastic_net_model(self, X_train, y_train, X_test, y_test):
        elast_net_model = ElasticNet(alpha=.253)

        elast_net_model.fit(X_train, y_train)

        y_train_pred = elast_net_model.predict(X_train)
        y_test_pred = elast_net_model.predict(X_test)

        # Scoring the model
        print(elast_net_model.score(X_train, y_train))
        print(elast_net_model.score(X_test, y_test))
        print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
                                                   mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    # MultiTaskLassoCV from sklearn used to determine best alpha for Multi-task Lasso Regression.
    def select_mtlasso(self, X, y):
        mtlasso_alphas = MultiTaskLassoCV(
            alphas=[0.00001, .0001, .001, .002, .003, .004, .005, .006, .007, .008, .009, .099,
                    .01, .011, .012, .013, .014, .015, .016, .017, .018, .019, .02, .025, .03,
                    .035, .036, .037, .038, .039, .04, .041, .042, .043, .044, .045, .05, .06,
                    .075, .1, .2, .225, .23, .24, .245, .246, .247, .248, .249, .25, .251, .252,
                    .253, .254, .255, .26, .27, .275, .3, .35, .4, .45, .46, .47, .48, .481,
                    .482, .483, .484, .485, .486, .487, .488, .489, .49, .491, .492, .493, .494,
                    .495, .496, .497, .498, .499, .5, .51, .511, .512, .513, .514, .515, .516,
                    .517, .518, .519, .52, .525, .53, .54, .55, .6, .75, .752, .7527, .7528,
                    .7529, .753, .7531, .754, .7545, .755, .756, .76, .765, .77, .78, .79, .8, .9,
                    1.0, 1.2, 1.25, 1.5, 1.75, 2.0])

        sel_alpha = mtlasso_alphas.fit(X, y)
        sel_alpha.alpha_
        print(sel_alpha.alpha_)

    # Multi-Task Lasso Regression Model
    def mtlasso_model(self, X_train, y_train, X_test, y_test):


        mtlasso_model = MultiTaskLasso(alpha=.005)

        mtlasso_model.fit(X_train, y_train)

        y_train_pred = mtlasso_model.predict(X_train)
        y_test_pred = mtlasso_model.predict(X_test)

        # Scoring the model
        print(mtlasso_model.score(X_train, y_train))
        print(mtlasso_model.score(X_test, y_test))
        print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
                                                   mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    # MultiTaskElasticNetCV from sklearn used to determine best alpha for Multi-task Elastic-Net Regression.
    def select_mtelastic(self, X, y):
        # MultiTaskElasticCV from sklearn used to determine best alpha for Multi-task Elastic-Net Regression.

        mtlasso_alphas = MultiTaskElasticNetCV(
            alphas=[0.00001, .0001, .001, .002, .003, .004, .005, .006, .007, .008, .009,
                    .099, .01, .011, .012, .013, .014, .015, .016, .017, .018, .019, .02,
                    .025, .026, .027, .028, .029, .03, .031, .032, .033, .034, .035, .036,
                    .037, .038, .039, .04, .041, .042, .043, .044, .045, .05, .06, .07, .071,
                    .072, .073, .074, .075, .076, .077, .078, .079, .08, .1, .2, .225, .23,
                    .24, .245, .246, .247, .248, .249, .25, .251, .252, .253, .254, .255,
                    .26, .27, .275, .3, .35, .4, .45, .46, .47, .48, .481, .482, .483, .484,
                    .485, .486, .487, .488, .489, .49, .491, .492, .493, .494, .495, .496,
                    .497, .498, .499, .5, .51, .511, .512, .513, .514, .515, .516, .517,
                    .518, .519, .52, .525, .53, .54, .55, .6, .75, .752, .7527, .7528, .7529,
                    .753, .7531, .754, .7545, .755, .756, .76, .765, .77, .78, .79, .8, .9,
                    1.0, 1.2, 1.25, 1.5, 1.75, 2.0])

        sel_alpha = mtlasso_alphas.fit(X, y)
        sel_alpha.alpha_
        print(sel_alpha.alpha_)

    def mtelastic_model(self, X_train, y_train, X_test, y_test):
        # Multi-task Elastic-Net Regression Model

        mten_model = MultiTaskElasticNet(alpha=.1918)

        mten_model.fit(X_train, y_train)

        y_train_pred = mten_model.predict(X_train)
        y_test_pred = mten_model.predict(X_test)

        # To score the model I can either use the .score from sklearn or use the MSE R^2 from the Machine Learning Book
        print(mten_model.score(X_train, y_train))
        print(mten_model.score(X_test, y_test))
        print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
                                                   mean_squared_error(y_test, y_test_pred)))
        print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    # XGBoost model using a linear function to compare to all other linear regressors
    def XGBoost_model(self, X_train, y_train, X_test, y_test):
        xg_model = XGBRegressor(booster='gblinear', reg_alpha=.844)
        xg_model.fit(X_train, y_train)
        y_pred_train = xg_model.predict(X_train)
        y_pred_test = xg_model.predict(X_test)

        # Scoring the model
        print(xg_model.score(X_train, y_train))
        print(xg_model.score(X_test, y_test))
        print('MSE train: %.3f, test: %.3f' % (mean_squared_error(y_train, y_pred_train),
                                               mean_squared_error(y_test, y_pred_test)))
        print('R^2 train: %.3f, test: %.3f' % (r2_score(y_train, y_pred_train), r2_score(y_test, y_pred_test)))


if __name__ == '__main__':
    # Read in the dataset: final_dataset.csv
    data = pd.read_csv("/home/thomas/final_dataset.csv")
    # Remove display restrictions to see the entire results
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_columns', None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.min_rows", None)

    # Columns associated with the dataset
    data.columns = ["wtd_range_atomic_mass", "range_ElectronAffinity", "range_FusionHeat", "gmean_ThermalConductivity",
                    "wtd_entropy_ThermalConductivity", "range_Valence", "wtd_range_Valence", "B", "O", "Si", "S", "Ca",
                    "V",
                    "Fe", "Co", "Ni", "Cu", "Ge", "Se", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh", "In", "Sn", "Sb", "Ba",
                    "La", "Lu", "Os", "Ir", "Pt", "Tl", "Bi", "critical_temp"]

    mlr = MLR_models()

    X = data.iloc[:, 0:37]
    # Non-transformed critical_temp
    y = data.iloc[:, 37:38]

    #These are the transformations available for the critical temperature. Simply comment it out and run.
    '''
    #np.log(y) transformed critical_temp
    y = np.log(y)
    '''

    '''
    #np.square(y) transformed critical_temp
    y = np.square(y)
    '''

    '''
    #np.sqrt(y) transformed critical_temp
    y = np.sqrt(y)
    '''

    '''
    #boxcox(y) transformed critical_temp
    y = stats.boxcox(y, .6)
    '''

    '''
    #yeojohnson(y) transformed critical_temp
    yf= yeojohnson(y, 1)
    '''

    '''
    #Uniform Quantile Transform transformed critical_temp
    trans = QuantileTransformer(n_quantiles=19791, output_distribution='uniform')
    y = trans.fit_transform(y)
    '''

    '''
    #Uniform Quantile Transform transformed critical_temp
    trans = QuantileTransformer(n_quantiles=19791, output_distribution='normal')
    y = trans.fit_transform(y)
    '''

    # Train, test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    mlr.multi_linear_regression(X_train, y_train, X_test, y_test)
    mlr.select_ridge(X, y)
    mlr.ridge_model(X_train, y_train, X_test, y_test)
    mlr.select_lasso(X, y)
    mlr.lasso_model(X_train, y_train, X_test, y_test)
    mlr.bay_ridge_model(X_train, y_train, X_test, y_test)
    mlr.select_elastic(X, y)
    mlr.elastic_net_model(X_train, y_train, X_test, y_test)
    mlr.select_mtlasso(X, y)
    mlr.mtlasso_model(X_train, y_train, X_test, y_test)
    mlr.select_mtelastic(X, y)
    mlr.mtelastic_model(X_train, y_train, X_test, y_test)
    mlr.XGBoost_model(X_train, y_train, X_test, y_test)