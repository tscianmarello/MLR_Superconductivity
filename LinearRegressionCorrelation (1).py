#This scratch file contains the results of the LINEAR regression with the best features selected based on the correlation
#of the different features and the value I want to predict: critical_temp.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mlxtend.plotting import scatterplotmatrix
from mlxtend.plotting import heatmap
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_regression, f_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from scipy import stats
from scipy.stats import yeojohnson
from sklearn.preprocessing import QuantileTransformer



class LinearRegressionCorrelation():
    def varianceThreshold(self, X):
        #Variance Threshold for features from sklearn used to determine what variance percent would be used to remove
        #features. I placed different variances in variance_percent and then plotted the result to get a visual
        #representation to select the variance percent.

        #Calls VarianceThreshold()
        transform = VarianceThreshold()
        #Fit, transform the data.
        X_sel = transform.fit_transform(X)

        #Selected variance percentages to map to the features removed.
        variance_percent = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35,
                            0.375, 0.4, 0.425, 0.45, 0.475, 0.5)

        #List that will hold the features removed.
        results = list()
        for i in variance_percent:
            transform = VarianceThreshold(threshold=i)
            X_sel = transform.fit_transform(X)
            n_feat = X_sel.shape[1]
            print('Variance Percent = %.4f, Number of Features = %d' % (i, n_feat))

            results.append(n_feat)

        #Plot of the number of features removed vs the variance percentage to visualize this result.
        pyplot.plot(variance_percent, results)
        pyplot.xlabel("Variance Percent")
        pyplot.ylabel("Number of Features")
        pyplot.title("Variance Percent vs Number of Features")
        pyplot.show()


        selected_feat = VarianceThreshold(threshold = .05)
        selected_feat.fit(X)
        print(X)

    #Scatterplot matrix of all the features related to the dependent variable.
    def scatter_plot(self, data, cols):
        print("1")
        scatterplotmatrix(data[cols].values, figsize=(10, 10), names=cols, alpha=0.5)
        print("2")
        plt.show()
        print('3')

    #MLxtend heat map of the features correlation
    def heat_map(self, data, cols):
        corr_map = np.corrcoef(data[cols].values.T)
        heat_map = heatmap(corr_map, row_names = cols, column_names = cols)
        plt.show()

    # Function that outputs the linear regression plot
    def lin_regplot(self, X, y, model):
        plt.scatter(X, y, c='white', edgecolor='black', s=70)
        plt.plot(X, model.predict(X), color='red', lw=2)
        plt.title('wtd_std_ThermalConductivity vs critical_temp')
        plt.xlabel('wtd_std_ThermalConductivity')
        plt.ylabel('critical_temp')
        plt.show()

    #Function that outputs the OLS table with the different statistic information.
    def table_OLS(self, X, y):
        # OLS table
        x = sm.add_constant(X)
        variables = sm.OLS(y, x)
        OLS_table = variables.fit()
        print(OLS_table.summary())

    # This is the SelectKBest from sklearn used to select the best features based on mutual_info_regression. These features
    def mutual_info_regression(self, X_train, y_train, X_test):
        f_sel = SelectKBest(score_func=mutual_info_regression, k='all')
        f_sel.fit(X_train, y_train)
        X_train_feat_sel = f_sel.transform(X_train)
        X_test_feat_sel = f_sel.transform(X_test)

        plt.bar([i for i in range(len(f_sel.scores_))], f_sel.scores_)

        plt.xlabel("Feature Index")
        plt.ylabel("Mutual Info Regression Value")
        plt.show()

    # Using SelectKBest from sklearn to select best features based on f_regression
    def f_regression(self, cols, X_train, y_train, X_test):
        f_sel = SelectKBest(score_func=f_regression, k='all')
        f_sel.fit(X_train, y_train)
        X_train_fs = f_sel.transform(X_train)
        X_test_fs = f_sel.transform(X_test)

        for i in range(len(f_sel.scores_)):
            print('Feature %d: %f' % (i, f_sel.scores_[i]))
            print(cols[i])

        pyplot.bar([i for i in range(len(f_sel.scores_))], f_sel.scores_)
        pyplot.show()

if __name__ == '__main__':
    #Place the csv
    data = pd.read_csv("/home/thomas/train.csv")
    pd.set_option('display.max_rows', None)
    element_data = pd.read_csv("/home/thomas/unique_m.csv")

    #Drops all values that are 0, NaN, or missing in train.csv.
    data.drop(data.columns[[0]], axis=1)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    data = data[(data != 0).all(1)]  # Find all 0s and remove them from dataset. Changed from 21262 to 19791
    element_data = element_data.iloc[data.index]
    data = data.reset_index(drop=True)
    element_data = element_data.reset_index(drop=True)

    cols = ["number_of_elements", "mean_atomic_mass", "wtd_mean_atomic_mass", "gmean_atomic_mass",
            "wtd_gmean_atomic_mass",
            "entropy_atomic_mass", "wtd_entropy_atomic_mass", "range_atomic_mass", "wtd_range_atomic_mass",
            "std_atomic_mass",
            "wtd_std_atomic_mass", "mean_fie", "wtd_mean_fie", "gmean_fie", "wtd_gmean_fie", "entropy_fie",
            "wtd_entropy_fie",
            "range_fie", "wtd_range_fie", "std_fie", "wtd_std_fie", "mean_atomic_radius", "wtd_mean_atomic_radius",
            "gmean_atomic_radius",
            "wtd_gmean_atomic_radius", "entropy_atomic_radius", "wtd_entropy_atomic_radius", "range_atomic_radius",
            "wtd_range_atomic_radius",
            "std_atomic_radius", "wtd_std_atomic_radius", "mean_Density", "wtd_mean_Density", "gmean_Density",
            "wtd_gmean_Density",
            "entropy_Density", "wtd_entropy_Density", "range_Density", "wtd_range_Density", "std_Density",
            "wtd_std_Density",
            "mean_ElectronAffinity", "wtd_mean_ElectronAffinity", "gmean_ElectronAffinity",
            "wtd_gmean_ElectronAffinity",
            "entropy_ElectronAffinity", "wtd_entropy_ElectronAffinity", "range_ElectronAffinity",
            "wtd_range_ElectronAffinity",
            "std_ElectronAffinity", "wtd_std_ElectronAffinity", "mean_FusionHeat", "wtd_mean_FusionHeat",
            "gmean_FusionHeat",
            "wtd_gmean_FusionHeat", "entropy_FusionHeat", "wtd_entropy_FusionHeat", "range_FusionHeat",
            "wtd_range_FusionHeat",
            "std_FusionHeat", "wtd_std_FusionHeat", "mean_ThermalConductivity", "wtd_mean_ThermalConductivity",
            "gmean_ThermalConductivity",
            "wtd_gmean_ThermalConductivity", "entropy_ThermalConductivity", "wtd_entropy_ThermalConductivity",
            "range_ThermalConductivity",
            "wtd_range_ThermalConductivity", "std_ThermalConductivity", "wtd_std_ThermalConductivity", "mean_Valence",
            "wtd_mean_Valence", "gmean_Valence", "wtd_gmean_Valence", "entropy_Valence", "wtd_entropy_Valence",
            "range_Valence",
            "wtd_range_Valence", "std_Valence", "wtd_std_Valence", "critical_temp"]

    lin_reg = LinearRegressionCorrelation()
    # Features
    X = data.iloc[:, 0:81].values
    # critical_temp variable

    y = data.iloc[81:82]

#Only 1 of these transformations can be applied to y at a time. Just comment out the # and run the program for that
#transformation:
    #y = data['critical_temp']
    #y = np.log(y)
    #y = np.square(y)
    #y = np.sqrt(y)
    #y = stats.boxcox(y, .099999)
    #yf= yeojohnson(y, 1)
    #trans_uniform = QuantileTransformer(n_quantiles=19791, output_distribution='uniform')
    #trans_normal = QuantileTransformer(n_quantiles=19791, output_distribution='normal')


    #Train, test split.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    #Prints out the Mean-Squared Error and the R^2 score related to the linear regression.
    print('MSE train: %.6f,MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
                                              mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.6f, R^2 test: %.6f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    #Instantiate the moddel

    #Displays the scatter plot
    lin_reg.scatter_plot(data, cols)
    #Displays the heat map
    lin_reg.heat_map(data, cols)
    #Displays the OLS table
    lin_reg.table_OLS(X, y)
    #Displays the plot of the linear regression
    lin_reg.lin_regplot(X, y, lr)
    #Displays the variance for each of the features.
    lin_reg.varianceThreshold(X)
    #Displays the f_regression plot of best features from SelectKBest
    lin_reg.f_regression(cols, X_train, y_train, X_test)
    #Displays the mutual_info_regression plot of best features from SelectKBest
    lin_reg.mutual_info_regression(X_train, y_train, X_test)