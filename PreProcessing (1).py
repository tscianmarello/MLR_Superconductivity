#This is the file that dealt with the data pre-processing. This was completed after the linear regression models did not
#produce the best results.

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.formula.api as statm
import statsmodels.stats.api as sms
from scipy.stats import yeojohnson
from scipy import stats
from scipy.special import lambertw
from sklearn.preprocessing import QuantileTransformer

#Residual plot of a linear regression model to test the dataset.
def residual_plot():
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_train_pred = lr.predict(X_train)
    y_test_pred = lr.predict(X_test)

    #Scoring the model
    print('MSE train: %.6f, MSE test: %.6f' % (mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))
    print('R^2 train: %.6f, R^2 test: %.3f' % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred)))

    plt.scatter(y_train_pred,  y_train_pred - y_train, c='red', marker='o', edgecolor='white', label='Training data')
    plt.scatter(y_test_pred,  y_test_pred - y_test, c='blue', marker='s', edgecolor='white', label='Test data')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.legend(loc='lower left')
    plt.hlines(y=0, xmin=-200, xmax=1000, color='black', lw=2)
    plt.xlim([-100, 100])
    plt.tight_layout()

    plt.show()

#Function that selects the features that based on variance.
def Var_Percent_Select(X):
    var_sel = VarianceThreshold(.05)
    var_sel.fit(X)
    var_sel = (pd.DataFrame(var_sel.transform(data)))

    return var_sel

#This block of code was used to determine the correlations between the features, much like the heatmap, and then used to
#drop the features that had a correlation of .9 or higher.
def correlation(data):
    X = data
    y = data.iloc[:, 81:82].values
    y = y.ravel()
    mat = data.corr().abs()
    #print(mat)

    up_tri_mat = mat.where(np.triu(np.ones(mat.shape),k=1).astype(np.bool))
    print(up_tri_mat)

    print()
    print()
    feat_drop = [column for column in up_tri_mat.columns if any(up_tri_mat[column] > 0.9)]
    print("Number of features dropped: ")
    print(len(feat_drop))
    print()

    print("Features dropped: ")
    print(feat_drop)

#This is the calculation of the variance inflation factor
def vif():
    #Features
    X = final_dataset
    #critical_temp variable
    y = data.iloc[:, 81:82].values

    vif_stat = pd.DataFrame()
    vif_stat['Feature'] = X.columns
    vif_stat['VIF'] = [variance_inflation_factor(X.values, i)
                       for i in range(X.shape[1])]

    print(vif_stat)

#Breusch-Pagan test for heteroskedasticity
def breusch_pagan_test(data):
    bp_test = statm.ols('y~X', data = data).fit()
    test = sms.het_breuschpagan(bp_test.resid, bp_test.model.exog)
    print(test)

if __name__ == '__main__':
    # This block of code removed any rows with 0 values in the train.csv dataset and then removed those corresponding rows
    # in the unique_m.csv dataset, reindexed both csvs, and then outputted the modified datasets into 2 new csvs
    # (train.csv = feat_data.csv, unique_m.csv = elem_data.csv). Then, I manually removed the extra critical temperature
    # feature in the elem_data.csv and copied over all the data in the feat_data.csv and elem_data.csv into 1 csv called
    # combined.csv.

    pd.set_option('display.max_rows', None)
    data = pd.read_csv("/home/thomas/train.csv")
    element_data = pd.read_csv("/home/thomas/unique_m.csv")

    # complete_data = data.append(element_data)
    # print(complete_data.head())

    data.columns = ["number_of_elements", "mean_atomic_mass", "wtd_mean_atomic_mass", "gmean_atomic_mass",
                    "wtd_gmean_atomic_mass",
                    "entropy_atomic_mass", "wtd_entropy_atomic_mass", "range_atomic_mass", "wtd_range_atomic_mass",
                    "std_atomic_mass",
                    "wtd_std_atomic_mass", "mean_fie", "wtd_mean_fie", "gmean_fie", "wtd_gmean_fie", "entropy_fie",
                    "wtd_entropy_fie",
                    "range_fie", "wtd_range_fie", "std_fie", "wtd_std_fie", "mean_atomic_radius",
                    "wtd_mean_atomic_radius", "gmean_atomic_radius",
                    "wtd_gmean_atomic_radius", "entropy_atomic_radius", "wtd_entropy_atomic_radius",
                    "range_atomic_radius", "wtd_range_atomic_radius",
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
                    "wtd_range_ThermalConductivity", "std_ThermalConductivity", "wtd_std_ThermalConductivity",
                    "mean_Valence",
                    "wtd_mean_Valence", "gmean_Valence", "wtd_gmean_Valence", "entropy_Valence", "wtd_entropy_Valence",
                    "range_Valence",
                    "wtd_range_Valence", "std_Valence", "wtd_std_Valence", "critical_temp"]

    data.drop(data.columns[[0]], axis=1)
    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    data = data[(data != 0).all(1)]  # Find all 0s and remove them from dataset. Changed from 21262 to 19791
    element_data = element_data.iloc[data.index]
    data = data.reset_index(drop=True)
    element_data = element_data.reset_index(drop=True)
    data.to_csv('/home/thomas/feat_data.csv', index=False)
    element_data.to_csv('/home/thomas/elem_data.csv', index=False)

    # Read csv file and separate data into X and y
    data = pd.read_csv("/home/thomas/combined_data.csv")
    # Features
    X = data.iloc[:, np.r_[0:81, 82:168]]
    # critical_temp variable
    y = data.iloc[:, 81:82]
    # print(X.head())

    pd.set_option('display.width', None)
    pd.set_option('display.max_columns', None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.min_rows", None)

    # Train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    new_data = Var_Percent_Select(data)

    new_data.columns = ["number_of_elements", "mean_atomic_mass", "wtd_mean_atomic_mass", "gmean_atomic_mass",
                        "wtd_gmean_atomic_mass", "entropy_atomic_mass", "wtd_entropy_atomic_mass",
                        "range_atomic_mass", "wtd_range_atomic_mass", "std_atomic_mass", "wtd_std_atomic_mass",
                        "mean_fie", "wtd_mean_fie", "gmean_fie", "wtd_gmean_fie", "entropy_fie", "wtd_entropy_fie",
                        "range_fie", "wtd_range_fie", "std_fie", "wtd_std_fie", "mean_atomic_radius",
                        "wtd_mean_atomic_radius", "gmean_atomic_radius", "wtd_gmean_atomic_radius",
                        "entropy_atomic_radius", "wtd_entropy_atomic_radius", "range_atomic_radius",
                        "wtd_range_atomic_radius", "std_atomic_radius", "wtd_std_atomic_radius", "mean_Density",
                        "wtd_mean_Density", "gmean_Density", "wtd_gmean_Density", "entropy_Density",
                        "wtd_entropy_Density", "range_Density", "wtd_range_Density", "std_Density", "wtd_std_Density",
                        "mean_ElectronAffinity", "wtd_mean_ElectronAffinity", "gmean_ElectronAffinity",
                        "wtd_gmean_ElectronAffinity", "entropy_ElectronAffinity", "wtd_entropy_ElectronAffinity",
                        "range_ElectronAffinity", "wtd_range_ElectronAffinity", "std_ElectronAffinity",
                        "wtd_std_ElectronAffinity", "mean_FusionHeat", "wtd_mean_FusionHeat", "gmean_FusionHeat",
                        "wtd_gmean_FusionHeat", "entropy_FusionHeat", "wtd_entropy_FusionHeat", "range_FusionHeat",
                        "wtd_range_FusionHeat", "std_FusionHeat", "wtd_std_FusionHeat", "mean_ThermalConductivity",
                        "wtd_mean_ThermalConductivity", "gmean_ThermalConductivity", "wtd_gmean_ThermalConductivity",
                        "entropy_ThermalConductivity", "wtd_entropy_ThermalConductivity", "range_ThermalConductivity",
                        "wtd_range_ThermalConductivity", "std_ThermalConductivity", "wtd_std_ThermalConductivity",
                        "mean_Valence", "wtd_mean_Valence", "gmean_Valence", "wtd_gmean_Valence", "entropy_Valence",
                        "wtd_entropy_Valence", "range_Valence", "wtd_range_Valence", "std_Valence", "wtd_std_Valence",
                        "critical_temp", "H", "Be", "B", "C", "O", "F", "Al", "Si", "P", "S", "Ca", "Ti", "V", "Cr",
                        "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Sr", "Y", "Zr", "Nb", "Mo", "Ru", "Rh",
                        "Pd", "In", "Sn", "Sb", "Te", "Ba", "La", "Pr", "Nd", "Lu", "Ta", "Re", "Os", "Ir", "Pt",
                        "Au", "Tl", "Pb", "Bi"]

    new_data = new_data.drop(['critical_temp'], axis=1)

    residual_plot()
    Var_Percent_Select(X)
    correlation(data)

    new_data = new_data.drop(['gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass',
                              'wtd_entropy_atomic_mass', 'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie',
                              'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'std_fie', 'wtd_std_fie',
                              'range_ThermalConductivity', 'mean_ThermalConductivity', 'wtd_mean_ElectronAffinity',
                              'mean_atomic_radius', 'wtd_range_fie', 'wtd_entropy_fie', 'mean_atomic_mass',
                              'wtd_mean_atomic_mass', 'number_of_elements', 'wtd_mean_atomic_radius',
                              'gmean_atomic_radius',
                              'wtd_gmean_atomic_radius', 'entropy_atomic_radius', 'wtd_entropy_atomic_radius',
                              'range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius', 'wtd_mean_Density',
                              'gmean_Density', 'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density',
                              'wtd_range_Density', 'std_Density', 'wtd_std_Density', 'gmean_ElectronAffinity',
                              'wtd_gmean_ElectronAffinity', 'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity',
                              'wtd_range_ElectronAffinity', 'std_ElectronAffinity', 'wtd_std_ElectronAffinity',
                              'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat', 'entropy_FusionHeat',
                              'wtd_entropy_FusionHeat', 'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat',
                              'wtd_gmean_ThermalConductivity', 'wtd_range_ThermalConductivity',
                              'std_ThermalConductivity',
                              'wtd_std_ThermalConductivity', 'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence',
                              'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence', 'std_Valence',
                              'wtd_std_Valence', 'H', 'Be', 'C', 'F', 'Al', 'P', 'Ti', 'Cr', 'Zn', 'Ga', 'Pd', 'Te',
                              'Pr',
                              'Nd', 'Ta', 'Re', 'Au', 'Pb', 'As'], axis=1)

    # The new_data set was then made into a csv file.
    # new_data.to_csv('/home/thomas/new_data.csv', index = False)

    final_dataset = new_data.drop(['range_fie', 'entropy_ThermalConductivity', 'range_atomic_mass', 'mean_Density',
                                   'mean_ElectronAffinity', 'mean_FusionHeat', 'wtd_range_atomic_radius',
                                   'wtd_mean_ThermalConductivity', 'range_Density'], axis=1)
    vif()
    breusch_pagan_test(data)

    # Then this dataset was made into the final dataset.
    # final_dataset.to_csv('/home/thomas/final_dataset.csv', index = False)