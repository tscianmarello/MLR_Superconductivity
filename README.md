# MLR_Superconductivity
This was a project performed on the superconductivity dataset found here: https://archive.ics.uci.edu/ml/datasets/superconductivty+data
It is cited as so: Hamidieh, Kam, A data-driven statistical model for predicting the critical temperature of a superconductor, Computational Materials Science, Volume 154, November 2018, Pages 346-354

This repository contains 3 files: the LinearRegressionCorrelation, the PreProcessing, and the AllMLRModelsWithTransformations

The LinearRegressionCorrelation contains all the information regarding the variance threshold selection, the scatter plot diagrams, the heat map, the ols table, the plotted linear
regression, the selected f_regression features, and the selectred mutual_info_regression features.
The features with the highest correlation based on the heat map were: 
wtd_std_ThermalConductivity        .72     Feature #: 70
range_ThermalConductivity          .67     Feature #: 67
std_ThermalConductivity            .64     Feature #: 69
range_atomic_radius                .64     Feature #: 27
wtd_mean_Valence                  -.63     Feature #: 72
wtd_gmean_Valence                 -.62     Feature #: 74

Based on the f_regression, the best features were: 
0:1, 6:7, 17:18, 26:28, 30:31, 67:68, 69:73, 74:77
Values 27:28, 67:68, 70:71 were considered the best. They did not display multicollinearity.
[27:28] = range_atomic_radius, [67:68] = wtd_entropy_ThermalConductivity, [70:71] = wtd_std_ThermalConductivity

Based on the mutual_info_regression, the best features were: 
5:6, 17:18, 19:20, 25:26, 33:34, 35:36, 45:46, 47:48, 55:56, 69:71, 72:73, 74:75
Values 25:26, 69:70 were best features that did not display multicollinearity.

Additional information from LinearRegression Correlation:
1) The scatter plot can be set to include all the features of the dataset: X = data.iloc[0:81], but it will take a very long time to output the result.
2) The heat map plot can be computed for all the features as well and will output a result faster, but you will have to zoom in to see the values.
3) The OLS table outputs the information regardless of the number of features, but the specific features selected here are more important as the statistics
will vary significantly.
4) The lin_regplot can only have 1 dimension as more dimensions will not output a usable plot.
5) The varianceThreshold function can take any number of features.

For the Pre-Processing information:
1) The Breusch-Pagan test prints in the following order:
(Lagrange multiplier Statistic, p-value, f-value, f p-value)
2) The features with more than 5% variance were selected and dropped. The features dropped were:
entropy_atomic_mass, wtd_entropy_atomic_mass, entropy_fie, wtd_entropy_fie, entropy_atomic_radius,
wtd_entropy_atomic_radius, entropy_Density, wtd_entropy_Density, entropy_ElectronAffinity,
wtd_entropy_ElectronAffinity, entropy_FusionHeat, wtd_entropy_FusionHeat, entropy_ThermalConductivity,
wtd_entropy_ThermalConductivity, entropy_Valence, wtd_entropy_Valence, He, Li, N, Ne, Na, Mg, Cl, Ar, K, Sc, Mn, Br,
Kr, Rb, Tc, Ag, Cd, I, Xe, Cs, Ce, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Hf, W, Hg, Po, At, Rn

The features kept were: 
number_of_elements, mean_atomic_mass, wtd_mean_atomic_mass, gmean_atomic_mass, wtd_gmean_atomic_mass,
range_atomic_mass, wtd_range_atomic_mass, std_atomic_mass, wtd_std_atomic_mass, mean_fie, wtd_mean_fie, gmean_fie,
wtd_gmean_fie,range_fie, wtd_range_fie, std_fie, wtd_std_fie, mean_atomic_radius, wtd_mean_atomic_radius,
gmean_atomic_radius, wtd_gmean_atomic_radius, range_atomic_radius, wtd_range_atomic_radius, std_atomic_radius,
wtd_std_atomic_radius, mean_Density, wtd_mean_Density, gmean_Density, wtd_gmean_Density,
range_Density, wtd_range_Density, std_Density, wtd_std_Density, mean_ElectronAffinity, wtd_mean_ElectronAffinity,
gmean_ElectronAffinity, wtd_gmean_ElectronAffinity, range_ElectronAffinity, wtd_range_ElectronAffinity,
std_ElectronAffinity, wtd_std_ElectronAffinity, mean_FusionHeat, wtd_mean_FusionHeat, gmean_FusionHeat
wtd_gmean_FusionHeat, range_FusionHeat, wtd_range_FusionHeat, std_FusionHeat, wtd_std_FusionHeat,
mean_ThermalConductivity, wtd_mean_ThermalConductivity, gmean_ThermalConductivity, wtd_gmean_ThermalConductivity,
range_ThermalConductivity, wtd_range_ThermalConductivity, std_ThermalConductivity, wtd_std_ThermalConductivity,
mean_Valence, wtd_mean_Valence, gmean_Valence, wtd_gmean_Valence, range_Valence, wtd_range_Valence, std_Valence
wtd_std_Valence, H, Be, B, C, O, F, Al, Si, P, S, Ca, Ti, V, Cr, Fe, Co, Ni, Cu, Zn, Ga, Ge, As, Se, Sr, Y, Zr, Nb, Mo,
Ru, Rh, Pd, In, Sn, Sb, Te, Ba, La, Pr, Nd, Lu, Ta, Re, Os, Ir, Pt, Au, Tl, Pb, Bi

3) Features were then dropped based on VIF, only keeping those features that displayed a VIF less than 10.

For the AllMLRModelsWithTransformation, these were the best alphas for each of the transformations:

Ridge selected alphas for each transformation:
Non-transformed y alpha outputted: 108.12
np.log(y) alpha outputted: 212.61
np.square alpha outputted: 116.7
np.sqrt(y) alpha outputted: 153.2
boxcox(y) alpha outputted: 139.4
yeojohnson(y) alpha outputted: 108.12
Uniform Quantile Transform alpha outputted: 196.3
Normal Quantile Transform alpha outputted: 179.9

Lasso selected alphas for each transformation:
non-transformed y alpha outputted: .753
np.log(y) alpha outputted: .026
np.square(y) alpha outputted: 35.1
np.sqrt(y) alpha outputted: 0.05
boxcox(y) alpha outputted: .13
yeojohnson(y) alpha outputted: .52
Uniform Quantile Transform alpha outputted: .005
Normal Quantile Transform alpha outputted: .015

Bayesian-ridge selected alphas for each transformation:
non-transformed y parameters used: (alpha_1=1, alpha_2= 1, lambda_1=900, lambda_2=100)
np.log(y) parameters used: (alpha_1=1, alpha_2= 1, lambda_1=900, lambda_2=100)
np.square(y) parameters used: (alpha_1=1, alpha_2= 1, lambda_1=10, lambda_2=10)
np.sqrt(y) parameters used: (alpha_1=1, alpha_2= 1, lambda_1=50, lambda_2=10)
boxcox(y) parameters used: (alpha_1=1, alpha_2= 1, lambda_1=10, lambda_2=10)
yeojohnson(y) parameters used: (alpha_1=1, alpha_2= 0, lambda_1=1, lambda_2=10)
Uniform Quantile Transform parameters used: (alpha_1=0, alpha_2= 0, lambda_1=1, lambda_2=10)
Normal Quantile Transform parameters used: (alpha_1=1, alpha_2= 1, lambda_1=10, lambda_2=10)

Elastic Net selected alphas for each transformation:
Non-transformed y alpha outputted: .253
np.log(y) alpha outputted: .041
np.square(y) alpha outputted: .491
np.sqrt(y) alpha outputted: .076
boxcox(y) alpha outputted: .14
yeojohnsons (y) alpha outputted: .253
Uniform Quantile Transform alpha outputted: .01
Normal Quantile Transform alpha outputted: .03

Multi-task Lasso selected alphas for each transformation:
non-transformed(y) alpha outputted: .516
np.log(y) alpha outputted: .026
np.square alpha outputted: 35.1
np.sqrt(y) alpha outputted: .05
boxcox(y) alpha outputted: alpha outputted: .491
yeojohnson(y) alpha outputted: .516
Uniform Quantile Transform alpha outputted: .005
Normal Quantile Transform alpha outputted: .015

Multi-task Elastic-Net selected alphas for each transformation:
non-transformed y alpha outputted: .253
np.log(y) alpha outputted: .041
np.square(y) alpha outputted: .491
np.sqrt(y) alpha outputted: .076
boxcox(y) alpha outputted: .076
yeojohnson(y) alpha outputted: .253
Uniform Quantile Transform alpha outputted: .01
Normal Quantile Transform alpha outputted: .03

XGBoost best selected alphas:
non-transformed y reg_alpha: .844
np.log(y) reg_alpha:.000001
np.square(y) reg_alpha=1
np.sqrt(y) reg_alpha=.001
boxcox(y) reg_alpha=.00001
yeojohnson(y) reg_alpha=.001
Uniform Quantile Transform reg_alpha=.001
Normal Quantile Transform reg_alpha=.0001
