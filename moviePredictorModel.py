import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm 
from math import ceil, sqrt
# from sklearn.ensemble import RandomForestRegressor

# NOTE: PROMPT: create a machine learning model to predict a particular movie’s revenue and rating based on 
# historical data of the genre. This information can help movie production companies decide what movies 
# to invest in based on how well they draw in an audience.

data = pd.read_excel("movie_data.xlsx", sheet_name="Movie Ratings")
get_revenue = pd.read_excel("movie_data.xlsx", sheet_name="Movie Revenues")
movieId = pd.read_excel("movie_data.xlsx", sheet_name="Movie Ratings")["movie_id"]

X = data.drop(columns=["movie_id", "vote_average"])
ratings = data["vote_average"]
revenues = get_revenue["revenue"]

# Train set: the data is seen and learned by the model 
# Test set: a subset of the traning data set, used to give evaluation of a final model fit
# Validation set: sample of data from model's training set that is used to estimate model performance while tuning the model's hyperparameter
# NOTE: tunning hyperparameter: the process of selecting the optimal values for a machine learning models 
X_train, X_test, ratings_train, ratings_test = train_test_split(X , ratings , test_size=0.2, random_state=42)
X2_train, X2_test, rev_train, rev_test, movieId_train, movieId_test = train_test_split(X, revenues, movieId, test_size=0.2, random_state=42)


linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train,ratings_train)
test_score = linear_regression_model.score(X_test,ratings_test)
train_score = linear_regression_model.score(X_train, ratings_train)
print("Comparing the R^2 of test and train set")
print(f"R^2 score on test set: {test_score:.4f}")
print(f"R^2 score on train set: {train_score:.4f} ")
print(f"This is beta_naught: {linear_regression_model.intercept_:.4f}")
print(f"Since the Train set and Test set have close R^2 value: {train_score:.4f} vs {test_score:.4f}")
print("The model generalize well with the unseen data")

coefficients = pd.Series(linear_regression_model.coef_, index=X.columns)
print("Feature Coefficients: ")
print(coefficients.sort_values(ascending=False))

# NOTE: Linear regression model for revenue 
print("===========================")
rev_model = LinearRegression()
rev_model.fit(X2_train,rev_train)
rev_test_score = rev_model.score(X2_test, rev_test)
rev_train_score = rev_model.score(X2_train, rev_train)
print("Comparing R^2 value of revenues between test and train sets")
print(f"R^2 on train set: {rev_train_score:.4f}")
print(f"R^2 on test set: {rev_test_score:.4f}")
print(f"This is beta_naught: {rev_model.intercept_:.4f}")

print("==============")

# NOTE: residual calculation 
ratings_hat = linear_regression_model.predict(X=X_train)
residual = ratings_train - ratings_hat 

rev_hat = rev_model.predict(X=X2_train)
rev_residual = rev_train - rev_hat

# NOTE: seting index to movie_id instead of standard indexing, optional 
# rev_residual1 = pd.Series(rev_residual.values, index=movieId_train)
# rev_hat = pd.Series(rev_hat,index=movieId_train)
# print(f"This rev residual values: {rev_residual.values}")


def sigma(res):
    avg = np.mean(res)
    return avg

def get_bin_histo(residual, binNumber): 
    '''Function calculation bin range for hist() function'''
    bin = []
    # Residual always have negative and positive values 
    width = (max(residual) + abs(min(residual))) / binNumber # keep the decimal point
    for i in range(binNumber + 1): 
        bin.append(min(residual) + (width*i))
    return bin
sample_average = sigma(residual)
print(f"Sample average(should be almost 0): {sample_average}")

bin_num = ceil(sqrt(len(X_train)))

# NOTE: making the plots and histograms 
fig2, (rev_plot,rev_hist) = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
fig2.subplots_adjust(wspace=0.4)

rev_plot.scatter(rev_hat, rev_residual, color="pink" , label="Residual vs. y_hat" )
rev_plot.axhline(y=0, color='red', linestyle="--", linewidth=1.5, label="Zero Line")
rev_plot.set_title("Rev Residual vs. y_hat")
rev_plot.set_xlabel("y_hat")
rev_plot.set_ylabel("Residual")
rev_plot.grid(True)
rev_plot.legend()

rev_bin = get_bin_histo(rev_residual, binNumber=bin_num)
rev_hist.hist(rev_residual, bins=rev_bin, edgecolor='black', color = "gold", label="Rev Residual Histogram")
rev_hist.set_title('Rev Histogram of Residual')
rev_hist.grid(True)
rev_hist.set_xlabel("Resiudal values")
rev_hist.set_ylabel("Frequency")
rev_hist.set_xlim(rev_bin[0],rev_bin[-1])

# NOTE: ANOVA
X_const = sm.add_constant(X_train)
# fit the model, OLS = Ordinary Least Square, or simple linear regression 
ratings_model = sm.OLS(ratings_train,X_const).fit() #OLS fit() , does the math: R square, F statistic, etc. 
rev_model_OLS = sm.OLS(rev_train, X_const).fit()

features_to_drop= []
# NOTE: droping highest p-values regressors 
while True: 
    ratings_p_values= ratings_model.pvalues.drop('const') # drop p-value of Beta_naught
    max_p_feature=  ratings_p_values.idxmax()
    features_to_drop.append(max_p_feature)

    max_p_val = ratings_p_values[max_p_feature]
    print(f"Max p-value: {max_p_val:.4f} for feature: {max_p_feature}")
    if max_p_val > 0.05: X_const = X_const.drop(columns=max_p_feature)
    else: break
    ratings_model = sm.OLS(ratings_train,X_const).fit()


X_adjust = X.drop(features_to_drop, axis=1)

# NOTE: retrain the model after removing the non-contributing genres
X_train, X_test, ratings_train, ratings_test = train_test_split(X_adjust , ratings , test_size=0.2, random_state=42)
linear_regression_model.fit(X_train, ratings_train)
test_score = linear_regression_model.score(X_test,ratings_test)
train_score = linear_regression_model.score(X_train, ratings_train)
print("Comparing the R^2 of test and train set")
print(f"R^2 score on test set: {test_score:.4f}")
print(f"R^2 score on train set: {train_score:.4f} ")


ratings_hat = linear_regression_model.predict(X=X_train)
residual = ratings_train - ratings_hat

# NOTE: Set up plot for Ratings 
fig, axs = plt.subplots(nrows=1, ncols=2, figsize= (12,5))

axs[0].scatter(ratings_hat, residual, color="blue", label="Residual vs. y_hat")
axs[0].axhline(y=0, color="red", linestyle="--", linewidth=1, label="Zero Line")
axs[0].grid(True)
axs[0].legend()
axs[0].set_title("Residual plot")
axs[0].set_xlabel("y_hat")
axs[0].set_ylabel("Residual")

bin = get_bin_histo(residual=residual, binNumber=bin_num)
axs[1].hist(residual, bins=bin, edgecolor="black",color="#96DD98" , label = "Residual Histogram")
axs[1].grid(True)
axs[1].legend()
axs[1].set_title("Histogram of Residual")
axs[1].set_xlabel("Residual value")
axs[1].set_ylabel("Frequency")
axs[1].set_xlim(bin[0],bin[-1])

# fig.savefig('adjusted_ratings_residual_plot.png')

# plt.tight_layout()
# plt.show()

# NOTE: turn ANOVA into image 
fig3, OLS_rating = plt.subplots()
OLS_rating.axis("off") # no axes 

OLS_rating.text(0,1, ratings_model.summary().as_text(), fontsize=10, fontfamily='monospace', va='top')
# fig3.savefig("adjusted_OLS_ratings.png",bbox_inches= 'tight')

fig4, OLS_rev = plt.subplots()
OLS_rev.axis('off')

OLS_rev.text(0,1,rev_model_OLS.summary().as_text(), fontsize=10,fontfamily='monospace',va='top')
# fig4.savefig('OLS_rev.png',bbox_inches='tight')