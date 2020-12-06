# -*- coding: utf-8 -*-

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load in the data
df = pd.read_csv("wine_quality-red.csv")

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, Y_train, Y_test = train_test_split(df, y, test_size = 0.2)


# Fit a model on the train section
regr = RandomForestRegressor(max_depth = 2)
regr.fit(X_train, Y_train)

# Report training set score
train_score = regr.score(X_train, Y_train) * 100
# Report test set score
test_score = regr.score(X_test, Y_test) * 100

# Write scores to a file
with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)

# Calculate feature importance in random forest
importances = regr.feature_importances_
labels = df.columns
feature_df = pd.DataFrame(list(zip(labels, importances)), columns = ["feature", "importance"])
feature_df = feature_df.sort_values(by = 'importance', ascending = False,)

# image formatting
axis_fs = 18 #fontsize
title_fs = 22 #fontsize
sns.set(style = "whitegrid")

ax = sns.barplot(x = "importance", y = "feature", data = feature_df)
ax.set_xlabel('Importance', fontsize = axis_fs) 
ax.set_ylabel('Feature', fontsize = axis_fs)#ylabel
ax.set_title('Random forest\nfeature importance', fontsize = title_fs)

plt.tight_layout()
plt.savefig("feature_importance.png", dpi = 120) 
plt.close()