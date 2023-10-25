import matplotlib.pyplot as plt
import numpy as np
import scipy as scp
import sklearn as skl
import sklearn.datasets
import sklearn.naive_bayes
import seaborn as sns

sns.set_style('darkgrid')

data = scp.io.loadmat('ExamData3D.mat')

# print(data)

iris = skl.datasets.load_iris(return_X_y=False)

# print(iris)


xtrain = np.swapaxes(np.array(data['X_train']), 0, 1)
xtest = np.swapaxes(np.array(data['X_test']), 0, 1)
ytrain = np.array([fucked_value - 1 for fucked_value in data['Y_train'][0]])
ytest = np.array([fucked_value - 1 for fucked_value in data['Y_test'][0]])

X_test = data['X_test']
X_train = data['X_train']
Y_test = data['Y_test']
Y_train = data['Y_train']

# print(X_test.shape)
# print(Y_test)

# Extract the x and y values from X_test for plotting
feature_x1 = X_test[0]
feature_x2 = X_test[1]
feature_x3 = X_test[2]

print(feature_x1.shape)

print('\n defining classes...\n')

# Class labels
class_labels = [1, 2, 3]
print(f' we have {len(class_labels)} classes with labels {class_labels} \n')

# defining feature combinations
combinations = [(1,), (2,), (3,), (1, 2), (2, 3), (1, 3), (1, 2, 3)]
# colors = ['r','g','b']  # set color for each class
colors = 'muted'

# 1a
l, N = X_train.shape
c = np.max(Y_train)

m = np.zeros((l, c))
Sw = np.zeros((l, l))

for c in class_labels:
    Y_temp = np.where(Y_train == c)[1]
    X_temp = X_train[:, Y_temp]
    P = len(Y_temp) / N
    m[:, c - 1] = np.mean(X_temp, axis=1)
    Sw += P * np.cov(X_temp)

print(Sw)

m0 = np.sum((np.ones((l, 1)) * P * m.T).T, axis=0)
Sb = np.zeros((l, l))

for c in class_labels:
    Y_temp = np.where(Y_train == c)[1]
    Sb += P*np.outer(m[:, c - 1] - m0, m[:, c - 1] - m0)

Sm = Sw + Sb
# J3 = np.trace(np.dot(np.linalg.inv(Sw),Sm))
J3 = np.trace(np.linalg.inv(Sw) @ Sm)

print(f'\n the j3 for 3 separable classes is {J3:.3}...')

# J3_min / J3 for inseparable classes
# Sb = 0
Sb = np.zeros_like(Sw)
Sm = Sw + Sb

# J3_min = np.trace(np.dot(np.linalg.inv(Sw),Sm))
J3_min = np.trace(np.linalg.inv(Sw) @ Sm)
print(f'The expected minimum value for j3 is {J3_min}')

# Thus
J3_new = J3 - J3_min
print(f'The new J3 is thus {J3_new}')

# importance of this new j3:
'''
J3 - Separability when classes are distinct
J3_min - Separability when classes are identical
J3_new - Improvement in class separabilit achieved by making classes more distinct

Thus the modified version of J3 can help assess relative change in class separability
for feature selection or dimensionality reduction
'''

# print(' \n Plotting X_test without classes...')
# plt.figure()
# plt.scatter(X_test[0], X_test[1])
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Scatter Plot of X_test')
# plt.savefig('PATTERN HOMEEXAM/img/X_test')

# def plot_hist(feature):
#     plt.figure()
#     plt.hist(X_train[:, feature], bins=20)  # Adjust the number of bins as needed
#     plt.xlabel('Feature 1')
#     plt.ylabel('Frequency')
#     plt.title('Histogram of Feature 1 in X_train')
#     plt.savefig(f'PATTERN HOMEEXAM/img/Feature_{feature}_Hist')


# 1b)

J3_scores = []
feature_pairs = [(1, 2), (2, 3), (1, 3)]

for feature_1, feature_2 in feature_pairs:
    X_train_pair = X_train[[feature_1 - 1, feature_2 - 1], :]

    l, N = X_train_pair.shape
    c = np.max(Y_train)

    m = np.zeros((l, c))
    Sw = np.zeros((l, l))

    for c in class_labels:
        Y_temp = np.where(Y_train == c)[1]
        X_temp = X_train_pair[:, Y_temp]
        P = len(Y_temp) / N

        m[:, c - 1] = np.mean(X_temp, axis=1)
        Sw += P * np.cov(X_temp)

    # print(f'Within class matrix: {Sw}')

    # m0 = np.sum((np.ones((l,1)) * P * m.T).T, axis=0)
    m0 = np.mean(m, axis=1)
    Sb = np.zeros((l, l))

    for c in class_labels:
        Y_temp = np.where(Y_train == c)[1]
        Sb += P * np.outer(m[:, c - 1] - m0, m[:, c - 1] - m0)

    # print(f'Between class matrix for feature {feature_1,feature_2}: {Sb}')
    Sm = Sw + Sb
    J3 = np.trace(np.linalg.inv(Sw) @ Sm)
    # print(J3)

    Sb_ = 0
    Sm = Sw + Sb_
    # print(f'Mixture matrix for feature {feature_1,feature_2}: {Sm}')

    J3_min = np.trace(np.linalg.inv(Sw) @ Sm)
    print(f'Minimum separability J3_min = {J3_min}')

    J3_new = J3 - J3_min
    print(J3_new)

    J3_scores.append(J3_new)

color_pallette = sns.color_palette("muted")

# Create subplots in a single figure
# ax1.hist(X_train[200:400, :3], bins=20, color=color_pallette[:3])
# print(np.shape(X_train[:, :3]))
# print(X_train[:200, :3])
# ax1.set_title('SCE')
# ax2.hist(X_train[400:, :3], bins=20, color=color_pallette[:3])
# print(count1)
# count2, bin2 = np.histogram(X_train[200:400, 1], bins=20)
# print(count2)
# count3, bin3 = np.histogram(X_train[400:, 2], bins=20)
# print(count3)
# counts = [count1, count2, count3]
# bins = [bin1, bin2, bin3]
# ax.hist(X_train[:, 0], bins=20, alpha=1, label=f'Feature 1', color=color_pallette[0])
# ax.hist(X_train[:, 1], bins=20, alpha=1, label=f'Feature 2', color=color_pallette[1])
# ax.hist(X_train[:, 2], bins=20, alpha=1, label=f'Feature 3', color=color_pallette[2])
# ax.hist(X_train[:, :3], bins=20, color=color_pallette[:3])
# fig.savefig('img/good.png')
# # ax
# plt.figure(figsize=(15, 5))
# fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15, 5))
# # Plot the histogram for Feature 1
# # plt.subplot(1, 3, 1)

# # for label in class_labels:
# ax1.hist(X_train[:, 0], bins=20, alpha=1, label=f'Class 1', color=color_pallette[0])
# ax1.hist(X_train[:, 0], bins=20, alpha=1, label=f'Class 2', color=color_pallette[1])
# ax1.hist(X_train[:, 0], bins=20, alpha=1, label=f'Class 3', color=color_pallette[2])


# # ax1.xlabel('Feature 1')
# # ax1.ylabel('Frequency')
# # ax1.title('Histogram of Feature 1 in X_train')
# # plt.legend()

# # Plot the histogram for Feature 2
# plt.subplot(1, 3, 2)
# for label in class_labels:
#     plt.hist(X_train[:, 1], bins=20, alpha=1, label=f'Class {label}', color=color_pallette[label])
# plt.xlabel('Feature 2')
# plt.ylabel('Frequency')
# plt.title('Histogram of Feature 2 in X_train')
# # plt.legend()


# # Plot the histogram for Feature 3
# plt.subplot(1, 3, 3)
# # for label in class_labels:
# plt.hist(X_train[:, 2], bins=20, alpha=1, label=f'Class {label}',color=color_pallette[label])
# plt.xlabel('Feature 3')
# plt.ylabel('Frequency')
# plt.title('Histogram of Feature 3 in X_train')
# plt.legend()

# fig, ((ax0, ax1, ax2), (ax10, ax11, ax12)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 10), layout='constrained')
# c0, b0 = np.histogram(X_train[0, :200], bins=20)
# print(c0)
# c1, b1 = np.histogram(X_train[0, 200:400], bins=20)
# print(c1)
# c2, b2 = np.histogram(X_train[0, 400:], bins=20)
# print(c2)
# plt.tight_layout()
# plt.savefig('img/histogram.png')
# fig.savefig('img/histogram1.png')
# plt.show()

# plot_hist(3)

# print('\n Plotting 2D scatterplots...')

# print(X_train[:,2])


# Iterate through the data points and plot them with different colors based on their class labels
# sns.scatterplot(x=feature_x1, y=feature_x2, hue=Y_test.flatten(), palette=colors)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('2D Scatter Plot of X_test')
# plt.legend(title='Class')
# plt.savefig('PATTERN HOMEEXAM/img/2D_Scatt_F1_vs_F2')
# plt.show()

# # Create a Seaborn scatter plot
# sns.scatterplot(x=feature_x1, y=feature_x3, hue=Y_test.flatten(), palette=colors)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 3')  # Update the y-axis label to Feature 3
# plt.title('2D Scatter Plot (Feature 1 vs. Feature 3)')
# plt.legend(title='Class')
# plt.savefig('PATTERN HOMEEXAM/img/2D_Scatt_F1_vs_F3')
# plt.show()

# # Create a Seaborn scatter plot
# sns.scatterplot(x=feature_x2, y=feature_x3, hue=Y_test.flatten(), palette=colors)
# plt.xlabel('Feature 2')
# plt.ylabel('Feature 3')  # Update the y-axis label to Feature 3
# plt.title('2D Scatter Plot (Feature 2 vs. Feature 3)')
# plt.legend(title='Class')
# plt.savefig('PATTERN HOMEEXAM/img/2D_Scatt_F2_vs_F3')
# plt.show()


# Create a 1x3 grid of subplots
fig, (axes, (ax0, ax1, ax2)) = plt.subplots(2, 3, figsize=(15, 10), layout='constrained')
# fig.suptitle('Figure 1: 2D Feature Scatterplot', verticalalignment='bottom')
fig.set_label('ZOINKSSS')
fig.legend(loc='center right')
import matplotlib.image as imge
# image = imge.imread('img/cursed-emoji.png')
# print(np.shape(image[]))

def filter_white(c):
    if c[0] > 0.85 and c[1] > 0.85 and c[2] > 0.85:
        return [c[0], c[1], c[2], 0.0]
    else:
        return [1.0 - (c[0] ** 2.0), 1.0 - (c[1] ** 2.0), 1.0 - (c[2] ** 2.0), 1.0]
image = [[filter_white(color) for color in row] for row in imge.imread('img/cursed-emoji.png')]
# print(np.shape(xdx))
# from matplotlib import image
# image.imread
fig.figimage(image, xo=500)
fig.set_rasterized(True)

# Scatter plot 1: Feature 1 vs. Feature 2
sns.scatterplot(x=feature_x1, y=feature_x2, hue=Y_test.flatten(), palette=colors, ax=axes[0])
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
# axes[0].set_title(f'Feature 1 vs. Feature 2 | J3 score = {J3_scores[0]:.4}')
axes[0].set_title(f'J3 score = {J3_scores[0]:.4}')
axes[0].legend(title='Class')

# Scatter plot 2: Feature 1 vs. Feature 3
sns.scatterplot(x=feature_x2, y=feature_x3, hue=Y_test.flatten(), palette=colors, ax=axes[1])
axes[1].set_xlabel('Feature 2')
axes[1].set_ylabel('Feature 3')
# axes[1].set_title(f'Feature 2 vs. Feature 3 | J3 score = {J3_scores[1]:.4}')
axes[1].set_title(f'J3 score = {J3_scores[1]:.4}')
axes[1].legend(title='Class')

# Scatter plot 3: Feature 2 vs. Feature 3
sns.scatterplot(x=feature_x1, y=feature_x3, hue=Y_test.flatten(), palette=colors, ax=axes[2])
axes[2].set_xlabel('Feature 1')
axes[2].set_ylabel('Feature 3')
# axes[2].set_title(f'Feature 1 vs. Feature 3 | J3 score = {J3_scores[2]:.4}')
axes[2].set_title(f'J3 score = {J3_scores[2]:.4}')
axes[2].legend(title='Class')

ws = [np.ones(200) / 600, np.ones(200) / 600, np.ones(200) / 600]
ax0.hist([X_train[0, :200], X_train[0, 200:400], X_train[0, 400:]], bins=20, color=color_pallette[:3], label=['1', '2', '3'], rwidth=0.9, weights=ws)
ax0.legend(title='Class')
from matplotlib.ticker import PercentFormatter
ax0.yaxis.set_major_formatter(PercentFormatter(xmax=1))
# ax0.hist(X_train[1, 200:400], bins=20, color=color_pallette[1])
# ax0.hist(X_train[2, 400:], bins=20, color=color_pallette[2])
ax0.set_title('THE', loc='left')
# ax1.hist(X_train[0, :200], bins=20, color=color_pallette[0])
ax1.hist([X_train[1, :200], X_train[1, 200:400], X_train[1, 400:]], bins=20, color=color_pallette[:3], label=['1', '2', '3'], rwidth=0.9, weights=ws)
ax1.legend(title='Class')
ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1))
# ax1.hist(X_train[1, :], bins=20, color=color_pallette[1])
# ax1.hist(X_train[2, 400:], bins=20, color=color_pallette[2])
ax1.set_title('LEGEND', loc='left')
# ax2.hist(X_train[0, :200], bins=20, color=color_pallette[0])
# ax2.hist(X_train[1, 200:400], bins=20, color=color_pallette[1])
# ax2.hist(X_train[2, :], bins=20, color=color_pallette[2])
ax2.hist([X_train[2, :200], X_train[2, 200:400], X_train[2, 400:]], bins=20, color=color_pallette[:3], label=['1', '2', '3'], rwidth=0.9, weights=ws)
ax2.legend(title='Class')
ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax2.set_title('27', loc='left')
fig.savefig('img/suckmynuts.png')
# Adjust layout and save the figure
# plt.tight_layout()
# plt.savefig('img/2D_Scatt')
# plt.show()
