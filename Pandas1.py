# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و بهِ نَستَعين

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

np.random.seed(0)
pd.set_option('display.max_columns', None)  # Display all columns in a dataframe
pd.set_option('expand_frame_repr', False)  # Display the dataframe records in the same line
plt.style.use('seaborn')
plt.rcParams["figure.autolayout"] = True

train_df = pd.read_csv("housing_train.csv")
train_df.info()

# Notice that the data entries are sorted by "longitude", resulting in undesired regularity in the weight updates during training
# These regularities cause overfitting since the batches are not good representations of the entire dataset
print(train_df.head(10))

# Shuffle the data before training:
train_df = train_df.sample(frac=1)

# Scale the label/target ("median_house_value") to be in the range of thousands:
train_df["median_house_value"] /= 1000  # Scaling will keep the loss values and learning rates more stable

# Notice that the maximum values of some features are very large compared to the feature's quantiles, indicating possible anomalies in the features
print(train_df.describe())


class BuildModel:
    def __init__(self, lr):
        self.lr = lr
        self.w = np.random.rand()
        self.b = 0
        self.data_pts = len(train_df)
        self.losses = []

    def fit(self, features, labels, epochs, bs):
        v_w = 0
        v_b = 0
        eps = 1e-8
        beta = 0.999
        for _ in trange(epochs):
            updates = self.data_pts // bs
            loss_mean = 0
            for i in range(updates):
                x = features.iloc[bs*i: bs*(i+1)]
                y = labels.iloc[bs*i: bs*(i+1)]
                y_hat = self.w * x + self.b
                loss = np.mean((y_hat - y)**2)/2
                loss_mean += loss

                g_w = np.mean((y_hat - y) * x)  # gradient
                v_w = beta*v_w + (1 - beta)*abs(g_w)  # second moment

                g_b = np.mean(y_hat - y)
                v_b = beta*v_b + (1 - beta)*abs(g_b)

                self.w -= self.lr * g_w/(v_w + eps)
                self.b -= self.lr * g_b/(v_b + eps)
            self.losses.append(loss_mean/updates)

    def predict(self, x):
        return self.w * x + self.b


def train_model(model: BuildModel, df: pd.DataFrame, feature: str, label: str, epochs: int, batch_size: int):
    model.fit(df[feature], df[label], epochs, batch_size)
    return model, model.losses


def plot_model(model: BuildModel, feature: str, label: str):
    random_samples = train_df.sample(200)
    x = random_samples[feature]
    y = random_samples[label]
    plt.scatter(x, y)
    x0 = x.min()
    x1 = x.max()
    y0 = model.predict(x0)  # Predict min point and max point to draw a line between them and compare it against scatter plot
    y1 = model.predict(x1)
    plt.plot([x0, x1], [y0, y1], color='orange')
    plt.show()# بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ و بهِ نَستَعين

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import trange

np.random.seed(0)
pd.set_option('display.max_columns', None)  # Display all columns in a dataframe
pd.set_option('expand_frame_repr', False)  # Display the dataframe records in the same line
plt.style.use('seaborn')
plt.rcParams["figure.autolayout"] = True

train_df = pd.read_csv("housing_train.csv")
train_df.info()

# Notice that the data entries are sorted by "longitude", resulting in undesired regularity in the weight updates during training
# These regularities cause overfitting since the batches are not good representations of the entire dataset
print(train_df.head(10))

# Shuffle the data before training:
train_df = train_df.sample(frac=1)

# Scale the label/target ("median_house_value") to be in the range of thousands:
train_df["median_house_value"] /= 1000  # Scaling will keep the loss values and learning rates more stable

# Notice that the maximum values of some features are very large compared to the feature's quantiles, indicating possible anomalies in the features
print(train_df.describe())


class BuildModel:
    def __init__(self, lr):
        self.lr = lr
        self.w = np.random.rand()
        self.b = 0
        self.data_pts = len(train_df)
        self.losses = []

    def fit(self, features, labels, epochs, bs):
        v_w = 0
        v_b = 0
        eps = 1e-8
        beta = 0.999
        for _ in trange(epochs):
            updates = self.data_pts // bs
            loss_mean = 0
            for i in range(updates):
                x = features.iloc[bs*i: bs*(i+1)]
                y = labels.iloc[bs*i: bs*(i+1)]
                y_hat = self.w * x + self.b
                loss = np.mean((y_hat - y)**2)/2
                loss_mean += loss

                g_w = np.mean((y_hat - y) * x)  # gradient
                v_w = beta*v_w + (1 - beta)*abs(g_w)  # second moment

                g_b = np.mean(y_hat - y)
                v_b = beta*v_b + (1 - beta)*abs(g_b)

                self.w -= self.lr * g_w/(v_w + eps)
                self.b -= self.lr * g_b/(v_b + eps)
            self.losses.append(loss_mean/updates)

    def predict(self, x):
        return self.w * x + self.b


def train_model(model: BuildModel, df: pd.DataFrame, feature: str, label: str, epochs: int, batch_size: int):
    model.fit(df[feature], df[label], epochs, batch_size)
    return model, model.losses


def plot_model(model: BuildModel, feature: str, label: str):
    random_samples = train_df.sample(200)
    x = random_samples[feature]
    y = random_samples[label]
    plt.scatter(x, y)
    x0 = x.min()
    x1 = x.max()
    y0 = model.predict(x0)  # Predict min point and max point to draw a line between them and compare it against scatter plot
    y1 = model.predict(x1)
    plt.plot([x0, x1], [y0, y1], color='orange')
    plt.show()


def plot_loss(losses):
    plt.semilogy(losses, c='g', label="MSE Loss")
    print(f"Final loss: {losses[-1]}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


m = BuildModel(0.01)
my_feature = "total_rooms"
my_label = "median_house_value"

trained_model, losses = train_model(m, train_df, my_feature, my_label, epochs=3, batch_size=30)

plot_model(trained_model, my_feature, my_label)
plot_loss(losses)


# We can see that the "total_rooms" features had little predictive power, we should try using a different feature:
m = BuildModel(0.05)
my_feature = "population"

trained_model, losses = train_model(m, train_df, my_feature, my_label, epochs=5, batch_size=3)

plot_model(trained_model, my_feature, my_label)
plot_loss(losses)


# The "population" feature also turned out to be bad, maybe we should create a synthetic feature based on "rooms_per_person":
train_df["rooms_per_person"] = train_df["total_rooms"]/train_df["population"]

m = BuildModel(0.05)
my_feature = "rooms_per_person"

trained_model, losses = train_model(m, train_df, my_feature, my_label, epochs=5, batch_size=30)

plot_model(trained_model, my_feature, my_label)
plot_loss(losses)


# The hand-crafted feature turned out to be better but still not sufficient
# Instead of guess work and relying on trial-and-error, let's use statistics
# Let's view our feature's correlation matrix and determine which feature has the highest correlation with our label/target
sns.clustermap(train_df.corr(method='spearman', min_periods=1), annot=True)  # "pearson" correlation is heavily influenced by outliers!
plt.title('Cross-Correlation Heatmap')
plt.show()

# As can be seen from the correlation matrix heatmap, the "median_income" feature has the highest correlation with the label "median_house_value"
# Another way to determine which feature might be best used for predicting the target is by looking at the R^2 value of fitting a feature against the target



def plot_loss(losses):
    plt.semilogy(losses, c='g', label="MSE Loss")
    print(f"Final loss: {losses[-1]}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


m = BuildModel(0.01)
my_feature = "total_rooms"
my_label = "median_house_value"

trained_model, losses = train_model(m, train_df, my_feature, my_label, epochs=3, batch_size=30)

plot_model(trained_model, my_feature, my_label)
plot_loss(losses)


# We can see that the "total_rooms" features had little predictive power, we should try using a different feature:
m = BuildModel(0.05)
my_feature = "population"

trained_model, losses = train_model(m, train_df, my_feature, my_label, epochs=5, batch_size=3)

plot_model(trained_model, my_feature, my_label)
plot_loss(losses)


# The "population" feature also turned out to be bad, maybe we should create a synthetic feature based on "rooms_per_person":
train_df["rooms_per_person"] = train_df["total_rooms"]/train_df["population"]

m = BuildModel(0.05)
my_feature = "rooms_per_person"

trained_model, losses = train_model(m, train_df, my_feature, my_label, epochs=5, batch_size=30)

plot_model(trained_model, my_feature, my_label)
plot_loss(losses)


# The hand-crafted feature turned out to be better but still not sufficient
# Instead of guess work and relying on trial-and-error, let's use statistics
# Let's view our feature's correlation matrix and determine which feature has the highest correlation with our label/target
sns.clustermap(train_df.corr(method='spearman', min_periods=1), annot=True)  # "pearson" correlation is heavily influenced by outliers!
plt.title('Cross-Correlation Heatmap')
plt.show()

# As can be seen from the correlation matrix heatmap, the "median_income" feature has the highest correlation with the label "median_house_value"
# Another way to determine which feature might be best used for predicting the target is by looking at the R^2 value of fitting a feature against the target
