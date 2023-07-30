import numpy as np
import pandas as pd

dataset = [[32, 50391, 3, 'Male', 'yes'],
           [23, 47785, 5, 'Female', 'yes'],
           [50, 28141, 4, 'Male', 'yes'],
           [42, 32003, 3, 'Female', 'no'],
           [21, 45268, 5, 'Male', 'no'],
           [56, 33184, 4, 'Male', 'yes'],
           [19, 78637, 2, 'Female', 'no'],
           [45, 42442, 1, 'Male', 'yes'],
           [26, 23647, 5, 'Female', 'yes'],
           [57, 56611, 1, 'Female', 'no']
           ]
label = ['age', 'income', 'education', 'gender', 'label']
dataset = np.array(dataset)
print(dataset)
df = pd.DataFrame(dataset, columns=label)


# left_data = df[df['income'] <= '33184']
# right_data = df[df['gender'] > 'Female']
# print(left_data['label'].value_counts())
# print(right_data['label'].mode())


def build_decision_tree(data):
    if len(set(data['label'])) == 1:
        return data['label'].iloc[0]

    if len(data.columns) == 1:
        return data[label].mode().iloc[0]

    best_feature = None
    best_split_value = None
    best_gini = 1

    for feature in data.columns[:-1]:
        unique_values = data[feature].unique()
        for value in unique_values:
            left_data = data[data[feature] <= value]
            right_data = data[data[feature] > value]
            left_gini = gini_index(left_data['label'])
            right_gini = gini_index(right_data['label'])
            weighted_gini = (len(left_data) / len(data)) * left_gini + (len(right_data) / len(data)) * right_gini

            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature
                best_split_value = value

    if best_gini >= gini_index(data['label']):
        return data['label'].mode().iloc[0]

    left_data = data[data[best_feature] <= best_split_value]
    right_data = data[data[best_feature] > best_split_value]

    left_subtree = build_decision_tree(left_data)
    right_subtree = build_decision_tree(right_data)

    return best_feature, best_split_value, left_subtree, right_subtree


def gini_index(labels):
    total_samples = len(labels)
    if total_samples == 0:
        return 0.0
    label_counts = labels.value_counts()
    gini = 1

    for count in label_counts:
        prob = count / total_samples
        gini -= prob ** 2

    return gini


decision_tree = build_decision_tree(df)
print(decision_tree)
