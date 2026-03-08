import numpy as np
import pandas as pd

class Node:
    def __init__(self, feature = None, threshold = None, value = None):
        self.feature = feature
        self.split = threshold
        self.children = {}
        self.value = value

class DecisionTree:
    def __init__(self, max_depth = None, criterion = 'gini'):
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def fit(self, X, y, index_categorical, index_numerical):
        self.columns = X.columns
        self.n_features = X.shape[1]
        self.index_categorical = index_categorical
        self.index_numerical = index_numerical
        X = np.array(X)
        y = np.array(y)
        self.root = self.build_tree(X, y, 0, [])

    def gini(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probs = counts/y.shape[0]
        return 1 - np.sum(probs ** 2)
    
    def entropy(self, y):
        unique, counts = np.unique(y, return_counts=True)
        probs = counts/y.shape[0]
        return -np.sum(probs * np.log2(probs + 1e-9))

    def impurity(self, y):
        if self.criterion == 'gini':
            return self.gini(y)
        else:
            return self.entropy(y)

    def find_best_split(self, X, y, ignore_index):
        best_feature, best_split = None, None
        best_val = -float('inf')

        data_impurity = self.impurity(y)

        for feature in range(self.n_features):
            if feature in ignore_index:
                continue
            thresholds = np.unique(X[:, feature])
            if feature in self.index_categorical:
                if self.criterion == 'gini':
                    for category in thresholds:
                        left_idx = np.where(X[:, feature] == category)[0]
                        right_idx = np.where(X[:, feature] != category)[0]
                        gini_left = self.impurity(y[left_idx])
                        gini_right = self.impurity(y[right_idx])
                        weighted_gini = left_idx.shape[0]/y.shape[0] * gini_left + right_idx.shape[0]/y.shape[0] * gini_right
                        gini_gain = data_impurity - weighted_gini

                        if best_val < gini_gain:
                            best_val = gini_gain
                            best_feature = feature
                            best_split = category

                else:
                    conditional_weighted_entropy = 0
                    for category in thresholds:
                        index = np.where(X[:, feature] == category)[0]
                        impurity = self.impurity(y[index])
                        conditional_weighted_entropy += index.shape[0]/y.shape[0] * impurity

                    info_gain = data_impurity - conditional_weighted_entropy
                    if self.criterion == 'gain_ratio':
                        entropy_feature = self.impurity(X[:, feature])
                        gain_ratio = info_gain / entropy_feature
                    else:
                        gain_ratio = info_gain

                    if best_val < gain_ratio:
                        best_val = gain_ratio
                        best_feature = feature
                        best_split = thresholds

            else:

                thresholds = [(thresholds[i] + thresholds[i+1])/2 for i in range(0, len(thresholds) - 1)]
                for threshold in thresholds:
                    left_idx = np.where(X[:, feature] <= threshold)[0]
                    right_idx = np.where(X[:, feature] > threshold)[0]
                    left_impurity = self.impurity(y[left_idx])
                    right_impurity = self.impurity(y[right_idx])
                    weighted_val = left_idx.shape[0]/y.shape[0] * left_impurity + right_idx.shape[0]/y.shape[0] * right_impurity
                    val = data_impurity - weighted_val

                    if best_val < val:
                        best_val = val
                        best_feature = feature
                        best_split = threshold

        return best_feature, best_split

    def build_tree(self, X, y, depth, ignore_index):
        if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            unique, counts = np.unique(y, return_counts=True)
            return Node(value = unique[np.argmax(counts)])
        
        best_feature, best_split = self.find_best_split(X, y, ignore_index)

        node = Node(feature = best_feature, threshold = best_split)

        if best_feature in self.index_categorical:
            if self.criterion == 'gini':
                left_idx = np.where(X[:, best_feature] == best_split)[0]
                right_idx = np.where(X[:, best_feature] != best_split)[0]
                node.children["left"] = self.build_tree(X[left_idx], y[left_idx], depth + 1, ignore_index)
                node.children["right"] = self.build_tree(X[right_idx], y[right_idx], depth + 1, ignore_index)

            else:
                for category in np.unique(X[:, best_feature]):
                    index = np.where(X[:, best_feature] == category)[0]
                    node.children[category] = self.build_tree(X[index], y[index], depth + 1, ignore_index + [best_feature])

        else:
            left_idx = np.where(X[:, best_feature] <= best_split)[0]
            right_idx = np.where(X[:, best_feature] > best_split)[0]
            node.children["left"] = self.build_tree(X[left_idx], y[left_idx], depth + 1, ignore_index)
            node.children["right"] = self.build_tree(X[right_idx], y[right_idx], depth + 1, ignore_index)

        return node
    
    def predict(self, test_data):
        test_data = np.array(test_data)
        return np.array([self.traverse_tree(data, self.root) for data in test_data])
    
    def traverse_tree(self, data, node):
        if node.value is not None:
            return node.value
        
        feature_split = node.feature
        if feature_split in self.index_categorical:
            if self.criterion == 'gini':
                if data[feature_split] == node.split:
                    return self.traverse_tree(data, node.children['left'])
                else:
                    return self.traverse_tree(data, node.children['right'])
            else:
                return self.traverse_tree(data, node.children[data[feature_split]])
            
        else:
            if data[feature_split] <= node.split:
                return self.traverse_tree(data, node.children['left'])
            else:
                return self.traverse_tree(data, node.children['right'])
            
 