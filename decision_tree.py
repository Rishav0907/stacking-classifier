import numpy as np
from config import DECISION_TREE_CONFIG


class DecisionTree:
    def __init__(self) -> None:
        self.max_depth = DECISION_TREE_CONFIG['max_depth']
        self.min_samples_for_split = DECISION_TREE_CONFIG['min_samples_for_split']

    def gini_impurity(self, y):
        # print(np.unique(y, return_counts=True))
        classes, classes_count = np.unique(y, return_counts=True)
        class_probabilities = classes_count/np.sum(classes_count)
        # print(class_probabilities)
        gini_impurity = 1 - np.sum(class_probabilities**2)
        return gini_impurity

    def information_gain(self, y, left_split, right_split):
        # Calculate the Gini impurity for the parent node
        parent_gini_impurity = self.gini_impurity(y)

        # Calculate the Gini impurity for the left and right splits
        left_gini_impurity = self.gini_impurity(left_split)
        right_gini_impurity = self.gini_impurity(right_split)

        # Calculate the weighted average of the Gini impurity after the split
        left_weight = len(left_split) / len(y)
        right_weight = len(right_split) / len(y)
        weighted_gini = (left_weight * left_gini_impurity +
                         right_weight * right_gini_impurity)

        # Information Gain is the reduction in impurity
        return parent_gini_impurity - weighted_gini

    def find_feature_with_max_gini_impurity(self, X, y, feature_names):
        num_features = X.shape[1]
        max_information_gain = -float('inf')
        best_feature = None
        best_threshold = None
        best_split = None
        for feature in range(num_features):
            # print(feature_names[feature])
            current_feature_values = X.loc[:, feature_names[feature]]
            # print(current_feature_values)
            current_feature_unique_values = np.unique(current_feature_values)
            threshold_values = current_feature_unique_values
            # print(threshold_values)
            count = 0
            for threshold in threshold_values:
                left_split = np.where(
                    X.loc[:, feature_names[feature]] < threshold)[0]
                # print(left_split[0])
                right_split = np.where(
                    X.loc[:, feature_names[feature]] >= threshold)[0]

                if len(left_split) > 0 and len(right_split) > 0:
                    left_split_gini_impurity = self.gini_impurity(
                        y.iloc[left_split])
                    right_split_gini_impurity = self.gini_impurity(
                        y.iloc[right_split])

                    information_gain = self.information_gain(
                        y, left_split=y.iloc[left_split], right_split=y.iloc[right_split])
                    if information_gain > max_information_gain:
                        max_information_gain = information_gain
                        best_feature = feature_names[feature]
                        best_threshold = threshold
                        best_split = (left_split, right_split)

            return best_feature, best_threshold, best_split

    def build_decision_tree(self, X, y, current_tree_depth, feature_columns):
        # print("working")
        # print(min(X['age']))
        num_samples, num_features = X.shape
        best_feature, best_threshold, best_split = self.find_feature_with_max_gini_impurity(
            X, y, feature_names=feature_columns)
        # print(best_feature, best_threshold, best_split)

        if best_feature is not None:
            left_subtree = self.build_decision_tree(
                X.iloc[best_split[0]], y.iloc[best_split[0]], current_tree_depth+1, feature_columns)
            right_subtree = self.build_decision_tree(
                X.iloc[best_split[1]], y.iloc[best_split[1]], current_tree_depth+1, feature_columns)
