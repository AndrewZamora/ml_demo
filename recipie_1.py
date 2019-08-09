from sklearn import tree
# 0 = bumpy 1 = smooth
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
# 0 = apples 1 = oranges
labels = [0, 0, 1, 1]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)
print(clf.predict([[150,0]]))
# Example is from this video https://www.youtube.com/watch?v=cKxRvEZd3Mw