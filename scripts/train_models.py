import ember
from sklearn import metrics
import joblib
import numpy as np


def get_xy():
    # Reading in the data and vectorising it
    x_train, y_train = ember.read_vectorized_features("files", "train", feature_version=2)
    x_test, y_test = ember.read_vectorized_features("files", "test", feature_version=2)

    return x_train, y_train, x_test, y_test


def save_model(model):
    # Save a ML model to a file
    joblib.dump(model, "model.sav")


def apply_model(directory, filename):
    # Import a ML model and classify a file
    imported_model = joblib.load(directory)
    file_data = open(filename, "rb").read()
    extractor = ember.PEFeatureExtractor(2)
    features = np.array(extractor.feature_vector(file_data), dtype=np.float32)

    # Use the loaded model to make predictions
    predicted_class = imported_model.predict([features])[0]
    predicted_proba = imported_model.predict_proba([features])[0]

    # Output predictions
    print("Classification: " + str(predicted_class))
    if predicted_proba[0] not in [0.0, 1.0]:
        print("Benign Probability: " + str(predicted_proba[0]))
        print("Malware Probability: " + str(predicted_proba[1]))


def ml_dt(x_train, y_train, x_test, y_test):
    # Decision Tree
    from sklearn.tree import DecisionTreeClassifier

    # Training model
    dt = DecisionTreeClassifier()
    dt.fit(x_train, y_train)
    predicted = dt.predict(x_test)

    # Output results
    results = [dt, dt.score(x_test, y_test), metrics.confusion_matrix(y_test, predicted)]

    return results


def ml_lr(x_train, y_train, x_test, y_test):
    # Logistic Regression
    from sklearn.linear_model import LogisticRegression

    # Training model
    lr = LogisticRegression(max_iter=40000)
    lr.fit(x_train, y_train)
    predicted = lr.predict(x_test)

    # Output results
    results = [lr, lr.score(x_test, y_test), metrics.confusion_matrix(y_test, predicted)]

    return results


def ml_knn(x_train, y_train, x_test, y_test):
    # K-Nearest Neighbors
    from sklearn.neighbors import KNeighborsClassifier

    # Training model
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    predicted = knn.predict(x_test)

    # Output results
    results = [knn, knn.score(x_test, y_test), metrics.confusion_matrix(y_test, predicted)]

    return results


def ml_rf(x_train, y_train, x_test, y_test):
    # Random Forest
    from sklearn.ensemble import RandomForestClassifier

    # Training model
    rf = RandomForestClassifier(n_estimators=30, max_depth=11)
    rf.fit(x_train, y_train)
    predicted = rf.predict(x_test)

    # Output results
    results = [rf, rf.score(x_test, y_test), metrics.confusion_matrix(y_test, predicted)]

    return results


def ml_svm(x_train, y_train, x_test, y_test):
    # Support Vector Machine
    from sklearn import svm

    # Training model
    svm_model = svm.SVC(probability=True)
    svm_model.fit(x_train, y_train)
    predicted = svm_model.predict(x_test)

    # Output results
    results = [svm_model, svm_model.score(x_test, y_test), metrics.confusion_matrix(y_test, predicted)]

    return results


def ml_lda(x_train, y_train, x_test, y_test):
    # Latent Dirichlet Allocation
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    # Training model
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    predicted = lda.predict(x_test)

    # Output results
    results = [lda, lda.score(x_test, y_test), metrics.confusion_matrix(y_test, predicted)]

    return results


def ml_lgbm(x_train, y_train, x_test, y_test):
    # LightGBM
    import lightgbm as lgb

    # Training model
    lgbm = lgb.LGBMClassifier()
    lgbm.fit(x_train, y_train)
    predicted = lgbm.predict(x_test)

    # Output results
    results = [lgbm, lgbm.score(x_test, y_test), metrics.confusion_matrix(y_test, predicted)]

    return results


def example_script():
    # Example use case
    x_train, y_train, x_test, y_test = get_xy()
    save_model(ml_lgbm(x_train, y_train, x_test, y_test)[0])
    apply_model("model.sav", "unknownfile.exe")


example_script()





