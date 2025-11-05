from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

def build_and_fit_stacking(rf_model, lgbm_model, X_train, y_train, random_state):
    final = LogisticRegression(solver='saga', class_weight='balanced', random_state=random_state, max_iter=1000)
    stack = StackingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lgbm', lgbm_model)
        ],
        final_estimator=final,
        n_jobs=-1,
        passthrough=False,
        cv=5
    )
    stack.fit(X_train, y_train)
    return stack