import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from contextlib import contextmanager
from sklearn.model_selection import GridSearchCV, ParameterGrid, StratifiedKFold
from contextlib import contextmanager

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def tune_pipeline(vectorizer, model, param_grid, X_train, y_train, model_name,
                  output_dir="outputs", refit_metric="f1_macro", use_parallel=True):
        
    os.makedirs(output_dir, exist_ok=True)

    pipe = Pipeline([
        ("vec", vectorizer),
        ("model", model)
    ])

    scoring = {
        "accuracy": "accuracy",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    n_jobs_value = -1 if use_parallel else 1

    total_candidates = len(list(ParameterGrid(param_grid)))
    n_splits = cv.get_n_splits(X_train, y_train)
    total_fits = total_candidates * n_splits
    print(f"GridSearch: {total_candidates} combinations, {n_splits} folds => {total_fits} fits")
    total_candidates = len(list(ParameterGrid(param_grid)))
    n_splits = cv.get_n_splits(X_train, y_train)
    total_fits = total_candidates * n_splits
    print(f"GridSearch: {total_candidates} combinations, {n_splits} folds => {total_fits} fits")

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        n_jobs=n_jobs_value,
        verbose=0
    )

    with tqdm_joblib(tqdm(total=total_fits, desc=f"{model_name} GridSearch", unit="fit")):
        search.fit(X_train, y_train)

    # 1. Export the best model
    model_filename = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_best.joblib")
    joblib.dump(search.best_estimator_, model_filename)
    print(f"--- Best model saved to: {model_filename}")

    # 2. Save Tuning Report
    results_df = pd.DataFrame(search.cv_results_)
    rank_column = f"rank_test_{refit_metric}"
    
    results_df = results_df.sort_values(by=rank_column)
    
    report_filename = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_tuning_report.csv")
    results_df.to_csv(report_filename, index=False)
    print(f"--- Performance report saved to: {report_filename}")

    return search

# source : https://neuralib.readthedocs.io/en/v0.4.1/_modules/neuralib/util/tqdm.html

@contextmanager
def tqdm_joblib(tqdm_bar):
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_bar.close()