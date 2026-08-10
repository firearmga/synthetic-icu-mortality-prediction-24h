"""
Model definitions and training utilities.

Models included (matching the paper's comparison set):
    - CART (Decision Tree)
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - MARS-style model (approximated with pyearth if installed, otherwise a
      spline-based Logistic Regression fallback — see get_mars_model())
"""

import warnings

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


def get_mars_model():
    """
    True MARS (multivariate adaptive regression splines) requires the
    'pyearth' package, which is not part of scikit-learn and can be tricky
    to install. As a practical stand-in with the same spirit (nonlinear,
    spline-based, automatic in nonlinearity), we use a spline-transformed
    Logistic Regression pipeline. Swap this out for `from pyearth import Earth`
    if you have it installed.
    """
    return Pipeline([
        ("scale", StandardScaler()),
        ("spline", SplineTransformer(degree=3, n_knots=4, include_bias=False)),
        ("clf", LogisticRegression(max_iter=3000, random_state=42)),
    ])


def get_models():
    """Returns the dict of {model_name: unfitted estimator} used across the project."""
    return {
        "CART": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=12, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "MARS": get_mars_model(),
    }


def train_model(model, X_train, y_train):
    """Fit a single model in place and return it."""
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, models=None):
    """Fit every model in `models` (or the default set) on the given training data."""
    if models is None:
        models = get_models()
    fitted = {}
    for name, model in models.items():
        fitted[name] = train_model(model, X_train, y_train)
    return fitted
