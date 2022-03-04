from feature_engine.encoding import OrdinalEncoder
from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline as imbPipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from gbm_model.config.core import config

# from gbm_model.processing import features as pp

model_pipe = imbPipeline(
    [
        # categorical encoding
        (
            "cat_encoding",
            OrdinalEncoder(
                encoding_method="arbitrary",
                variables=config.model_config.categorical_vars,
            ),
        ),
        # feature scaling
        ("scale", StandardScaler()),
        # oversampling with ADASYN
        (
            "oversampling",
            ADASYN(
                sampling_strategy="auto",
                random_state=config.model_config.random_state,
                n_neighbors=config.model_config.n_neighbors,
            ),
        ),
        # classifier
        (
            "gbm",
            GradientBoostingClassifier(
                n_estimators=config.model_config.n_estimators,
                criterion=config.model_config.criterion,
                max_depth=config.model_config.max_depth,
                min_samples_split=config.model_config.min_samples_split,
                learning_rate=config.model_config.learning_rate,
            ),
        ),
    ]
)
