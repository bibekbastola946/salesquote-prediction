import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.utils import class_weight
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from tensorflow import keras
from tensorflow.keras import layers, callbacks, optimizers, metrics
import matplotlib.pyplot as plt
import seaborn as sns

def create_advanced_features(df):
    df = df.copy()
    df['CostToPriceRatio'] = df['TotalCost'] / (df['SalePrice'] + 1e-6)
    df['ActualPriceRatio'] = df['ActualSalePrice'] / (df['SalePrice'] + 1e-6)
    df['ProfitMargin'] = df['FinalProfit'] / (df['SalePrice'] + 1e-6)
    df['DiscountGiven'] = 1 - (df['ActualSalePrice'] / (df['SalePrice'] + 1e-6))

    if 'QuoteDate' in df.columns:
        df['QuoteDate'] = pd.to_datetime(df['QuoteDate'], errors='coerce')
        df['QuoteAgeDays'] = (pd.Timestamp.now() - df['QuoteDate']).dt.days
        df['QuoteMonth'] = df['QuoteDate'].dt.month
        df['QuoteQuarter'] = df['QuoteDate'].dt.quarter
        df['QuoteDayOfWeek'] = df['QuoteDate'].dt.dayofweek
        df['QuoteMonth_sin'] = np.sin(2 * np.pi * df['QuoteMonth'] / 12)
        df['QuoteMonth_cos'] = np.cos(2 * np.pi * df['QuoteMonth'] / 12)
        df['QuoteDayOfWeek_sin'] = np.sin(2 * np.pi * df['QuoteDayOfWeek'] / 7)
        df['QuoteDayOfWeek_cos'] = np.cos(2 * np.pi * df['QuoteDayOfWeek'] / 7)
        df['IsEndOfQuarter'] = df['QuoteMonth'].isin([3, 6, 9, 12]).astype(int)
    bins = [0, 400000, 1000000, 3000000, np.inf]
    labels = ['Small', 'Medium', 'Large', 'XLarge']
    df['DealSize'] = pd.cut(df['SalePrice'], bins=bins, labels=labels)

    df = df.drop(columns=['QuoteDate'], errors='ignore')
    return df

def build_enhanced_model(input_shape):
    inputs = keras.Input(shape=(input_shape,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)

    attention = layers.Dense(input_shape, activation='sigmoid')(x)
    x = layers.Multiply()([inputs, attention])

    x = layers.Dense(192, activation='relu', kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(32, activation='relu')(x)
    x = layers.BatchNormalization()(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    optimizer = optimizers.AdamW(learning_rate=0.0005, weight_decay=0.005)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[
        'accuracy', metrics.Precision(name='precision'), metrics.Recall(name='recall'),
        metrics.AUC(name='auc'), metrics.AUC(name='prc', curve='PR')
    ])
    return model


class HybridEnsemble:
    def __init__(self, n_models=3, n_folds=5, meta_model_type='lgbm'):
        self.n_models = n_models
        self.n_folds = n_folds
        self.nn_models = []
        self.xgb_models = []
        self.meta_model_type = meta_model_type
        self.meta_model = self._get_meta_model()
        self.meta_feature_names = []

    def _get_meta_model(self):
        if self.meta_model_type == 'lgbm':
            return LGBMClassifier(n_estimators=200, random_state=42, class_weight='balanced')
        elif self.meta_model_type == 'catboost':
            return CatBoostClassifier(
                iterations=500,
                learning_rate=0.05,
                depth=6,
                loss_function='Logloss',
                eval_metric='AUC',
                random_seed=42,
                verbose=0,
                early_stopping_rounds=20
            )
        elif self.meta_model_type == 'logistic':
            return LogisticRegression(max_iter=1000, class_weight='balanced')
        elif self.meta_model_type == 'rf':
            return RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')

    def fit(self, X, y):
        skf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((X.shape[0], self.n_models * 2 * self.n_folds))

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n[Fold {fold+1}/{self.n_folds}]")
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]

            smote = SMOTE(random_state=fold)
            X_train_res, y_train_res = smote.fit_resample(X_train_fold, y_train_fold)

            for i in range(self.n_models):
                print(f"Training NN {i+1} Fold {fold+1}")
                model = build_enhanced_model(X.shape[1])
                # Compute actual class weights
                cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train_res), y=y_train_res)
                class_weights = dict(enumerate(cw))
                model.fit(
                    X_train_res, y_train_res,
                    epochs=100, batch_size=64, validation_split=0.2, verbose=0,
                    class_weight=class_weights,
                    callbacks=[
                        callbacks.EarlyStopping(monitor='val_prc', patience=8, mode='max', restore_best_weights=True),
                        callbacks.ReduceLROnPlateau(monitor='val_prc', factor=0.5, patience=4, min_lr=1e-6, mode='max')
                    ]
                )

                self.nn_models.append(model)
                meta_features[val_idx, fold * self.n_models + i] = model.predict(X_val_fold).flatten()

            for i in range(self.n_models):
                print(f"Training XGB {i+1} Fold {fold+1}")
                xgb = XGBClassifier(
                    n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, random_state=fold * 100 + i,
                    scale_pos_weight=np.sum(y_train_fold == 0) / np.sum(y_train_fold == 1)
                )
                xgb.fit(X_train_fold, y_train_fold)
                self.xgb_models.append(xgb)
                meta_features[val_idx, self.n_models * self.n_folds + (fold * self.n_models + i)] = xgb.predict_proba(X_val_fold)[:, 1]

        self.meta_feature_names = [f'm{i}' for i in range(meta_features.shape[1])]
        print("\nTraining Meta Model...")
        self.meta_model.fit(pd.DataFrame(meta_features, columns=self.meta_feature_names), y)

    def predict_proba(self, X):
        X_meta = self._create_meta_features(X)
        return self.meta_model.predict_proba(pd.DataFrame(X_meta, columns=self.meta_feature_names))

    def _create_meta_features(self, X):
        nn_preds = np.array([model.predict(X).flatten() for model in self.nn_models]).T
        xgb_preds = np.array([model.predict_proba(X)[:, 1] for model in self.xgb_models]).T
        return np.hstack([nn_preds, xgb_preds])
    
def plot_and_save_confusion_matrix(y_true, y_pred, class_names, path):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "confusion_matrix.png"))
    plt.show()


def plot_and_save_roc_curve(y_true, y_proba, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(path, "roc_curve.png"))
    plt.show()


def plot_and_save_precision_recall_curve(y_true, y_proba, path):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure()
    plt.plot(recall, precision, label="PR Curve")
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(path, "precision_recall_curve.png"))
    plt.show()

def main():
    df = pd.read_csv('sales_quote_data.csv')
    df = create_advanced_features(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['InternalResult'])

    X = df.drop(columns=['InternalResult'])
    X = pd.get_dummies(X, columns=['EquipmentCategory', 'DealSize'], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs('model_artifacts_hybrid', exist_ok=True)
    joblib.dump(X_train.columns.tolist(), 'model_artifacts_hybrid/X_train_columns.pkl')
    joblib.dump(scaler, 'model_artifacts_hybrid/scaler.pkl')
    joblib.dump(label_encoder, 'model_artifacts_hybrid/label_encoder.pkl')

    ensemble = HybridEnsemble(n_models=3, n_folds=5, meta_model_type='lgbm')
    #ensemble = HybridEnsemble(n_models=2, n_folds=3, meta_model_type='lgbm')
    ensemble.fit(X_train_scaled, y_train)

    y_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

    best_threshold = 0.5
    best_macro_f1 = 0
    thresholds = np.arange(0.3, 0.71, 0.01)
    for threshold in thresholds:
        temp_pred = (y_proba >= threshold).astype(int)
        report = classification_report(y_test, temp_pred, target_names=label_encoder.classes_, output_dict=True)
        macro_f1 = (report['Lost']['f1-score'] + report['Won']['f1-score']) / 2
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_threshold = threshold

    print(f"\nBest threshold for Macro F1-score: {best_threshold:.2f} (Macro F1: {best_macro_f1:.4f})")

    y_pred = (y_proba >= best_threshold).astype(int)
    print("\nHybrid Ensemble Final Evaluation:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Save evaluation outputs
    joblib.dump(y_test, 'model_artifacts_hybrid/y_test.pkl')
    joblib.dump(y_pred, 'model_artifacts_hybrid/y_pred.pkl')
    joblib.dump(y_proba, 'model_artifacts_hybrid/y_proba.pkl')

    feature_importances = ensemble.meta_model.feature_importances_
    meta_feature_names = ensemble.meta_feature_names
    importance_df = pd.DataFrame({
        'Feature': meta_feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Plot Feature Importances
    plt.figure(figsize=(12, 6))
    plt.bar(importance_df['Feature'], importance_df['Importance'])
    plt.xticks(rotation=90)
    plt.title('Meta Model Feature Importance (Base Model Strength)')
    plt.xlabel('Base Model (NN/XGB Fold Model)')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.savefig('model_artifacts_hybrid/feature_importance.png')
    plt.show()

    print("\nTop 10 Most Important Base Models:")
    print(importance_df.head(10))

    # Confusion Matrix, ROC, PR curves
    class_names = ['Lost', 'Won']
    plot_and_save_confusion_matrix(y_test, y_pred, class_names, 'model_artifacts_hybrid')
    plot_and_save_roc_curve(y_test, y_proba, 'model_artifacts_hybrid')
    plot_and_save_precision_recall_curve(y_test, y_proba, 'model_artifacts_hybrid')

if __name__ == "__main__":
    main()
