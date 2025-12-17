import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Sklearn classifiers
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve

# Gradient Boosting libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Local preprocessing
from preprocessing import load_and_preprocess_data

# Set style
sns.set(style='whitegrid')

# Define models by category (NO linear/regression models)
MODELS = {
    'Tree-Based Models': {
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Extra Tree': ExtraTreeClassifier(random_state=42),
    },
    'Ensemble Models': {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, verbosity=0),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
        'CatBoost': cb.CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
    },
    'Distance & Kernel Models': {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'SVM (RBF Kernel)': SVC(kernel='rbf', probability=True, random_state=42),
        'SVM (Linear Kernel)': SVC(kernel='linear', probability=True, random_state=42),
    },
    'Probabilistic & Neural Models': {
        'Gaussian Naive Bayes': GaussianNB(),
        'MLP Classifier': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    },
}


def train_and_evaluate(models_dict, X_train, X_test, y_train, y_test):
    """Train all models and return results DataFrame."""
    results = []
    
    os.makedirs('results/plots', exist_ok=True)
    
    for category, models in models_dict.items():
        print(f"\n=== {category} ===")
        for name, model in models.items():
            print(f"  Training {name}...")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
            
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            results.append({
                'Category': category,
                'Model': name,
                'Accuracy': round(acc, 4),
                'F1-Score': round(f1, 4),
                'ROC-AUC': round(auc, 4)
            })
    
    return pd.DataFrame(results)


def plot_roc_curves(models_dict, X_train, X_test, y_train, y_test):
    """Generate ROC curve plot for all models."""
    plt.figure(figsize=(12, 10))
    
    for category, models in models_dict.items():
        for name, model in models.items():
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                y_prob = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - All Classification Models')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig('results/plots/roc_curves_all.png', dpi=150)
    plt.close()


def plot_comparison_by_category(results_df):
    """Generate bar charts grouped by category."""
    categories = results_df['Category'].unique()
    
    # Overall comparison
    plt.figure(figsize=(14, 8))
    melted = results_df.melt(id_vars=['Category', 'Model'], var_name='Metric', value_name='Score')
    sns.barplot(x='Model', y='Score', hue='Metric', data=melted, palette='viridis')
    plt.title('Classification Model Performance Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('results/plots/model_comparison_all.png', dpi=150)
    plt.close()
    
    # Per-category comparison
    for cat in categories:
        cat_df = results_df[results_df['Category'] == cat]
        plt.figure(figsize=(10, 6))
        melted = cat_df.melt(id_vars=['Category', 'Model'], var_name='Metric', value_name='Score')
        sns.barplot(x='Model', y='Score', hue='Metric', data=melted, palette='viridis')
        plt.title(f'{cat} - Performance Comparison')
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        filename = cat.replace(' ', '_').replace('&', 'and').lower()
        plt.savefig(f'results/plots/{filename}.png', dpi=150)
        plt.close()


if __name__ == "__main__":
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data('data/telco_churn.csv')
    
    # Train and evaluate
    results_df = train_and_evaluate(MODELS, X_train, X_test, y_train, y_test)
    
    # Plots
    print("\nGenerating plots...")
    plot_roc_curves(MODELS, X_train, X_test, y_train, y_test)
    plot_comparison_by_category(results_df)
    
    # Save results
    results_df = results_df.sort_values(by='ROC-AUC', ascending=False)
    results_df.to_csv('results/benchmark_results.csv', index=False)
    
    print("\n" + "="*60)
    print("BENCHMARK RESULTS (Sorted by ROC-AUC)")
    print("="*60)
    print(results_df.to_string(index=False))
    print("\nPlots saved to results/plots/")
    print("Results saved to results/benchmark_results.csv")
