import sys, pandas as pd
sys.path.append('src')
from huggingface_hub import HfApi
from datasets import load_dataset
from preprocessing import clean_darija_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

api = HfApi()
candidates = [d.id for d in api.list_datasets(search='darija sentiment', limit=10)]
print("Datasets trouves :", candidates)

df = None
for ds_name in candidates:
    try:
        ds = load_dataset(ds_name)
        split = 'train' if 'train' in ds else list(ds.keys())[0]
        temp_df = pd.DataFrame(ds[split])
        if len(temp_df) > 500 and len(temp_df.columns) >= 2:
            df = temp_df
            print("Dataset charge avec succes :", ds_name, "Lignes :", len(df))
            break
    except Exception:
        continue

df = df.dropna(subset=[df.columns[0], df.columns[1]])
df['clean_text'] = df[df.columns[0]].astype(str).apply(clean_darija_text)
df = df[df['clean_text'].str.len() > 2]
df.to_csv("data/processed/darija_clean.csv", index=False)

X_tr, X_te, y_tr, y_te = train_test_split(df['clean_text'], df[df.columns[1]].astype(str), test_size=0.2, random_state=42)
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
X_tr_v = vec.fit_transform(X_tr)
X_te_v = vec.transform(X_te)

results = []
for name, clf in [("Baseline_LogReg", LogisticRegression(max_iter=1000)), ("Baseline_LinearSVM", LinearSVC())]:
    preds = clf.fit(X_tr_v, y_tr).predict(X_te_v)
    results.append({"Model": name, "Accuracy": round(accuracy_score(y_te, preds), 4), "F1_Macro": round(f1_score(y_te, preds, average="macro"), 4)})

res_df = pd.DataFrame(results)
res_df.to_csv("results/benchmark_table.csv", index=False)
print("\n--- RESULTATS OBTENUS ---")
print(res_df)
