import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

def evaluate_and_save(model_name, y_true, y_pred, labels=None, save_plot=True):
    acc = round(accuracy_score(y_true, y_pred), 4)
    f1_macro = round(f1_score(y_true, y_pred, average='macro'), 4)
    f1_weighted = round(f1_score(y_true, y_pred, average='weighted'), 4)
    prec = round(precision_score(y_true, y_pred, average='macro', zero_division=0), 4)
    rec = round(recall_score(y_true, y_pred, average='macro', zero_division=0), 4)

    result = {
        "Model": model_name,
        "Accuracy": acc,
        "F1_Macro": f1_macro,
        "F1_Weighted": f1_weighted,
        "Precision": prec,
        "Recall": rec
    }

    csv_path = "../results/benchmark_table.csv"
    try:
        df_res = pd.read_csv(csv_path)
        df_res = df_res[df_res["Model"] != model_name]
        df_res = pd.concat([df_res, pd.DataFrame([result])], ignore_index=True)
    except Exception:
        df_res = pd.DataFrame([result])

    df_res.to_csv(csv_path, index=False)
    print(f"Résultats enregistrés pour {model_name} :", result)

    if save_plot:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels if labels else "auto",
                    yticklabels=labels if labels else "auto")
        plt.title(f"Matrice de Confusion - {model_name}")
        plt.xlabel("Prédiction")
        plt.ylabel("Réalité")
        plt.savefig(f"../results/plots/cm_{model_name.replace(' ', '_')}.png", bbox_inches='tight')
        plt.close()

    return result