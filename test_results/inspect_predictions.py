import sys
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import pandas as pd

df = pd.read_csv('test_results/predictions_log.csv')
folders = ['No Impairment', 'Very Mild Impairment', 'Mild Impairment', 'Moderate Impairment']

for folder in folders:
    sub = df[df['original_folder'] == folder].head(4)
    print(f"\n==================== {folder} ({len(df[df['original_folder'] == folder])} Scans) ====================")
    for _, r in sub.iterrows():
        status = "✅" if r['is_correct'] else "❌"
        print(f"{status} {r['file_name']:<18} | Pred: {r['predicted_label']:<12} | Conf: {r['confidence']:>5.1f}% | Non: {r['prob_NonDemented']:>4.1f}% | VeryMild: {r['prob_VeryMild']:>4.1f}% | Mild: {r['prob_Mild']:>4.1f}% | Mod: {r['prob_Moderate']:>4.1f}%")
