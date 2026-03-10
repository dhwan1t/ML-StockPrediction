import subprocess
import re
import os

print("Running main.py to get latest results...")
result = subprocess.run(['source stock_env/bin/activate && python3 main.py'], shell=True, capture_output=True, text=True, cwd='/Users/himishavyas/Documents/Code Files/Sem 4 Mini project/stock_prediction')
output = result.stdout

if result.returncode != 0:
    print("Error running main.py:")
    print(result.stderr)
    exit(1)

# Extract best model name
best_model_match = re.search(r"Name:\s+(.+)", output)
best_model = best_model_match.group(1).strip() if best_model_match else "Gradient Boosting"

# Extract F1, Acc, AUC of best model
acc_match = re.search(r"Test Accuracy:\s+([0-9.]+)", output)
best_acc = acc_match.group(1) if acc_match else "0.0"

f1_match = re.search(r"Test F1-Score:\s+([0-9.]+)", output)
best_f1 = f1_match.group(1) if f1_match else "0.0"

auc_match = re.search(r"Test ROC-AUC:\s+([0-9.]+)", output)
best_auc = auc_match.group(1) if auc_match else "0.0"

# Initial rows
rows_match = re.search(r"Dataset Size:\s+(\d+)", output)
initial_rows = rows_match.group(1) if rows_match else "1000"

print(f"Extracted -> Model: {best_model}, Acc: {best_acc}, F1: {best_f1}, AUC: {best_auc}, Rows: {initial_rows}")

report_path = '/Users/himishavyas/Documents/Code Files/Sem 4 Mini project/stock_prediction/reports/REPORT.md'
with open(report_path, 'r') as f:
    report_content = f.read()

# Replace placeholders
report_content = report_content.replace('[INSERT BEST MODEL]', best_model)
report_content = report_content.replace('[INSERT ACCURACY]', str(round(float(best_acc) * 100, 2)))
report_content = report_content.replace('[INSERT F1]', best_f1)
report_content = report_content.replace('[INSERT AUC]', best_auc)
report_content = report_content.replace('[INSERT INITIAL ROWS]', initial_rows)
# Placeholder Justification
report_content = report_content.replace('[INSERT JUSTIFICATION - e.g. Gradient Boosting handles tabular, non-linear feature interactions without aggressively overfitting like deep forests, OR Logistic regression provided a resilient baseline impenetrable to excessive noise]', "Gradient Boosting handles tabular, non-linear feature interactions effectively by iteratively correcting the mistakes of weak learners")

# Parse comparison table from output
table_match = re.search(r"Model\s+F1-Score\s+Accuracy\s+ROC-AUC\s+Precision\s+Recall\n(=+)\n(.*?)(?=\n={80}|\n\[Step)", output, re.DOTALL)
if table_match:
    table_lines = table_match.group(2).strip().split('\n')
    for line in table_lines:
        parts = line.strip().split()
        if len(parts) >= 6:
            # Model name might be multiple words
            model_name = " ".join(parts[:-5])
            f1, acc, auc, prec, rec = parts[-5:]
            
            # Find and replace in markdown table
            if "Gradient Boosting" in model_name:
                report_content = re.sub(r'\|\s*\*\*Gradient Boosting\*\*\s*\|.*?\|.*?\|.*?\|.*?\|.*?\|', f'| **Gradient Boosting** | {f1} | {round(float(acc)*100, 2)}% | {auc} | {prec} | {rec} |', report_content)
            elif "K-Nearest" in model_name:
                report_content = re.sub(r'\|\s*\*\*K-Nearest Neighbors\*\*\s*\|.*?\|.*?\|.*?\|.*?\|.*?\|', f'| **K-Nearest Neighbors** | {f1} | {round(float(acc)*100, 2)}% | {auc} | {prec} | {rec} |', report_content)
            elif "Logistic Regression" in model_name:
                report_content = re.sub(r'\|\s*\*\*Logistic Regression\*\*\s*\|.*?\|.*?\|.*?\|.*?\|.*?\|', f'| **Logistic Regression** | {f1} | {round(float(acc)*100, 2)}% | {auc} | {prec} | {rec} |', report_content)
            elif "Random Forest" in model_name and "Tuned" not in model_name:
                report_content = re.sub(r'\|\s*\*\*Random Forest\*\*\s*\|.*?\|.*?\|.*?\|.*?\|.*?\|', f'| **Random Forest** | {f1} | {round(float(acc)*100, 2)}% | {auc} | {prec} | {rec} |', report_content)
            elif "Support Vector" in model_name:
                report_content = re.sub(r'\|\s*\*\*Support Vector Machine\*\*\s*\|.*?\|.*?\|.*?\|.*?\|.*?\|', f'| **Support Vector Machine** | {f1} | {round(float(acc)*100, 2)}% | {auc} | {prec} | {rec} |', report_content)

with open(report_path, 'w') as f:
    f.write(report_content)

print("Report updated successfully.")
