import matplotlib.pyplot as plt
import pandas as pd
df_graph = pd.read_csv('eval_metrics_Random_Forest_20.tsv')
#fig, ax = plt.subplots()
y1 = df_graph['Precision'].to_numpy()
x = df_graph['Field'].to_numpy()
y2 = df_graph['Recall'].to_numpy()
y3 = df_graph['F1-Score'].to_numpy()
fig, ax = plt.subplots(figsize=(12, 6))
#plt.figure(figsize=(12, 6))
#df_graph.reset_index()
#print(df_graph)
#print(df)
ax.plot(x, y2, label='Recall', color='blue', marker='o', linestyle='-', linewidth=2)
ax.plot(x, y3, label='F1-Score', color='green', marker='s', linestyle='--', linewidth=2)
ax.plot(x, y1, label='Precision', color='red', marker='^', linestyle='-.', linewidth=2)

# Customize the plot
plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
ax.set_xlabel('Field')
ax.set_ylabel('Score')
ax.set_title('Metrics by Field')
ax.legend()

# Show the plot
plt.tight_layout()
plt.savefig('metrics_plot_Random_Forest_Classifier_20.png')  # Specify the desired file name and format
plt.show()
