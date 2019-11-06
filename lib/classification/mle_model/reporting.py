# statistics and machine learning
from sklearn.metrics import confusion_matrix
# visualizations
import seaborn as sns

def plot_confmat(y_true, y_pred, cluster_labels, ax, context_info):
    confmat = confusion_matrix(y_true, y_pred)
    sns.set()
    ax.set_title(''.join(
        f"{key}: {value}.\n"
        for key, value in context_info.items()
    ), weight="bold")
    sns.heatmap(
        confmat,
        ax=ax,
        annot=True,
        xticklabels=cluster_labels,
        yticklabels=cluster_labels,
    )