# statistics and machine learning
from sklearn.metrics import confusion_matrix
# visualizations
import seaborn as sns

def plot_confmat(y_true, y_pred, ax, context_info):
    confmat = confusion_matrix(y_true, y_pred)
    sns.set()
    ax.set_title(''.join(
        f"{key}: {value}.\n"
        for key, value in context_info.items()
    ))
    ax.set_xlabel("true label")
    ax.set_ylabel("predicted label")
    sns.heatmap(
        confmat,
        ax=ax,
        annot=True,
        xticklabels=context_info["cluster_labels"],
        yticklabels=context_info["cluster_labels"],
    )