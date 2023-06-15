from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

def ConfusionMatrixPlot(y, yhat, config, save=False):
        labels = ["Não Cancelado", "Cancelado"]
        cm = confusion_matrix(y, yhat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")

        if save == True:
            plt.savefig(config.plots.confusion_matrix.path, dpi=120)