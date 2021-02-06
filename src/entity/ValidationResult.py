from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


class ValidationResult:
    """
    This class serves as a place to standardize the saved validation measurements
    """

    def __init__(self, labels, pred):
        self.tn, self.fp, self.fn, self.tp = confusion_matrix(labels, pred).ravel()
        self.accuracy = accuracy_score(labels, pred)
        self.precision = precision_score(labels, pred)
        self.recall = recall_score(labels, pred)
        self.f1 = f1_score(labels, pred)

    def toDict(self):
        return {
            "tn": self.tn.item(),
            "fp": self.fp.item(),
            "fn": self.fn.item(),
            "tp": self.tp.item(),
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1
        }
