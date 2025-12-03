export type ModelResult = {
  confusion_matrix: number[][];
  precision_benign: number;
  recall_benign: number;
  f1_benign: number;
  precision_malignant: number;
  recall_malignant: number;
  f1_malignant: number;
  accuracy: number;
} | null;