def main():
    # Input confusion matrix values
    a = float(input("Enter a (True 0, Pred 0): "))
    b = float(input("Enter b (True 0, Pred 1): "))
    c = float(input("Enter c (True 1, Pred 0): "))
    d = float(input("Enter d (True 1, Pred 1): "))

    # Class 1 (positive)
    precision_1 = d / (d + b) if (d + b) != 0 else 0
    recall_1 = d / (d + c) if (d + c) != 0 else 0
    f1_1 = (2 * d) / (2 * d + b + c) if (2 * d + b + c) != 0 else 0

    # Class 0
    precision_0 = a / (a + c) if (a + c) != 0 else 0
    recall_0 = a / (a + b) if (a + b) != 0 else 0
    f1_0 = (2 * a) / (2 * a + b + c) if (2 * a + b + c) != 0 else 0

    # Macro averages
    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    # Print results
    print("\n--- Per-class Metrics ---")
    print(f"Class 0 -> Precision: {precision_0:.4f}, Recall: {recall_0:.4f}, F1: {f1_0:.4f}")
    print(f"Class 1 -> Precision: {precision_1:.4f}, Recall: {recall_1:.4f}, F1: {f1_1:.4f}")

    print("\n--- Macro-Averaged Metrics ---")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1-score:  {macro_f1:.4f}")


if __name__ == "__main__":
    main()
