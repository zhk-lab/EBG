def measure(rows, threshold):
    predictions = [int(row['score'] >= threshold) for row in rows]
    correct = sum(prediction == row['label'] for prediction, row in zip(predictions, rows))
    positives = sum(row['label'] for row in rows)
    hits = sum(prediction == 1 and row['label'] == 1 for prediction, row in zip(predictions, rows))
    return {'accuracy': correct / len(rows), 'recall': hits / positives,
            'correct': correct, 'total': len(rows), 'true_positives': hits, 'positives': positives}
