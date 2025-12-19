One idea of a loss function

def barcode_loss(predictions, targets):    
    total_loss = 0
    num_digits = predictions.shape[1]
    
    for i in range(num_digits):
        # Calculate loss for the i-th digit position
        digit_logits = predictions[:, i, :]
        digit_targets = targets[:, i]
        total_loss += F.cross_entropy(digit_logits, digit_targets)
        
    return total_loss / num_digits