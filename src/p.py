'''for epoch in range(epochs):
    predictions: torch.Tensor = self.predict_proba(features)
    loss: torch.Tensor = self.binary_cross_entropy_loss(predictions,labels)

    error: torch.Tensor = predictions - labels
    grad_weights: torch.Tensor =  torch.matmul(features.T,error) / features.shape[0]
    grad_bias: torch.Tensor = error.mean()

    self.weights[:-1] -= learning_rate*grad_weights
    self.weights[-1] -= learning_rate*grad_bias
return'''