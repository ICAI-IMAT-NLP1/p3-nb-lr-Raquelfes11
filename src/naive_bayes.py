import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.size(1) # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features,labels,delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        # TODO: Count number of samples for each output class and divide by total of samples
        unique_labels, counts = torch.unique(labels, return_counts=True)
        total_samples:int = labels.size(0)
        class_priors: Dict[int, torch.Tensor] = {int(label): count / total_samples for label, count in zip(unique_labels, counts)}

        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples. (Para ada dosumento, un alista de BoW: ej [[0,1,2],[2,3,0]] la palabra 0 aparece 0 veces en el 1º la segunda word aparece 1 en el 1º...)
            labels (torch.Tensor): Labels corresponding to each training example. (labels de cada docuemnto de arriba)
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        # TODO: Estimate conditional probabilities for the words in features and apply smoothing
        class_word_counts: Dict[int, torch.Tensor] = dict()
        unique_labels: torch.Tensor = torch.unique(labels)

        for label in unique_labels:
            class_mask: torch.Tensor = (labels == label)
            class_features: torch.Tensor = features[class_mask] # documents of class c

            word_counts_in_class:int = class_features.sum(dim=0) # suma de palabras por clases
            total_words_in_class: int = word_counts_in_class.sum() # num total de words in clase c

            probs: torch.Tensor = (word_counts_in_class + delta)/(total_words_in_class + delta*self.vocab_size)
            class_word_counts[label.item()] = probs

        return class_word_counts


    def estimate_class_posteriors(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate the class posteriors for a given feature using the Naive Bayes logic.

        Args:
            feature (torch.Tensor): The bag of words vector for a single example.

        Returns:
            torch.Tensor: Log posterior probabilities for each class.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError(
                "Model must be trained before estimating class posteriors."
            )
        # TODO: Calculate posterior based on priors and conditional probabilities of the words
        log_priors: torch.Tensor = torch.log(torch.tensor([self.class_priors[label] for label in self.class_priors]))
        cond_probs: torch.Tensor = torch.stack([self.conditional_probabilities[label] for label in self.class_priors])

        log_likelihoods = torch.matmul(feature, torch.log(cond_probs).T)
        log_posteriors: torch.Tensor = log_priors + log_likelihoods

        return log_posteriors
    
    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        # TODO: Calculate log posteriors and obtain the class of maximum likelihood 
        log_posteriors : torch.Tensor = self.estimate_class_posteriors(feature)
        pred: int = torch.argmax(log_posteriors).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        # TODO: Calculate log posteriors and transform them to probabilities (softmax)
        log_posteriors: torch.Tensor = self.estimate_class_posteriors(feature)
        exps: torch.Tensor = torch.exp(log_posteriors) 
        sum_exps: torch.Tensor = torch.sum(exps)  
        probs: torch.Tensor = exps / sum_exps
        return probs
