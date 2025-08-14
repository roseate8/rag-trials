"""
Embedding generation using HuggingFace models
"""
import logging
from typing import List
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """Initialize the embedding model"""
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Check if CUDA is available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        logger.info(f"Loaded embedding model: {model_name} on {self.device}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 4) -> List[List[float]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
        
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embeddings_batch = outputs.last_hidden_state.mean(dim=1)
                embeddings_batch = torch.nn.functional.normalize(embeddings_batch, p=2, dim=1)
                embeddings.extend(embeddings_batch.cpu().numpy().tolist())
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings
