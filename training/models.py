import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt


def scaled_dot_product_attention(query, key, value):
    """
    Compute scaled dot product attention.

    Args:
        query (torch.Tensor): Query tensor of shape (batch_size, seq_len_q, dim_q).
        key (torch.Tensor): Key tensor of shape (batch_size, seq_len_k, dim_k).
        value (torch.Tensor): Value tensor of shape (batch_size, seq_len_v, dim_v).

    Returns:
        torch.Tensor: Output tensor after applying scaled dot product attention of shape (batch_size, seq_len_q, dim_v).

    """
    dim_k = key.size(-1)
    scores = torch.bmm(query, key.transpose(1, 2)) / sqrt(dim_k)
    weights = F.softmax(scores, dim=-1)
    att_mat = weights.detach().cpu().numpy()
    return torch.bmm(weights, value), att_mat
    


class AttentionHead(nn.Module):
    """
    Attention head module for the Transformer model. Encapsulates the operations required to compute attention
    within a single attention head of the Transformer model.

    Args:
        embed_dim (int): Dimensionality of the input embeddings.
        head_dim (int): Dimensionality of the attention head.

    """

    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q = nn.Linear(embed_dim, head_dim)
        self.k = nn.Linear(embed_dim, head_dim)
        self.v = nn.Linear(embed_dim, head_dim)
        self.att_mat = None

    def forward(self, hidden_state):
        """
        Perform forward pass through the attention head.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor after applying scaled dot product attention of shape (batch_size, seq_len, head_dim).

        """
        attn_outputs, attn_mat = scaled_dot_product_attention(
            self.q(hidden_state), self.k(hidden_state), self.v(hidden_state)
        )
        self.att_mat = attn_mat
        return attn_outputs


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for the Transformer model. Combines the outputs of multiple attention heads
    and applies a linear transformation to produce the final output of the attention mechanism in the Transformer model.

    Args:
        config (object): Configuration object containing model parameters.

    """

    def __init__(self, config):
        super().__init__()
        embed_dim = config.hidden_size
        num_heads = config.num_attention_heads
        head_dim = embed_dim // num_heads  # Ensure integer division for head dimension

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )

        
        self.heads = nn.ModuleList(
            [
                AttentionHead(embed_dim, head_dim) for _ in range(num_heads)
            ]  # Create num_heads attention heads
        )
        self.att_mats = [i.att_mat for i in self.heads]
        # Linear layer to project the concatenated outputs back to embed_dim
        self.output_linear = nn.Linear(
            embed_dim, embed_dim
        )  # Input dimension is num_heads * head_dim which equals embed_dim

    def forward(self, hidden_state):
        """
        Perform forward pass through the multi-head attention module.

        For each attention head, the input tensor is passed through the corresponding AttentionHead instance,
        and the outputs are concatenated along the last dimension. The concatenated output is then passed
        through the output_linear layer to obtain the final output tensor.

        Args:
            hidden_state (torch.Tensor): Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            torch.Tensor: Output tensor after applying multi-head attention and linear transformation
                of shape (batch_size, seq_len, embed_dim).

        """
        # Concatenate outputs from all heads along the feature dimension
    
        concatenated_output = torch.cat([h(hidden_state) for h in self.heads], dim=-1)
        # Apply the final linear layer
        concatenated_output = self.output_linear(concatenated_output)
        return concatenated_output


class FeedForward(nn.Module):
    """
    This class implements the Feed Forward neural network layer within the Transformer model.

    Feed Forward layer is a crucial part of the Transformer's architecture, responsible for the actual
    transformation of the input data. It consists of two linear layers with a GELU activation function
    in between, followed by a dropout layer for regularization.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - hidden_size: The size of the hidden layer in the transformer model.
        - intermediate_size: The size of the intermediate layer in the Feed Forward network.
        - hidden_dropout_prob: The dropout probability for the hidden layer.

    Attributes
    ----------
    linear1 : torch.nn.Module
        The first linear transformation layer.
    linear2 : torch.nn.Module
        The second linear transformation layer.
    gelu : torch.nn.Module
        The Gaussian Error Linear Unit (GELU) activation function.
    dropout : torch.nn.Module
        The dropout layer for regularization.
    """

    def __init__(self, config):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.linear2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the Feed Forward network layer.

        Returns
        -------
        x : torch.Tensor
            The output tensor after passing through the Feed Forward network layer.
        """
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    This class implements the Transformer Encoder Layer as part of the Transformer model.

    Each encoder layer consists of a Multi-Head Attention mechanism followed by a Position-wise
    Feed Forward neural network. Additionally, residual connections around each of the two
    sub-layers are employed, followed by layer normalization.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - hidden_size: The size of the hidden layer in the transformer model.

    Attributes
    ----------
    layer_norm_1 : torch.nn.Module
        The first layer normalization.
    layer_norm_2 : torch.nn.Module
        The second layer normalization.
    attention : MultiHeadAttention
        The MultiHeadAttention mechanism in the encoder layer.
    feed_forward : FeedForward
        The FeedForward neural network in the encoder layer.
    """

    def __init__(self, config):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm_2 = nn.LayerNorm(config.hidden_size)
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)

    def forward(self, x):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor to the Transformer Encoder Layer.

        Returns
        -------
        x : torch.Tensor
            The output tensor after passing through the Transformer Encoder Layer.
        """
        # Apply layer norm before attention and feed-forward (Pre-LN)
        hidden_state = self.layer_norm_1(x)
        # Attention block with residual connection
        x = x + self.attention(hidden_state)
        # Feed-forward block with residual connection
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )  # Apply layer norm before feed-forward
        return x


class Embeddings(nn.Module):
    """
    This class implements the Embeddings layer as part of the Transformer model.

    The Embeddings layer is responsible for converting input tokens and their corresponding positions
    into dense vectors of fixed size. The token embeddings and position embeddings are summed up
    and subsequently layer-normalized and passed through a dropout layer for regularization.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - vocab_size: The size of the vocabulary (corrected from vocal_size in notebook).
        - hidden_size: The size of the hidden layer in the transformer model.
        - max_position_embeddings: The maximum number of positions that the model can accept.
        - layer_norm_eps: Epsilon for layer normalization (added based on common practice).
        - hidden_dropout_prob: Dropout probability (added based on common practice).

    Attributes
    ----------
    token_embeddings : torch.nn.Module
        The embedding layer for the tokens.
    position_embeddings : torch.nn.Module
        The embedding layer for the positions.
    layer_norm : torch.nn.Module
        The layer normalization.
    dropout : torch.nn.Module
        The dropout layer for regularization.
    """

    def __init__(self, config):
        super().__init__()
        # Ensure config has vocab_size, not vocal_size
        if not hasattr(config, "vocab_size"):
            raise AttributeError("Config object must have 'vocab_size' attribute.")
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        # Use eps from config if available, otherwise default
        layer_norm_eps = getattr(config, "layer_norm_eps", 1e-12)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=layer_norm_eps)
        # Use hidden_dropout_prob from config if available, otherwise default to 0.0
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_ids):
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input tensor to the Embeddings layer, typically the token ids (shape: batch_size, seq_length).

        Returns
        -------
        embeddings : torch.Tensor
            The output tensor after passing through the Embeddings layer (shape: batch_size, seq_length, hidden_size).
        """
        seq_length = input_ids.size(1)
        # Create position IDs dynamically based on input sequence length
        # Ensure position_ids are on the same device as input_ids
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = token_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerEncoder(nn.Module):  # Renamed from TransformerEncode for clarity
    """
    This class implements the Transformer Encoder as part of the Transformer model.

    The Transformer Encoder consists of a series of identical layers, each with a self-attention mechanism
    and a position-wise fully connected feed-forward network. The input to each layer is first processed by
    the Embeddings layer which converts input tokens and their corresponding positions into dense vectors of
    fixed size.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - num_hidden_layers: The number of hidden layers in the encoder (corrected from num_hidden_layer).

    Attributes
    ----------
    embeddings : Embeddings
        The embedding layer which converts input tokens and positions into dense vectors.
    layers : torch.nn.ModuleList
        The list of Transformer Encoder Layers.
    """

    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        # Ensure config has num_hidden_layers attribute
        if not hasattr(config, "num_hidden_layers"):
            raise AttributeError(
                "Config object must have 'num_hidden_layers' attribute."
            )
        # Initialize a list of Transformer Encoder Layers. The number of layers is defined by config.num_hidden_layers
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self, input_ids
    ):  # Changed input from generic x to input_ids for clarity
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input tensor (token ids) to the Transformer Encoder (shape: batch_size, seq_length).

        Returns
        -------
        hidden_states : torch.Tensor
            The output tensor (hidden states) after passing through the Transformer Encoder
            (shape: batch_size, seq_length, hidden_size).
        """
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TransformerForSequenceClassification(nn.Module):
    """
    This class implements the Transformer model for sequence classification tasks.

    The model architecture consists of a Transformer encoder, followed by a dropout layer for regularization,
    and a linear layer for classification. The output from the [CLS] token's embedding is used for the classification task.

    Parameters
    ----------
    config : object
        The configuration object containing model parameters. It should have the following attributes:
        - hidden_size: The size of the hidden layer in the transformer model.
        - hidden_dropout_prob: The dropout probability for the hidden layer.
        - num_labels: The number of labels in the classification task.

    Attributes
    ----------
    encoder : TransformerEncoder
        The Transformer Encoder.
    dropout : torch.nn.Module
        The dropout layer for regularization.
    classifier : torch.nn.Module
        The classification layer.
    """

    def __init__(self, config):
        super().__init__()
        self.encoder = TransformerEncoder(config)  # Use renamed TransformerEncoder
        # Use hidden_dropout_prob from config if available, otherwise default to 0.0
        dropout_prob = getattr(config, "hidden_dropout_prob", 0.0)
        self.dropout = nn.Dropout(dropout_prob)
        # Ensure config has num_labels attribute
        if not hasattr(config, "num_labels"):
            raise AttributeError("Config object must have 'num_labels' attribute.")
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self, input_ids
    ):  # Changed input from generic x to input_ids for clarity
        """
        Defines the computation performed at every call.

        Parameters
        ----------
        input_ids : torch.Tensor
            The input tensor (token ids) to the Transformer model (shape: batch_size, seq_length).

        Returns
        -------
        logits : torch.Tensor
            The output tensor (logits) after passing through the Transformer model and the classification layer
            (shape: batch_size, num_labels).
        """
        # Pass input_ids to the encoder
        encoder_output = self.encoder(input_ids)
        # Select the hidden state corresponding to the first token ([CLS] token equivalent)
        pooled_output = encoder_output[:, 0, :]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
