"""
OCVP (Object-Centric Video Prediction) Predictor Modules.

Adapted from: https://github.com/AIS-Bonn/OCVP-object-centric-video-prediction
Paper: "Object-Centric Video Prediction via Decoupling of Object Dynamics and Interactions" (CVPR 2023)

This module implements:
- OCVPSeq: Sequential object/time attention (alternating)
- OCVPPar: Parallel object/time attention (combined)
- OCVPWrapper: Autoregressive rollout wrapper for predictions
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class PositionalEncoding(nn.Module):
    """
    Positional encoding to be added to the input tokens of the transformer predictor.

    Our positional encoding only informs about the time-step, i.e., all slots extracted
    from the same input frame share the same positional embedding. This allows our predictor
    model to maintain the permutation equivariance properties.

    Args:
    -----
    d_model: int
        Dimensionality of the slots/tokens
    dropout: float
        Percentage of dropout to apply after adding the positional encoding. Default is 0.1
    max_len: int
        Length of the sequence.
    """

    def __init__(self, d_model, dropout=0.1, max_len=50):
        """
        Initializing the positional encoding
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # initializing sinusoidal positional embedding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.view(1, max_len, 1, d_model)
        self.pe = pe  # Not registered as buffer to match original implementation

    def forward(self, x, batch_size, num_slots):
        """
        Adding the positional encoding to the input tokens of the transformer

        Args:
        -----
        x: torch Tensor
            Tokens to enhance with positional encoding. Shape is (B, Seq_len, Num_Slots, Token_Dim)
        batch_size: int
            Given batch size to repeat the positional encoding for
        num_slots: int
            Number of slots to repeat the positional encoder for
        """
        if x.device != self.pe.device:
            self.pe = self.pe.to(x.device)
        cur_seq_len = x.shape[1]
        cur_pe = self.pe.repeat(batch_size, 1, num_slots, 1)[:, :cur_seq_len]
        x = x + cur_pe
        y = self.dropout(x)
        return y


class OCVPSeqLayer(nn.Module):
    """
    Sequential Object-Centric Video Prediction (OCVP-Seq) Transformer Layer.
    Sequentially applies object- and time-attention.

    Args:
    -----
    token_dim: int
        Dimensionality of the input tokens
    hidden_dim: int
        Hidden dimensionality of the MLPs in the transformer modules
    n_heads: int
        Number of heads for multi-head self-attention.
    """

    def __init__(self, token_dim=128, hidden_dim=256, n_heads=4):
        """
        Module initializer
        """
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.nhead = n_heads

        self.object_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )
        self.time_encoder_block = torch.nn.TransformerEncoderLayer(
                d_model=token_dim,
                nhead=self.nhead,
                batch_first=True,
                norm_first=True,
                dim_feedforward=hidden_dim
            )

    def forward(self, inputs, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-V1 Layer

        Args:
        -----
        inputs: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        time_mask: torch Tensor (optional)
            Attention mask for temporal attention. Currently unused.
        """
        B, num_imgs, num_slots, dim = inputs.shape

        # object-attention block. Operates on (B * N_imgs, N_slots, Dim)
        inputs = inputs.reshape(B * num_imgs, num_slots, dim)
        object_encoded_out = self.object_encoder_block(inputs)
        object_encoded_out = object_encoded_out.reshape(B, num_imgs, num_slots, dim)

        # time-attention block. Operates on (B * N_slots, N_imgs, Dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        object_encoded_out = object_encoded_out.reshape(B * num_slots, num_imgs, dim)
        object_encoded_out = self.time_encoder_block(object_encoded_out)
        object_encoded_out = object_encoded_out.reshape(B, num_slots, num_imgs, dim)
        object_encoded_out = object_encoded_out.transpose(1, 2)
        return object_encoded_out


class OCVPSeq(nn.Module):
    """
    Sequential Object-Centric Video Prediction Transformer Module (OCVP-Seq).
    This module models the temporal dynamics and object interactions in a decoupled manner by
    sequentially applying object- and time-attention, i.e. [time, obj, time, ...]

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, num_imgs=6, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5):
        """
        Module Initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        logger.info("Instantiating OCVP-Seq Predictor Module:")
        logger.info(f"  --> num_layers: {self.num_layers}")
        logger.info(f"  --> input_dim: {self.slot_dim}")
        logger.info(f"  --> token_dim: {self.token_dim}")
        logger.info(f"  --> hidden_dim: {self.hidden_dim}")
        logger.info(f"  --> num_heads: {self.nhead}")
        logger.info(f"  --> residual: {self.residual}")
        logger.info("  --> batch_first: True")
        logger.info("  --> norm_first: True")
        logger.info(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # Embed_dim will be split across num_heads, i.e., each head will have dim. embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[OCVPSeqLayer(
                    token_dim=token_dim,
                    hidden_dim=hidden_dim,
                    n_heads=n_heads
                ) for _ in range(num_layers)]
            )

        # custom temporal encoding. All slots from the same time step share the same encoding
        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)

    def forward(self, inputs):
        """
        Forward pass through OCVP-Seq

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # projecting slots into tokens, and applying positional encoding
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding through OCVP-Seq transformer blocks
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # mapping back to the slot dimension
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output


class OCVPParLayer(nn.TransformerEncoderLayer):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Module.
    This module models the temporal dynamics and object interactions in a dissentangled manner by
    applying object- and time-attention in parallel.

    Args:
    -----
    d_model: int
        Dimensionality of the input tokens
    nhead: int
        Number of heads in multi-head attention
    dim_feedforward: int
        Hidden dimension in the MLP
    dropout: float
        Amount of dropout to apply. Default is 0.1
    activation: callable
        Nonlinear activation in the MLP. Default is ReLU
    layer_norm_eps: float
        Epsilon value in the layer normalization components
    batch_first: bool
        If True, shape is (B, num_tokens, token_dim); otherwise, it is (num_tokens, B, token_dim)
    norm_first: bool
        If True, transformer is in mode pre-norm; otherwise, it is post-norm
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=True, norm_first=True, device=None, dtype=None):
        """
        Module initializer
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                layer_norm_eps=layer_norm_eps,
                batch_first=batch_first,
                norm_first=norm_first,
                device=device,
                dtype=dtype
            )

        self.self_attn_obj = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )
        self.self_attn_time = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout,
                batch_first=batch_first,
                **factory_kwargs
            )

    def forward(self, src, time_mask=None):
        """
        Forward pass through the Object-Centric Transformer-v2.
        Overloads PyTorch's transformer forward pass.

        Args:
        -----
        src: torch Tensor
            Tokens corresponding to the object slots from the input images.
            Shape is (B, N_imgs, N_slots, Dim)
        time_mask: torch Tensor (optional)
            Attention mask for temporal attention.
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), time_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, time_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(self, x, time_mask):
        """
        Forward pass through the parallel attention branches
        """
        B, num_imgs, num_slots, dim = x.shape

        # object-attention
        x_aux = x.clone().view(B * num_imgs, num_slots, dim)
        x_obj = self.self_attn_obj(
                query=x_aux,
                key=x_aux,
                value=x_aux,
                need_weights=False
            )[0]
        x_obj = x_obj.view(B, num_imgs, num_slots, dim)

        # time-attention
        x = x.transpose(1, 2).reshape(B * num_slots, num_imgs, dim)
        x_time = self.self_attn_time(
                query=x,
                key=x,
                value=x,
                attn_mask=time_mask,
                need_weights=False
            )[0]
        x_time = x_time.view(B, num_slots, num_imgs, dim).transpose(1, 2)

        y = self.dropout1(x_obj + x_time)
        return y


class OCVPPar(nn.Module):
    """
    Parallel Object-Centric Video Prediction (OCVP-Par) Transformer Predictor Module.
    This module models the temporal dynamics and object interactions in a dissentangled manner by
    applying relational- and temporal-attention in parallel.

    Args:
    -----
    num_slots: int
        Number of slots per image. Number of inputs to Transformer is num_slots * num_imgs
    slot_dim: int
        Dimensionality of the input slots
    num_imgs: int
        Number of images to jointly process. Number of inputs to Transformer is num_slots * num_imgs
    token_dim: int
        Input slots are mapped to this dimensionality via a fully-connected layer
    hidden_dim: int
        Hidden dimension of the MLPs in the transformer blocks
    num_layers: int
        Number of transformer blocks to sequentially apply
    n_heads: int
        Number of attention heads in multi-head self attention
    residual: bool
        If True, a residual connection bridges across the predictor module
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input
    """

    def __init__(self, num_slots, slot_dim, num_imgs=6, token_dim=128, hidden_dim=256, num_layers=2,
                 n_heads=4, residual=False, input_buffer_size=5):
        """
        Module initializer
        """
        super().__init__()
        self.num_slots = num_slots
        self.num_imgs = num_imgs
        self.slot_dim = slot_dim
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = n_heads
        self.residual = residual
        self.input_buffer_size = input_buffer_size

        # Encoder will be applied on tensor of shape (B, nhead, slot_dim)
        logger.info("Instantiating OCVP-Par Predictor Module:")
        logger.info(f"  --> num_layers: {self.num_layers}")
        logger.info(f"  --> input_dim: {self.slot_dim}")
        logger.info(f"  --> token_dim: {self.token_dim}")
        logger.info(f"  --> hidden_dim: {self.hidden_dim}")
        logger.info(f"  --> num_heads: {self.nhead}")
        logger.info(f"  --> residual: {self.residual}")
        logger.info("  --> batch_first: True")
        logger.info("  --> norm_first: True")
        logger.info(f"  --> input_buffer_size: {self.input_buffer_size}")

        # Linear layers to map from slot_dim to token_dim, and back
        self.mlp_in = nn.Linear(slot_dim, token_dim)
        self.mlp_out = nn.Linear(token_dim, slot_dim)

        # embed_dim will be split across num_heads, i.e. each head will have dim embed_dim // num_heads
        self.transformer_encoders = nn.Sequential(
            *[OCVPParLayer(
                    d_model=token_dim,
                    nhead=self.nhead,
                    batch_first=True,
                    norm_first=True,
                    dim_feedforward=hidden_dim
                ) for _ in range(num_layers)]
            )

        self.pe = PositionalEncoding(d_model=self.token_dim, max_len=input_buffer_size)

    def forward(self, inputs):
        """
        Forward pass through Object-Centric Transformer v1

        Args:
        -----
        inputs: torch Tensor
            Input object slots from the previous time steps. Shape is (B, num_imgs, num_slots, slot_dim)

        Returns:
        --------
        output: torch Tensor
            Predictor object slots. Shape is (B, num_imgs, num_slots, slot_dim), but we only care about
            the last time-step, i.e., (B, -1, num_slots, slot_dim).
        """
        B, num_imgs, num_slots, slot_dim = inputs.shape

        # projecting slots and applying positional encodings
        token_input = self.mlp_in(inputs)
        time_encoded_input = self.pe(
                x=token_input,
                batch_size=B,
                num_slots=num_slots
            )

        # feeding tokens through transformer layers
        token_output = time_encoded_input
        for encoder in self.transformer_encoders:
            token_output = encoder(token_output)

        # projecting back to slot-dimensionality
        output = self.mlp_out(token_output)
        output = output + inputs if self.residual else output
        return output


class OCVPWrapper(nn.Module):
    """
    Wrapper module that autoregressively applies any predictor module on a sequence of data.
    Adapted from the original PredictorWrapper to work with our pipeline.

    Args:
    -----
    predictor: nn.Module
        Instantiated predictor module to wrap.
    num_context: int
        Number of context frames used for prediction.
    num_preds: int
        Number of frames to predict.
    input_buffer_size: int
        Maximum number of consecutive time steps that the transformer receives as input.
    skip_first_slot: bool
        If True, skip the first slot (useful when first frame has no motion info).
    """

    def __init__(
        self,
        predictor: nn.Module,
        num_context: int = 6,
        num_preds: int = 10,
        input_buffer_size: int = 6,
        skip_first_slot: bool = False,
    ):
        """
        Module initializer
        """
        super().__init__()
        self.predictor = predictor
        self.num_context = num_context
        self.num_preds = num_preds
        self.input_buffer_size = input_buffer_size
        self.skip_first_slot = skip_first_slot

        # Adjust input_buffer_size if needed
        if self.input_buffer_size < self.num_context:
            logger.warning(f"  --> input_buffer_size ({self.input_buffer_size}) is too small.")
            logger.warning(f"  --> Using num_context ({self.num_context}) instead...")
            self.input_buffer_size = self.num_context
        else:
            logger.info(f"  --> Using buffer size {self.input_buffer_size}...")

    def forward(self, slot_history: torch.Tensor, teacher_force: bool = False) -> torch.Tensor:
        """
        Iterating over a sequence of slots, predicting the subsequent slots

        Args:
        -----
        slot_history: torch Tensor
            Decomposed slots from the seed and predicted images.
            Shape is (B, num_frames, num_slots, slot_dim)
        teacher_force: bool
            If True, use ground truth for next input instead of predictions.

        Returns:
        --------
        pred_slots: torch Tensor
            Predicted subsequent slots. Shape is (B, num_preds, num_slots, slot_dim)
        """
        first_slot_idx = 1 if self.skip_first_slot else 0
        predictor_input = slot_history[:, first_slot_idx:self.num_context].clone()  # initial token buffer

        pred_slots = []
        for t in range(self.num_preds):
            cur_preds = self.predictor(predictor_input)[:, -1]  # get predicted slots from last step
            if teacher_force and (self.num_context + t) < slot_history.shape[1]:
                next_input = slot_history[:, self.num_context + t]
            else:
                next_input = cur_preds
            predictor_input = torch.cat([predictor_input, next_input.unsqueeze(1)], dim=1)
            predictor_input = self._update_buffer_size(predictor_input)
            pred_slots.append(cur_preds)
        pred_slots = torch.stack(pred_slots, dim=1)  # (B, num_preds, num_slots, slot_dim)
        return pred_slots

    def _update_buffer_size(self, inputs):
        """
        Updating the inputs of a transformer model given the 'buffer_size'.
        We keep a moving window over the input tokens, dropping the oldest slots if the buffer
        size is exceeded.
        """
        num_inputs = inputs.shape[1]
        if num_inputs > self.input_buffer_size:
            extra_inputs = num_inputs - self.input_buffer_size
            inputs = inputs[:, extra_inputs:]
        return inputs


def build_ocvp_predictor(
    predictor_type: str = "ocvp_seq",
    num_slots: int = 7,
    slot_dim: int = 128,
    num_imgs: int = 6,
    token_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    n_heads: int = 4,
    residual: bool = False,
    input_buffer_size: int = 6,
    num_context: int = 6,
    num_preds: int = 10,
    skip_first_slot: bool = False,
) -> OCVPWrapper:
    """
    Factory function to build OCVP predictor with wrapper.

    Args:
    -----
    predictor_type: str
        "ocvp_seq" or "ocvp_par"
    num_slots: int
        Number of slots per image
    slot_dim: int
        Slot dimensionality
    num_imgs: int
        Number of images to jointly process
    token_dim: int
        Internal token dimension
    hidden_dim: int
        FFN hidden dimension
    num_layers: int
        Number of transformer layers
    n_heads: int
        Number of attention heads
    residual: bool
        Use residual connection (default False to match original)
    input_buffer_size: int
        Max buffer size for transformer input
    num_context: int
        Number of context frames
    num_preds: int
        Number of prediction frames
    skip_first_slot: bool
        If True, skip the first slot in context

    Returns:
    --------
    OCVPWrapper containing the specified predictor
    """
    predictor_cls = OCVPSeq if predictor_type.lower() == "ocvp_seq" else OCVPPar

    predictor = predictor_cls(
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_imgs=num_imgs,
        token_dim=token_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        n_heads=n_heads,
        residual=residual,
        input_buffer_size=input_buffer_size,
    )

    wrapper = OCVPWrapper(
        predictor=predictor,
        num_context=num_context,
        num_preds=num_preds,
        input_buffer_size=input_buffer_size,
        skip_first_slot=skip_first_slot,
    )

    return wrapper
