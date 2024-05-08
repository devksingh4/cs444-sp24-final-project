from modified_vit import VisionTransformer
import torch.nn as nn
from segformer import SegFormerMLPDecoder
from transformers.models.segformer.configuration_segformer import SegformerConfig
from transformers.models.segformer.modeling_segformer import SegformerPreTrainedModel
import torch


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config: SegformerConfig, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHeadModified(SegformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # linear layers to unify the channel dimension of each encoder block
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following layers implement the ConvModule of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * 2 * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.Mish()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size, config.num_labels, kernel_size=1)

        # Additional layers for skip connections
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(config.hidden_sizes[i], config.decoder_hidden_size, kernel_size=1) for i in range(config.num_encoder_blocks)
        ])
        self.skip_norms = nn.ModuleList([nn.BatchNorm2d(config.decoder_hidden_size) for _ in range(config.num_encoder_blocks)])
        self.skip_activations = nn.ModuleList([nn.Mish() for _ in range(config.num_encoder_blocks)])

    def forward(self, encoder_hidden_states):
        batch_size, _, _, _ = encoder_hidden_states[-1].shape
        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.linear_c):
            # Unify channel dimension
            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            # Upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (encoder_hidden_state,)

        # Apply skip connections
        for i, encoder_hidden_state in enumerate(encoder_hidden_states):
            skip_out = self.skip_convs[i](encoder_hidden_state)
            skip_out = self.skip_norms[i](skip_out)
            skip_out = self.skip_activations[i](skip_out)
            # Upsample skip connection output to the same size as the final decoder features
            skip_out = nn.functional.interpolate(
                skip_out, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
            )
            all_hidden_states += (skip_out,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Logits are of shape (batch_size, num_labels, height, width)
        logits = self.classifier(hidden_states)

        return logits
