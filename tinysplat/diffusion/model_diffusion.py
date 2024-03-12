from typing import Dict, List, Optional
import itertools

import kornia
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection

from diffusers import UNet2DModel
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from ..scene import Camera
from ..utils import unproj_map


class FeatureVolumeEncoder(ModelMixin, ConfigMixin):
    """
    A pixelNeRF-like model that uses a UNet to encode input images into a feature volume.
    Takes in a batch of N RGB images (B, N, 3, H, W) and returns a feature volume batch (B, N, C, D, D)

    Notes:
    - B: Number of batches (i.e. target images)
    - N: Number of input images.
    - H, W: Image height and width, respectively.
    - C: Number of channels in the feature volume.
    - D: Latent dimension
    """

    @register_to_config
    def __init__(
        self,
        unet: Optional[Dict] = None,
        latent_dim: int = 64,
    ):
        assert unet is not None, "UNet configuration must be provided."

        super().__init__()
        self.image_dims = (unet['sample_size'], unet['sample_size'])
        self.num_channels = unet['out_channels']
        self.latent_dim = latent_dim
        self.encoder = UNet2DModel.from_config(unet)

    def forward(self, targets: List[Camera], inputs: List[List[Camera]]) -> Tensor:
        """
        Forward pass of the model.

        Assumes that the number of input cameras is the same in each batch.

        Parameters:
        - target: The target camera view for which we generate the feature output.
        - inputs: A list of input cameras that contribute to reconstruction.

        Returns:
        - features_projected: The projected feature volume for each input view. (N, C, D, D)
        - xyz: The projected sampled ray coordinates for each input view (N, C, D, D)
        """

        B, N = len(inputs), len(inputs[0])

        assert len(targets) == B, "Not enough target cameras specified for the input batches"

        # Create feature volumes from input images (N, C, H, W)
        images = [cam.get_original_image(dims=self.image_dims, pad=True).to(self.device) for cam in itertools.chain(*inputs)]
        images = [image.permute(2, 0, 1) for image in images]

        images = torch.stack(images, dim=0)
        features = self.encoder(images, 1).sample
        features = features.reshape(B, N, *features.shape[1:]) # (B, N, C, H, W)

        # Cast rays through pixels of the target cameras
        rays = torch.stack([self._cast_rays(target).squeeze(0) for target in targets], dim=0) # (B, H, W, 8)

        # Sample C points along each ray 
        points, z_samp = self._sample_rays(rays, num_points=self.num_channels) # (B, H, W, C, 3)

        # Reproject points back onto the input cameras and trilinearly sample corresponding features
        points = points.reshape(B, -1, 3)
        index = [i for i, batch in enumerate(inputs) for _ in batch]
        xyz = torch.stack([
            cam.project_points(points[i], screen_coordinates=False, return_depth=True) for (i, cam) in zip(index, itertools.chain(*inputs))
        ], dim=0) # (N*B, H*W*C, 3)
        xyz = xyz.reshape(B, N, *xyz.shape[1:]) # (B, N, H*W*C, 3)
        features_projected = self._sample_features(features, xyz, z_samp.min(), z_samp.max()) # (B, N, C, H, W)

        # Downsample projected features
        features_projected = features_projected.reshape(-1, self.num_channels, *self.image_dims)
        features_projected = features_projected.unsqueeze(0)
        features_projected = F.interpolate(
            features_projected,
            size=(self.num_channels, self.latent_dim, self.latent_dim),
            mode='trilinear',
        ).reshape(B, N, self.num_channels, self.latent_dim, self.latent_dim) # (N, C, D, D)

        # Downsample XYZ coordinates
        xyz = xyz.reshape(-1, self.image_dims[0], self.image_dims[1], self.num_channels, 3)
        xyz = xyz.permute(0, 4, 3, 1, 2)
        xyz = F.interpolate(
            xyz,
            size=(self.num_channels, self.latent_dim, self.latent_dim),
            mode='trilinear',
        ).reshape(B, N, 3, self.num_channels, self.latent_dim, self.latent_dim)  # (B, N, 3, C, D, D)

        # FIXME: Resolve NaN issue that arises during training (and remove `nan_to_num` band-aid)
        #print("isnan(xyz)", torch.isnan(xyz).any())
        xyz = torch.nan_to_num(xyz)

        return features_projected, xyz

    ## Modified from PixelNeRF (https://www.github.com/sxyu/pixel-nerf)
    def _cast_rays(self, target: Camera, c=None, ndc=False):
        """
        Generate camera rays
        :return (N, H, W, 8)
        """
        device = self.device
        num_images = 1 
        width = 512
        height = 512
        
        # Adjust focal length for new width and height
        f_x = target.f_x
        f_y = target.f_y
        c_x = target.width / 2
        c_y = target.height / 2
        f_x *= width / 2 / c_x
        f_y *= height / 2 / c_y
        
        focal = torch.as_tensor([f_x, f_y])
        z_near = target.z_near
        z_far = target.z_far
        cam_unproj_map = (
            unproj_map(width, height, focal, c=c, device=device)
            .unsqueeze(0)
                .repeat(num_images, 1, 1, 1)
        )
        view_mat = target.view_matrix.to(device).unsqueeze(0)
        cam_centers = torch.as_tensor(target.position).to(device)[None, None, None, :]
        cam_centers = cam_centers.expand(-1, height, width, -1).float()
        cam_raydir = torch.matmul(
            -torch.inverse(view_mat[:, None, None, :3, :3]), cam_unproj_map.unsqueeze(-1)
        )[:, :, :, :, 0]

        cam_nears = (
            torch.tensor(z_near, device=device)
            .view(1, 1, 1, 1)
            .expand(num_images, height, width, -1)
        )
        cam_fars = (
            torch.tensor(z_far, device=device)
            .view(1, 1, 1, 1)
            .expand(num_images, height, width, -1)
        )
        return torch.cat((cam_centers, cam_raydir, cam_nears, cam_fars), dim=-1)

    def _sample_rays(self, rays: Tensor, num_points: int) -> Tensor:
        """
        Sample C points along each ray

        Parameters:
        - rays: [origins (3), directions (3), near (1), far (1)] (B, H, W, 8)

        Returns:
        - points: (B, H, W, C, 3)
        - z_samp: (B, H, W, C, 1)
        """
        B = rays.shape[0]
        near, far = rays[..., -2:-1], rays[..., -1:]
        step = 1.0 / num_points
        rays = rays.unsqueeze(-2) # (B, H, W, 1, 8)

        z_steps = torch.linspace(0, 1 - step, num_points, device=self.device)
        z_steps = z_steps.unsqueeze(0).repeat(B, 1)
        z_steps = z_steps[:, None, None, :] # (B, 1, 1, C)

        # Use linear sampling in disparity space
        z_samp = 1 / (1 / near * (1 - z_steps) + 1 / far * z_steps) 
        z_samp = z_samp.unsqueeze(-1) # (B, H, W, C, 1)

        points = rays[..., :3] + 1./z_samp * rays[..., 3:6]
        return points, z_samp

    def _sample_features(self, features, xyz, z_min, z_max):
        """
        Get pixel-aligned image features at 2D image coordinates

        Parameters:
        - features:
        - xyz: Coordinates
        - z_min:
        - z_max:

        Returns:
        - samples: (B, N, C, H, W)
        """

        B, N, C, H, W = features.shape

        # Reshape and permute for 'grid_sample'
        features = features.reshape(B*N, 1, C, H, W).permute(0,1,3,4,2)
        xyz = xyz.reshape(B*N, 1, -1, 3).unsqueeze(2)

        # Remap Z-coordinate to lie within [-1, 1]
        xyz[...,2] = 2 * (xyz[...,2] - z_min) / (z_max - z_min) - 1

        samples = F.grid_sample(
            features,
            xyz[...,[2,0,1]],
            align_corners=True,
            mode="bilinear",
            padding_mode="border",
        )
        samples = samples.reshape(B,N,H,W,C).permute(0,1,4,2,3)

        return samples # (B, N, C, H, W)


class FeatureAggregator(ModelMixin, ConfigMixin):
    """
    Reduce feature volumes to a single (B, C+3, D, D) tensor.
    """

    @register_to_config
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.input_dim = input_dim
        output_dim = input_dim + 3

        # First MLP (to compute intermediate features and weights)
        self.fc1 = nn.Linear(input_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim * 2)

        # Second MLP (to compute final features and RGB)
        self.fc3 = nn.Linear(input_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, output_dim)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, xyz):
        """
        Forward pass of the MLPs

        Args:
          features: Feature volume (B, N, C, D, D)
          xyz: Coordinates (B, N, 3, C, D, D)

        Returns:
          output: Output features + RGB tensor of shape (B, C+3, D, D).
        """

        # TODO: Positionally encode XYZ
        B, N, C, D = features.shape[:4]
        xyz = xyz.permute(2, 0, 1, 4, 5, 3)
        x = xyz[0, ...].reshape(-1, C)
        y = xyz[1, ...].reshape(-1, C)
        z = xyz[2, ...].reshape(-1, C) # (B*N*D*D, C)

        # Concatenate features and positionally-encoded coordinates
        features = features.permute(0, 1, 3, 4, 2).reshape(-1, C) # (B*N*D*D, C)
        input_batch = torch.cat([features, x, y, z], dim=1)

        # Process the input tensor using the first MLP
        h1 = self.fc1(input_batch)
        h2 = self.fc2(h1)
        tmp_features = self.silu(h2[:, :self.input_dim])
        weights = self.sigmoid(h2[:, self.input_dim:])

        # Apply weights and sum over the N dimension
        tmp_features = tmp_features * weights
        tmp_features = tmp_features.reshape(B, N, D, D, C)
        tmp_features = tmp_features.sum(dim=1) # (B, D, D, C)

        # Run intermediate features through a second MLP to get the final output
        h3 = self.silu(self.fc3(tmp_features))
        output = self.fc4(h3)
        output = output.permute(0, 3, 1, 2)
        return output


class EmbeddingMLP(ModelMixin, ConfigMixin):
    """
    Fully-connected layer used to feed conditioned CLIP image embeddings into the cross-attention mechanism.
    """

    @register_to_config
    def __init__(
        self,
        conditioned_images: int = 3
    ):
        super().__init__()
        self.input_dim = 768 * (conditioned_images + 2)
        self.output_dim = 768 * 2
        self.projection = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, clip_text_embedding: Tensor, input_embeds: Tensor) -> Tensor:
        """
        Inputs:
        - clip_text_embedding: (B, 2, 768)
        - input_embeds: (B, M, 768)

        Returns:
        - out: (B, 2, 768)

        Notes:
        - B: Number of batches
        - M: The number of conditioned images
        """

        B = input_embeds.shape[0]
        x = torch.cat([clip_text_embedding, input_embeds], dim=1).reshape(B, -1)
        out = self.projection(x).reshape(B, -1, 768)
        return out

def CLIP_preprocess(x):
    # TODO HF OpenAI CLIP preprocessing issue https://github.com/huggingface/transformers/issues/22505#issuecomment-1650170741
    # Follow OpenAI preprocessing to keep exact same, input tensor [-1, 1], otherwise the preprocessing will be different, https://github.com/huggingface/transformers/pull/22608
    if isinstance(x, torch.Tensor):
        if x.min() < -1.0 or x.max() > 1.0:
            raise ValueError("Expected input tensor to have values in the range [-1, 1]")
    x = kornia.geometry.resize(x.to(torch.float32), (224, 224), interpolation='bicubic', align_corners=True, antialias=False).to(dtype=x.dtype)
    x = (x + 1.) / 2.

    # Renormalize according to CLIP
    x = kornia.enhance.normalize(
        x,
        torch.Tensor([0.48145466, 0.4578275, 0.40821073]),
        torch.Tensor([0.26862954, 0.26130258, 0.27577711])
    )
    return x

def encode_images(
    image_encoder: CLIPVisionModelWithProjection,
    images: List[Tensor],
    device: str = "cuda",
):
    dtype = next(image_encoder.parameters()).dtype
    if not isinstance(images, torch.Tensor):
        raise ValueError(f"`image` has to be of type `torch.Tensor` but is {type(image)}")

    # Batch single image
    if images.ndim == 3:
        assert images.shape[0] == 3, "Image outside a batch should be of shape (3, H, W)"
        images = images.unsqueeze(0)
    assert images.ndim == 4, "Image must have 4 dimensions"

    # Check images are in [-1, 1]
    if images.min() < -1 or images.max() > 1:
        raise ValueError("Image should be in [-1, 1] range")

    images = images.to(device=device, dtype=dtype)
    images = CLIP_preprocess(images)
    image_embeddings = image_encoder(images).image_embeds.to(dtype=dtype)

    return image_embeddings

def encode_cross_attention_inputs(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    image_encoder: CLIPVisionModelWithProjection,
    embedding_mlp: EmbeddingMLP,
    cameras: List[Camera],
    height: int,
    width: int,
    device: str = "cuda",
    do_classifier_free_guidance: bool = False,
):
    # Empty text embedding
    # TODO: Move text embeds outside of this function (it should be initialized just once)
    text_inputs = tokenizer("", return_tensors="pt")
    text_input_ids = text_inputs.input_ids
    if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
        attention_mask = text_inputs.attention_mask.to(device)
    else:
        attention_mask = None
    text_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)[0]
    text_embeds = text_embeds.repeat(len(cameras), 1, 1)

    # Image embeddings
    images = [cam.get_original_image(dims=(height, width), pad=True) for cam in itertools.chain(*cameras)]
    images = [image.permute(2, 0, 1) for image in images]
    images = torch.stack(images, dim=0)
    image_embeds = encode_images(image_encoder, images, device)
    image_embeds = image_embeds.reshape(len(cameras), -1, 768)

    # Combine empty text and image embeddings 
    prompt_embeds = embedding_mlp(text_embeds, image_embeds)

    # Concatenate the negative embedding (the empty text embed) if doing classifier free guidance
    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([text_embeds, prompt_embeds])

    return prompt_embeds

def prepare_feature_latents(
    feature_encoder: FeatureVolumeEncoder,
    feature_aggregator: FeatureAggregator,
    target_cameras: List[Camera],
    input_cameras: List[List[Camera]],
    device: str = "cuda",
    do_classifier_free_guidance=False,
):
    # Construct a feature volume of shape (B, C, D, D)
    features, xyz = feature_encoder(target_cameras, input_cameras)
    features = feature_aggregator(features, xyz)

    # Adjust for classifier free guidance
    features = torch.cat([torch.zeros_like(features, device=device), features]) if do_classifier_free_guidance else features

    return features
