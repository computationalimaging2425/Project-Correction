from diffusers import UNet2DModel

def get_unet_model(sample_size=128, in_channels=1, out_channels=1, layers_per_block=2, block_out_channels=(64, 128, 256, 512), down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"), up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")):
    """
    Function to create a UNet model for image generation.

    Parameters:
    IMAGE_SIZE (int): The size of the images to be generated.
    LEARNING_RATE (float): The learning rate for the optimizer.

    Returns:
    UNet2DModel: A UNet model configured for image generation.
    """

    # Create the UNet model with the specified parameters
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=layers_per_block,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
    )

    return model