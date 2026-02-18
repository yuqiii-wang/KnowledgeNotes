# ControlNet

Reference: https://openaccess.thecvf.com/content/ICCV2023/papers/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.pdf

ControlNet locks the production-ready large diffusion models, and reuses their deep and robust encoding layers pretrained with billions of images as a strong backbone to learn a diverse set of conditional controls.

The inputs are pairs of diffusion and conditional controls:

* The "Diffusion" Input (for the Locked Model):
    * Text Prompt: The standard text description (e.g., "a cybernetic robot painting").
    * Latent Noise: The initial random noise (or latent image from VAE) that the diffusion model intends to denoise into an image.
* The "Control" Input (specific spatial guide for the Trainable Copy), e.g.,:
    * A Canny edge map (black image with white lines).
    * A Human Pose stick figure (OpenPose result).
    * A Depth map (grayscale representing distance).
    * A Scribble or Sketch.

The outputs are

* Residual Feature Maps: The ControlNet branch outputs feature maps that are technically "residuals" (offsets).
* Integration: These residuals are added to the feature maps of the original locked U-Net's decoder layers.

<div style="display: flex; justify-content: center;">
      <img src="imgs/controlnet_results.png" width="70%" height="30%" alt="controlnet_results" />
</div>
</br>

## Innovation in ControlNet

<div style="display: flex; justify-content: center;">
      <img src="imgs/controlnet.png" width="40%" height="70%" alt="controlnet" />
</div>
</br>

The grey blocks are frozen and the blue blocks are trainable copies of model's encoding layers.

### Locked vs. Trainable Copies

ControlNet **locks** the parameters of the original diffusion model ($\Theta$) to preserve its generation capability. It then creates a **trainable copy** ($\Theta_c$) of the model's encoding layers to learn the specific conditioning task (like following edges or depth maps).

### Zero Convolutions

These are $1 \times 1$ convolution layers initialized with both weights and biases at zero.

* Before training, the output of the ControlNet branch is zero, meaning the model behaves exactly like the original unconditioned model.
* As training progresses, the zero convolutions learn non-zero weights to gradually introduce the control signal.

### Architecture Integration

In a typical U-Net architecture (used in Stable Diffusion), ControlNet copies the Encoder blocks and the Middle block. Those copies process the conditioning image (e.g., a Canny edge map). The outputs of these trainable copies are added to the corresponding Decoder skip-connections of the locked model.

### Training Efficiency

Because the original model is locked, gradients are not calculated for the massive original backbone. This makes training ControlNet robust even on small datasets (fewer than 50k images) and computationally feasible on consumer hardware.
