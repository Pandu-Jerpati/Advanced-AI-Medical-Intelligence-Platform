import os
import cv2
import torch
import numpy as np


# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(
    model,
    input_tensor,
    original_image,
    predicted_index,
    output_path
):
    """
    Generate Grad-CAM overlay for DenseNet121.

    Parameters:
        model            : Loaded DenseNet121 model
        input_tensor     : Preprocessed image tensor
        original_image   : Original PIL image
        predicted_index  : Predicted class index
        output_path      : Path to save Grad-CAM overlay

    Returns:
        output_path
    """

    # ========================================================
    # GRAD-CAM VARIABLES
    # ========================================================

    activations = None
    gradients = None

    # ========================================================
    # FORWARD HOOK
    # ========================================================

    def forward_hook(module, input, output):
        nonlocal activations
        activations = output

    # ========================================================
    # BACKWARD HOOK
    # ========================================================

    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    # ========================================================
    # TARGET LAYER
    # ========================================================

    target_layer = model.features.denseblock4

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    backward_handle = target_layer.register_full_backward_hook(
        backward_hook
    )

    try:

        # ====================================================
        # FORWARD PASS
        # ====================================================

        model.zero_grad()

        output = model(input_tensor)

        # ====================================================
        # SELECT PREDICTED CLASS
        # ====================================================

        score = output[0, predicted_index]

        # ====================================================
        # BACKWARD PASS
        # ====================================================

        score.backward()

        # ====================================================
        # CHECK ACTIVATIONS
        # ====================================================

        if activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # ====================================================
        # DETACH TENSORS
        # ====================================================

        activation = activations.detach()
        gradient = gradients.detach()

        # ====================================================
        # CALCULATE GRAD-CAM WEIGHTS
        # ====================================================

        weights = gradient.mean(
            dim=(2, 3),
            keepdim=True
        )

        # ====================================================
        # CREATE CAM
        # ====================================================

        cam = (
            weights * activation
        ).sum(dim=1)

        cam = torch.relu(cam)

        cam = cam.squeeze()

        cam = cam.cpu().numpy()

        # ====================================================
        # NORMALIZE CAM
        # ====================================================

        cam -= cam.min()

        if cam.max() != 0:
            cam /= cam.max()

        # ====================================================
        # CONVERT ORIGINAL IMAGE
        # ====================================================

        original_np = np.array(
            original_image.convert("RGB")
        )

        # ====================================================
        # RESIZE HEATMAP
        # ====================================================

        heatmap = cv2.resize(
            cam,
            (
                original_np.shape[1],
                original_np.shape[0]
            )
        )

        # ====================================================
        # CONVERT HEATMAP TO 0-255
        # ====================================================

        heatmap_uint8 = np.uint8(
            255 * heatmap
        )

        # ====================================================
        # CREATE COLOR HEATMAP
        # ====================================================

        heatmap_color = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        # ====================================================
        # RGB → BGR
        # ====================================================

        original_bgr = cv2.cvtColor(
            original_np,
            cv2.COLOR_RGB2BGR
        )

        # ====================================================
        # CREATE OVERLAY
        # ====================================================

        overlay = cv2.addWeighted(
            original_bgr,
            0.6,
            heatmap_color,
            0.4,
            0
        )

        # ====================================================
        # CREATE OUTPUT DIRECTORY
        # ====================================================

        output_directory = os.path.dirname(
            output_path
        )

        if output_directory:
            os.makedirs(
                output_directory,
                exist_ok=True
            )

        # ====================================================
        # SAVE OVERLAY
        # ====================================================

        success = cv2.imwrite(
            output_path,
            overlay
        )

        if not success:
            raise RuntimeError(
                f"Failed to save Grad-CAM image: {output_path}"
            )

        return output_path

    finally:

        # ====================================================
        # REMOVE HOOKS
        # ====================================================

        forward_handle.remove()
        backward_handle.remove()