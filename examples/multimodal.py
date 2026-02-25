"""Multimodal input (images) with Portkey.

This example demonstrates sending image content to the model alongside
text prompts.  The image is base64-encoded and passed as a content block.

Prerequisites
-------------
    export PORTKEY_API_KEY="your-portkey-api-key"
    export PORTKEY_VIRTUAL_KEY="your-virtual-key"

Usage:
    python examples/multimodal.py
"""

import base64
import os

from strands import Agent

from strands_portkey import PortkeyModel

model = PortkeyModel(
    client_args={
        "api_key": os.environ.get("PORTKEY_API_KEY"),
        "virtual_key": os.environ.get("PORTKEY_VIRTUAL_KEY"),
    },
    # Use a vision-capable model
    model_id="gpt-4o-mini",
)

agent = Agent(
    model=model,
    system_prompt="You are a helpful assistant that can analyze images.",
    callback_handler=None,
)

# ---------------------------------------------------------------------------
# 1. Image from file
# ---------------------------------------------------------------------------
print("=" * 60)
print("Image analysis")
print("=" * 60)

# Create a simple test image (1x1 red pixel PNG) for demonstration.
# In practice, load a real image file.
# Minimal valid PNG: 1x1 red pixel
RED_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
)

result = agent(
    [
        {"text": "Describe what you see in this image. Keep it brief."},
        {
            "image": {
                "format": "png",
                "source": {"bytes": RED_PIXEL_PNG},
            }
        },
    ]
)
print(f"Image analysis: {result}")

# ---------------------------------------------------------------------------
# 2. Load a real image from disk (if available)
# ---------------------------------------------------------------------------
image_path = os.environ.get("TEST_IMAGE_PATH")
if image_path and os.path.exists(image_path):
    print("\n" + "=" * 60)
    print(f"Analyzing: {image_path}")
    print("=" * 60)

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Detect format from extension
    ext = os.path.splitext(image_path)[1].lstrip(".")
    fmt = {"jpg": "jpeg"}.get(ext, ext)

    result = agent(
        [
            {"text": "What do you see in this image? Describe it in detail."},
            {"image": {"format": fmt, "source": {"bytes": image_bytes}}},
        ]
    )
    print(f"Analysis: {result}")
else:
    print("\nSet TEST_IMAGE_PATH to analyze a real image file.")
