
import io
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision.models import resnet18
from torchvision import transforms


# ------------------ Configuration ------------------
IMG_SIZE = 128
MODEL_PATH = Path(__file__).parent / "best_generator.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------ Generator Architecture ------------------
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()

        # No need to download pretrained weights during inference
        r = resnet18(weights=None)

        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.enc1 = r.layer1
        self.enc2 = r.layer2
        self.enc3 = r.layer3
        self.enc4 = r.layer4

        self.dec3 = Block(512 + 256, 256)
        self.dec2 = Block(256 + 128, 128)
        self.dec1 = Block(128 + 64, 64)
        self.dec0 = Block(64 + 64, 64)

        self.out = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def up(self, x, skip, block):
        x = F.interpolate(
            x, size=skip.shape[-2:],
            mode="bilinear", align_corners=False
        )
        return block(torch.cat([x, skip], dim=1))

    def forward(self, gray):
        x = gray.repeat(1, 3, 1, 1)

        s0 = self.stem(x)
        s1 = self.enc1(self.pool(s0))
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        x = self.up(s4, s3, self.dec3)
        x = self.up(x, s2, self.dec2)
        x = self.up(x, s1, self.dec1)
        x = self.up(x, s0, self.dec0)

        x = F.interpolate(
            x, size=gray.shape[-2:],
            mode="bilinear", align_corners=False
        )
        return self.out(x)


# ------------------ Load Trained Model ------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH.name}. "
            "Place the trained generator file beside app.py."
        )

    model = Generator(pretrained=False).to(device)

    try:
        state = torch.load(
            MODEL_PATH, map_location=device, weights_only=True
        )
    except TypeError:
        state = torch.load(MODEL_PATH, map_location=device)

    # Supports a plain generator state_dict or a full checkpoint
    if "generator" in state:
        state = state["generator"]

    model.load_state_dict(state)
    model.eval()
    return model


# ------------------ Streamlit UI ------------------
st.set_page_config(
    page_title="Old Photo Colorization Using GAN",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Old Photo Colorization Using GAN")
st.write(
    "Upload a black-and-white photo and generate a colorized image "
    "using your trained Conditional GAN."
)

st.info(
    "This model was trained on your uploaded dataset. "
    "If that dataset contains mainly fruits and vegetables, "
    "results on historical portraits or buildings may be limited."
)

uploaded_file = st.file_uploader(
    "Upload a grayscale or old photograph",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    original = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded photograph")
    st.image(original, caption="Original image", width=500)

    if st.button("✨ Colorize Image", type="primary"):
        try:
            with st.spinner("Generating color image..."):
                model = load_model()

                # Match the grayscale preprocessing used in training
                gray_image = original.convert("L").resize(
                    (IMG_SIZE, IMG_SIZE),
                    Image.Resampling.LANCZOS
                )

                tensor = transforms.ToTensor()(gray_image)
                tensor = (tensor * 2 - 1).unsqueeze(0).to(device)

                with torch.inference_mode():
                    prediction = model(tensor)[0].cpu()

                prediction = (
                    (prediction.permute(1, 2, 0).numpy() + 1) / 2
                ).clip(0, 1)

                color_image = Image.fromarray(
                    (prediction * 255).round().astype("uint8")
                )

                # Resize output to the original photo dimensions
                color_image = color_image.resize(
                    original.size, Image.Resampling.LANCZOS
                )

            st.success("Colorization completed!")

            col1, col2 = st.columns(2)

            with col1:
                st.image(
                    original,
                    caption="Original photograph",
                    use_container_width=True
                )

            with col2:
                st.image(
                    color_image,
                    caption="GAN colorized output",
                    use_container_width=True
                )

            # Prepare downloadable PNG
            buffer = io.BytesIO()
            color_image.save(buffer, format="PNG")

            st.download_button(
                "⬇️ Download Colorized Image",
                data=buffer.getvalue(),
                file_name="colorized_photo.png",
                mime="image/png"
            )

        except Exception as e:
            st.error(f"Could not colorize the image: {e}")
            st.caption(
                "Check that the checkpoint is a generator state_dict "
                "saved from the matching Colab architecture."
            )

st.divider()
st.caption("Project: Old Photo Colorization Using Conditional GAN")
