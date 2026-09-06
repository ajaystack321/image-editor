import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
import numpy as np
import cv2
import io
import zipfile
from rembg import remove
import base64

# Page config - Professional Dark UI
st.set_page_config(
    page_title="AI Image Studio Pro",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark professional look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    .stButton>button {
        background-color: #00d4ff;
        color: #0e1117;
        border-radius: 8px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #00b8e6;
        color: white;
    }
    .stSlider > div > div > div {
        background-color: #00d4ff;
    }
    .css-1d391kg, .css-1v0mbdj {
        background-color: #1a1d24;
    }
    .uploadedFile {
        background-color: #1a1d24;
    }
    div[data-testid="stSidebar"] {
        background-color: #161b22;
    }
</style>
""", unsafe_allow_html=True)

# Session state initialization
if "images" not in st.session_state:
    st.session_state.images = []          # list of dicts: {id, original, current, name}
if "history" not in st.session_state:
    st.session_state.history = []          # for undo
if "selected_idx" not in st.session_state:
    st.session_state.selected_idx = 0

def get_image_bytes(img, format="PNG", quality=95):
    buf = io.BytesIO()
    if format.upper() == "JPG" or format.upper() == "JPEG":
        img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=quality, optimize=True)
    elif format.upper() == "WEBP":
        img.save(buf, format="WEBP", quality=quality, method=6)
    else:
        img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf

def apply_background_removal(img):
    """AI Background Removal using rembg"""
    img_rgba = img.convert("RGBA")
    result = remove(img_rgba)
    return result

def refine_edges(img, strength=1.0):
    """Simple edge refinement / hair cleanup using morphological ops + feather"""
    if img.mode != "RGBA":
        return img
    arr = np.array(img)
    alpha = arr[:, :, 3]
    
    # Morphological close/open for cleanup
    kernel = np.ones((3, 3), np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel)
    
    # Feather / slight blur on alpha for natural edges
    if strength > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), strength)
    
    arr[:, :, 3] = alpha
    return Image.fromarray(arr)

def apply_adjustments(img, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0, clarity=0.0):
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if saturation != 1.0:
        img = ImageEnhance.Color(img).enhance(saturation)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)
    
    # Clarity (local contrast via unsharp mask style)
    if clarity != 0:
        arr = np.array(img).astype(np.float32)
        blurred = cv2.GaussianBlur(arr, (0, 0), 3)
        arr = arr + clarity * (arr - blurred)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    return img

def apply_blur(img, radius=0):
    if radius > 0:
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img

def apply_background_blur(img, blur_radius=10):
    """Keep subject sharp, blur background (requires alpha)"""
    if img.mode != "RGBA":
        return img
    subject = img.copy()
    bg = img.convert("RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    bg = bg.convert("RGBA")
    # Composite
    result = Image.alpha_composite(bg, subject)
    return result

def apply_solid_background(img, color="#FFFFFF"):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, color)
    return Image.alpha_composite(bg, img)

def apply_gradient_background(img, color1="#1a1a2e", color2="#16213e", direction="vertical"):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = img.size
    gradient = Image.new("RGBA", (w, h))
    draw = ImageDraw.Draw(gradient)
    
    r1, g1, b1 = tuple(int(color1.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    r2, g2, b2 = tuple(int(color2.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    
    if direction == "vertical":
        for y in range(h):
            ratio = y / h
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            draw.line([(0, y), (w, y)], fill=(r, g, b, 255))
    else:  # horizontal
        for x in range(w):
            ratio = x / w
            r = int(r1 + (r2 - r1) * ratio)
            g = int(g1 + (g2 - g1) * ratio)
            b = int(b1 + (b2 - b1) * ratio)
            draw.line([(x, 0), (x, h)], fill=(r, g, b, 255))
    
    return Image.alpha_composite(gradient, img)

def apply_custom_background(img, bg_img):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    bg_img = bg_img.convert("RGBA").resize(img.size, Image.Resampling.LANCZOS)
    return Image.alpha_composite(bg_img, img)

def add_subject_shadow(img, offset=(8, 8), blur=12, opacity=120):
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    # Create shadow from alpha
    alpha = img.split()[-1]
    shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    shadow_alpha = alpha.point(lambda p: min(p, opacity))
    shadow.putalpha(shadow_alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur))
    
    # Offset shadow
    canvas = Image.new("RGBA", img.size, (0, 0, 0, 0))
    canvas.paste(shadow, offset)
    result = Image.alpha_composite(canvas, img)
    return result

def auto_enhance(img):
    """Simple auto enhance using histogram equalization + mild boost"""
    arr = np.array(img.convert("RGB"))
    # Convert to YCrCb and equalize Y channel
    ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    enhanced = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    img = Image.fromarray(enhanced)
    # Mild boost
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Color(img).enhance(1.1)
    img = ImageEnhance.Sharpness(img).enhance(1.2)
    return img

def hd_upscale(img, scale=2):
    """High-quality upscale using LANCZOS + mild sharpen"""
    w, h = img.size
    new_size = (int(w * scale), int(h * scale))
    upscaled = img.resize(new_size, Image.Resampling.LANCZOS)
    # Mild unsharp for clarity
    upscaled = ImageEnhance.Sharpness(upscaled).enhance(1.3)
    return upscaled

def crop_image(img, left, top, right, bottom):
    return img.crop((left, top, right, bottom))

def rotate_flip(img, angle=0, flip_h=False, flip_v=False):
    if angle != 0:
        img = img.rotate(angle, expand=True, fillcolor=(0, 0, 0, 0) if img.mode == "RGBA" else (0, 0, 0))
    if flip_h:
        img = ImageOps.mirror(img)
    if flip_v:
        img = ImageOps.flip(img)
    return img

# ==================== UI ====================
st.title("🖼️ AI Image Studio Pro")
st.caption("Professional local AI background removal + full editing suite • No API keys needed")

# Sidebar - Controls
with st.sidebar:
    st.header("⚙️ Controls")
    
    uploaded_files = st.file_uploader(
        "📁 Drag & Drop / Upload Images",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for f in uploaded_files:
            # Avoid duplicates by name
            if not any(img["name"] == f.name for img in st.session_state.images):
                img = Image.open(f).convert("RGBA")
                st.session_state.images.append({
                    "id": len(st.session_state.images),
                    "name": f.name,
                    "original": img.copy(),
                    "current": img.copy()
                })
                st.session_state.history.append([])  # history per image
    
    if st.session_state.images:
        names = [img["name"] for img in st.session_state.images]
        st.session_state.selected_idx = st.selectbox(
            "Select Image",
            range(len(names)),
            format_func=lambda i: names[i],
            index=min(st.session_state.selected_idx, len(names)-1)
        )
        
        st.divider()
        st.subheader("🎯 AI Tools")
        
        if st.button("🧹 Remove Background (AI)", use_container_width=True):
            with st.spinner("Removing background with rembg..."):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                result = apply_background_removal(curr)
                st.session_state.images[idx]["current"] = result
                st.success("Background removed!")
                st.rerun()
        
        edge_strength = st.slider("Edge / Hair Cleanup", 0.0, 5.0, 1.0, 0.1)
        if st.button("✨ Refine Edges", use_container_width=True):
            idx = st.session_state.selected_idx
            curr = st.session_state.images[idx]["current"]
            st.session_state.history[idx].append(curr.copy())
            st.session_state.images[idx]["current"] = refine_edges(curr, edge_strength)
            st.rerun()
        
        st.divider()
        st.subheader("🎨 Adjustments")
        
        brightness = st.slider("Brightness", 0.1, 2.0, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.1, 2.0, 1.0, 0.05)
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
        sharpness = st.slider("Sharpness", 0.0, 3.0, 1.0, 0.1)
        clarity = st.slider("Clarity", -1.0, 1.0, 0.0, 0.05)
        blur_radius = st.slider("Blur", 0.0, 20.0, 0.0, 0.5)
        
        if st.button("Apply Adjustments", use_container_width=True):
            idx = st.session_state.selected_idx
            curr = st.session_state.images[idx]["current"]
            st.session_state.history[idx].append(curr.copy())
            adjusted = apply_adjustments(curr, brightness, contrast, saturation, sharpness, clarity)
            adjusted = apply_blur(adjusted, blur_radius)
            st.session_state.images[idx]["current"] = adjusted
            st.rerun()
        
        st.divider()
        st.subheader("🌄 Background")
        
        bg_type = st.radio("Background Type", ["None", "Solid Color", "Gradient", "Custom Image", "Background Blur"])
        
        if bg_type == "Solid Color":
            bg_color = st.color_picker("Color", "#FFFFFF")
            if st.button("Apply Solid BG", use_container_width=True):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = apply_solid_background(curr, bg_color)
                st.rerun()
        
        elif bg_type == "Gradient":
            c1 = st.color_picker("Color 1", "#1a1a2e")
            c2 = st.color_picker("Color 2", "#16213e")
            direction = st.selectbox("Direction", ["vertical", "horizontal"])
            if st.button("Apply Gradient", use_container_width=True):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = apply_gradient_background(curr, c1, c2, direction)
                st.rerun()
        
        elif bg_type == "Custom Image":
            bg_file = st.file_uploader("Upload Background", type=["png", "jpg", "jpeg", "webp"], key="bg_upload")
            if bg_file and st.button("Apply Custom BG", use_container_width=True):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                bg_img = Image.open(bg_file)
                st.session_state.images[idx]["current"] = apply_custom_background(curr, bg_img)
                st.rerun()
        
        elif bg_type == "Background Blur":
            bg_blur = st.slider("BG Blur Radius", 1, 50, 15)
            if st.button("Apply BG Blur", use_container_width=True):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = apply_background_blur(curr, bg_blur)
                st.rerun()
        
        # Shadow
        if st.button("➕ Add Subject Shadow", use_container_width=True):
            idx = st.session_state.selected_idx
            curr = st.session_state.images[idx]["current"]
            st.session_state.history[idx].append(curr.copy())
            st.session_state.images[idx]["current"] = add_subject_shadow(curr)
            st.rerun()
        
        st.divider()
        st.subheader("✂️ Transform")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↺ Rotate -90°"):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = rotate_flip(curr, angle=-90)
                st.rerun()
            if st.button("↔ Flip Horizontal"):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = rotate_flip(curr, flip_h=True)
                st.rerun()
        with col2:
            if st.button("↻ Rotate +90°"):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = rotate_flip(curr, angle=90)
                st.rerun()
            if st.button("↕ Flip Vertical"):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = rotate_flip(curr, flip_v=True)
                st.rerun()
        
        # Simple crop (percentage based for ease)
        st.write("Crop (percent from edges)")
        crop_l = st.slider("Left %", 0, 40, 0)
        crop_t = st.slider("Top %", 0, 40, 0)
        crop_r = st.slider("Right %", 0, 40, 0)
        crop_b = st.slider("Bottom %", 0, 40, 0)
        if st.button("Apply Crop", use_container_width=True):
            idx = st.session_state.selected_idx
            curr = st.session_state.images[idx]["current"]
            w, h = curr.size
            left = int(w * crop_l / 100)
            top = int(h * crop_t / 100)
            right = w - int(w * crop_r / 100)
            bottom = h - int(h * crop_b / 100)
            if right > left and bottom > top:
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = crop_image(curr, left, top, right, bottom)
                st.rerun()
        
        st.divider()
        st.subheader("🚀 Enhance & Export")
        
        if st.button("✨ Auto Enhance", use_container_width=True):
            idx = st.session_state.selected_idx
            curr = st.session_state.images[idx]["current"]
            st.session_state.history[idx].append(curr.copy())
            st.session_state.images[idx]["current"] = auto_enhance(curr)
            st.rerun()
        
        upscale_factor = st.selectbox("HD Upscale", [1, 1.5, 2, 3, 4], index=1)
        if st.button("📈 HD Upscale", use_container_width=True):
            with st.spinner("Upscaling..."):
                idx = st.session_state.selected_idx
                curr = st.session_state.images[idx]["current"]
                st.session_state.history[idx].append(curr.copy())
                st.session_state.images[idx]["current"] = hd_upscale(curr, upscale_factor)
                st.rerun()
        
        st.divider()
        # Undo / Reset
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            if st.button("↩️ Undo", use_container_width=True):
                idx = st.session_state.selected_idx
                if st.session_state.history[idx]:
                    st.session_state.images[idx]["current"] = st.session_state.history[idx].pop()
                    st.rerun()
        with col_u2:
            if st.button("🔄 Reset", use_container_width=True):
                idx = st.session_state.selected_idx
                st.session_state.history[idx].append(st.session_state.images[idx]["current"].copy())
                st.session_state.images[idx]["current"] = st.session_state.images[idx]["original"].copy()
                st.rerun()

# ==================== Main Area ====================
if not st.session_state.images:
    st.info("👆 Sidebar se images upload karein (drag & drop supported). Multiple images supported for batch processing.")
    st.markdown("""
    ### Features included:
    - AI Background Removal (rembg + ONNX – no API key)
    - Edge / Hair refinement
    - Brightness, Contrast, Saturation, Sharpness, Clarity, Blur
    - Solid / Gradient / Custom / Blurred background
    - Subject shadow
    - Crop, Rotate, Flip
    - Auto Enhance + HD Upscale
    - PNG / JPG / WebP export with quality control
    - Batch processing + individual + ZIP download
    - Undo / Reset
    - Professional dark UI + Before/After preview
    """)
else:
    idx = st.session_state.selected_idx
    original = st.session_state.images[idx]["original"]
    current = st.session_state.images[idx]["current"]
    
    # Before / After
    col_b, col_a = st.columns(2)
    with col_b:
        st.subheader("Before")
        st.image(original, use_container_width=True)
    with col_a:
        st.subheader("After")
        st.image(current, use_container_width=True)
    
    st.divider()
    
    # Export section
    st.subheader("📥 Download")
    
    exp_col1, exp_col2, exp_col3, exp_col4 = st.columns(4)
    with exp_col1:
        export_format = st.selectbox("Format", ["PNG", "JPG", "WebP"])
    with exp_col2:
        quality = st.slider("Quality (JPG/WebP)", 10, 100, 95)
    with exp_col3:
        st.write("")  # spacer
        st.write("")
        # Individual download
        buf = get_image_bytes(current, export_format, quality)
        st.download_button(
            label=f"Download Current ({export_format})",
            data=buf,
            file_name=f"edited_{st.session_state.images[idx]['name'].rsplit('.',1)[0]}.{export_format.lower()}",
            mime=f"image/{export_format.lower()}",
            use_container_width=True
        )
    with exp_col4:
        st.write("")
        st.write("")
        # ZIP of all
        if st.button("📦 Download All as ZIP", use_container_width=True):
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for img_data in st.session_state.images:
                    img_buf = get_image_bytes(img_data["current"], export_format, quality)
                    name = f"edited_{img_data['name'].rsplit('.',1)[0]}.{export_format.lower()}"
                    zf.writestr(name, img_buf.getvalue())
            zip_buffer.seek(0)
            st.download_button(
                label="Click to save ZIP",
                data=zip_buffer,
                file_name="AI_Image_Studio_Batch.zip",
                mime="application/zip",
                key="zip_dl"
            )
    
    # Batch info
    st.caption(f"Loaded images: {len(st.session_state.images)} | Current: {st.session_state.images[idx]['name']} | Size: {current.size[0]}×{current.size[1]}")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#666;'>AI Image Studio Pro • Powered by rembg + Streamlit • Fully local, no API keys</div>",
    unsafe_allow_html=True
)