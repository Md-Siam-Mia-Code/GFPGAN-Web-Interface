import os
import glob
import torch
from flask import Flask, request, render_template_string, send_from_directory
from werkzeug.utils import secure_filename
import cv2
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

app = Flask(__name__)

# Create directories for input and output
UPLOAD_FOLDER = "Input"
OUTPUT_FOLDER = "Output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize GFPGAN with Real-ESRGAN for upscaling
gfpganer = GFPGANer(
    model_path="experiments/pretrained_models/GFPGANv1.4.pth",  # Path to GFPGANv1.4 model
    upscale=2,  # Default upscale, can be adjusted via UI
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None,  # Will be updated later with Real-ESRGAN
)

# Initialize Real-ESRGAN for background upscaling
bg_model = RRDBNet(
    num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2
)
bg_upsampler = RealESRGANer(
    scale=2,
    model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    model=bg_model,
    tile=400,  # Tile size for large images
    tile_pad=10,
    pre_pad=0,
    half=False,  # Disable half precision for CPU
)
gfpganer.bg_upsampler = bg_upsampler


@app.route("/", methods=["GET", "POST"])
def index():
    image_previews = ""
    restored_previews = ""
    num_uploaded = 0

    if request.method == "POST":
        upscale_factor = int(request.form.get("upscale_factor", 2))
        tile_size = int(request.form.get("tile_size", 400))

        gfpganer.upscale = upscale_factor

        uploaded_files = request.files.getlist("files[]")
        num_uploaded = len(uploaded_files)

        if uploaded_files:
            for uploaded_file in uploaded_files:
                filename = secure_filename(uploaded_file.filename)
                input_path = os.path.join(UPLOAD_FOLDER, filename)
                output_path = os.path.join(
                    OUTPUT_FOLDER, "restored_" + filename)

                # Save uploaded file
                uploaded_file.save(input_path)

                # Process image with GFPGAN
                img = cv2.imread(input_path, cv2.IMREAD_COLOR)
                _, _, output = gfpganer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True)

                # Save output image
                imwrite(output, output_path)

                # Add image preview for uploaded image from server input folder
                image_previews += f"""
                    <div class="image-container" id="uploaded-{filename}">
                        <img src="/input/{filename}" class="preview-image">
                            <div class="remove"><button class="remove-button custom-file-upload" onclick="removeImage('{filename}')">Remove</button></div>
                    </div>
                """

                # Add image preview for restored image with download button
                restored_previews += f"""
                    <div class="image-container">
                        <img src="/output/restored_{filename}" class="preview-image">
                        <a href="/output/restored_{filename}" download>
                            <div class="download"><button class="custom-file-upload">Download</button></div>
                        </a>
                    </div>
                """

    return render_template_string(
        generate_html(image_previews, restored_previews, num_uploaded)
    )


@app.route("/remove", methods=["POST"])
def remove_image():
    remove_file = request.form.get("remove_file")
    if remove_file:
        try:
            # Remove uploaded file only
            os.remove(os.path.join(UPLOAD_FOLDER, remove_file))
        except FileNotFoundError:
            pass  # If the file doesn't exist, just pass
    return index()  # Redirect back to the index page after removal


def generate_html(image_previews, restored_previews, num_uploaded):
    """HTML template with dynamic image previews."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GFPGAN Image Enhancement</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap');
*{{
            font-family: "Orbitron", sans-serif;
            font-weight: 600;
            margin:0;
            padding:0;
}}
        body {{
            text-align: center;
            height: 100vh;
            color: white;
            background: black;
        }}

        ::selection {{
            color: white;
        }}

        .container {{
            margin-top: 50px;
        }}

        h1 {{
            font-size: 50px;
        }}

        .custom-file-upload {{
            display: block;
            width:130px;
            text-align:center;
            padding: 10px 20px;
            cursor: pointer;
            background: rgb(0, 231, 231);
            color: white;
            font-size: 16px;
            border: 1px solid white;
            margin: 10px;
            background: linear-gradient(to right, white, rgb(255, 0, 128), cyan);
        }}

        input[type="file"] {{
            display: none;
        }}

        .file-drop-area {{
            width: 100%;
            max-width: 400px;
            margin: 20px auto;
            padding: 50px;
            border: 2px dashed #ffffff;
            background: linear-gradient(to right, white, rgb(255, 0, 128), cyan);
            color: white;
            font-size: 18px;
            text-align: center;
            cursor: pointer;
        }}

        input[type="range"] {{
            width: 300px;
            margin: 20px;
            accent-color: rgb(255, 0, 128);
        }}

        button[type="submit"] {{
            background: linear-gradient(to right, white, rgb(255, 0, 128), cyan);
            padding: 10px 20px;
            border: 1px solid white;
            color: white;
            font-size: 20px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-weight: 600;
        }}

        button[disabled] {{
            background: grey;
            cursor: not-allowed;
        }}

        .slider-label {{
            font-size: 25px;
            font-weight: bold;
            margin-bottom: 10px;
        }}

        .preview-image {{
            width: 300px;
            margin: 10px;
        }}

        .progress-container {{
            width: 100%;
            max-width: 400px;
            margin: 20px auto;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            overflow: hidden;
            display: none;  /* Hidden by default */
        }}

        .progress-bar {{
            height: 20px;
            width: 0;
            background-color: hotpink;
            transition: width 0.2s;
        }}

        .uploaded-images, .restored-images {{
            display: flex;
            flex-wrap:wrap;
            align-items: flex-start;
            justify-content: center;
            margin: 25px 0px;
            border:1px solid rgb(255,0,128);
            min-height:1000px;
        }}
.image-container {{
    display: block;
    # align-items: center;
    # justify-content: center;
    # flex-wrap:wrap;
    padding: 10px;
    margin: 35px 0px;
    width: 50%;
    height:650px;
}}
        .gradient-text{{
        color: transparent;
        background-clip: text;
        background-image: linear-gradient(to right, white, rgb(255, 0, 128), cyan);
    }}
    .main-image-container{{
        display:flex;
        justify-content:center;
    }}
    .text{{
        font-size:25px;
    }}
    .download, .remove{{
        width:100%;
display: flex;
    align-items: center;
    justify-content: center;
        margin:auto;
        text-align:center;
    }}
    </style>
    </head>
    <body>
        <div class="container">
            <h1 class="gradient-text">GFPGAN Image Enhancement</h1>
            <h2 id="uploaded-count" class="gradient-text">{num_uploaded} Images Uploaded</h2>

            <form id="upload-form" method="POST" enctype="multipart/form-data">
<label for="upscale_factor" class="gradient-text text">Select Upscaling Factor (1X-4X):</label>
<input type="range" id="upscale_factor" name="upscale_factor" min="1" max="4" value="4" oninput="this.nextElementSibling.value = this.value">
<output>4</output>
<br>

<label for="tile_size" class="gradient-text text">Select Tile Size (100-400):</label>
<input type="range" id="tile_size" name="tile_size" min="100" max="400" value="400" oninput="this.nextElementSibling.value = this.value">
<output>400</output>
<br>

                <!-- File Drop Area -->
                <div class="file-drop-area" id="file-drop-area">
                    Drag and drop images here or click to select
                </div>

                <!-- Hidden File Input -->
                <input type="file" id="file-input" name="files[]" multiple accept="image/*">
                <br>
                <button type="submit" disabled>Upload and Enhance</button>
            </form>

            <!-- Progress Bar -->
            <div class="progress-container" id="progress-container">
                <div class="progress-bar" id="progress-bar">Processing...</div>
            </div>
<div class="main-image-container">
<div class="image-container">
           <!-- Preview Images -->
                <div class="title"><p class="gradient-text text">Uploaded Images</p></div>
            <div class="uploaded-images">
                {image_previews}
            </div>
 </div>
<div class="image-container">
                <div class="title"><p class="gradient-text text">Enhanced Images</p></div>
            <div class="restored-images">
                {restored_previews}
            </div>
</div>
</div>

        </div>

        <script>
            const fileDropArea = document.getElementById('file-drop-area');
            const fileInput = document.getElementById('file-input');
            const progressBar = document.getElementById('progress-bar');
            const progressContainer = document.getElementById('progress-container');
            const imagePreviewsDiv = document.querySelector('.uploaded-images');
            const restoredPreviewsDiv = document.querySelector('.restored-images');
            const uploadButton = document.querySelector('button[type="submit"]');

            fileDropArea.addEventListener('click', () => fileInput.click());

            fileDropArea.addEventListener('dragover', (e) => {{
                e.preventDefault();
                fileDropArea.style.borderColor = 'cyan';
            }});

            fileDropArea.addEventListener('dragleave', () => {{
                fileDropArea.style.borderColor = 'white';
            }});

            fileDropArea.addEventListener('drop', (e) => {{
                e.preventDefault();
                fileDropArea.style.borderColor = 'white';
                const files = e.dataTransfer.files;
                fileInput.files = files;  // Assign the dropped files to the input
                toggleUploadButton(files.length);  // Enable button if files are present
            }});

            fileInput.addEventListener('change', () => {{
                const files = Array.from(fileInput.files);
                const uploadedCount = files.length;
                document.getElementById('uploaded-count').innerText = uploadedCount + ' images uploaded';

                // Show preview of the first image
                const previewImg = document.createElement('img');
                previewImg.src = URL.createObjectURL(files[0]);  // Create a preview URL
                previewImg.className = 'preview-image';
                imagePreviewsDiv.innerHTML = '';  // Clear previous previews
                imagePreviewsDiv.appendChild(previewImg);
                toggleUploadButton(uploadedCount);  // Enable button if files are present
            }});

            function toggleUploadButton(count) {{
                uploadButton.disabled = count === 0;  // Disable button if no images are uploaded
            }}

            document.getElementById('upload-form').addEventListener('submit', function(e) {{
                progressContainer.style.display = 'block';  // Show the progress bar
                let interval = setInterval(() => {{
                    // Simulate progress bar filling
                    const currentWidth = parseInt(getComputedStyle(progressBar).width);
                    const maxWidth = parseInt(getComputedStyle(progressContainer).width);
                    if (currentWidth < maxWidth) {{
                        progressBar.style.width = (currentWidth + 10) + 'px';  // Adjust this for actual progress
                    }} else {{
                        clearInterval(interval);
                    }}
                }}, 100);  // Update progress every 100 ms
            }});
            function removeImage(filename) {{
    const container = document.getElementById('uploaded-' + filename);
    if (container) {{
        container.remove();  // Remove the image preview from the DOM
    }}

    // Optionally, you can make an AJAX request to the server to remove the file
    fetch('/remove', {{
        method: 'POST',
        headers: {{
            'Content-Type': 'application/json',
        }},
        body: JSON.stringify({{
            remove_file: filename
        }})
    }});
}}

        </script>
    </body>
    </html>
    """


@app.route("/output/<filename>")
def output(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


@app.route("/input/<filename>")
def input_images(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
