import os
import glob
import torch
import secrets
from flask import Flask, request, render_template_string, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
from basicsr.utils import imwrite
from gfpgan import GFPGANer
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import zipfile

app = Flask(__name__)

# Generate a random secret key
app.secret_key = secrets.token_hex(16)

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
    show_download_all = False
    uploaded_text = "Images Added"  # Initialize uploaded_text here

    if request.method == "POST":
        upscale_factor = int(request.form.get("upscale_factor", 4))
        tile_size = int(request.form.get("tile_size", 400))

        gfpganer.upscale = upscale_factor

        uploaded_files = request.files.getlist("files[]")
        num_uploaded = len(uploaded_files)

        if uploaded_files:
            session['uploaded_files'] = []
            for uploaded_file in uploaded_files:
                filename = secure_filename(uploaded_file.filename)
                input_path = os.path.join(UPLOAD_FOLDER, filename)
                output_path = os.path.join(
                    OUTPUT_FOLDER, "Enhanced_" + os.path.splitext(filename)[0] + ".png")

                # Save uploaded file
                uploaded_file.save(input_path)

                # Process image with GFPGAN
                img = cv2.imread(input_path, cv2.IMREAD_COLOR)
                _, _, output = gfpganer.enhance(
                    img, has_aligned=False, only_center_face=False, paste_back=True)

                # Save output image
                imwrite(output, output_path)

                # Add image preview for restored image with download button
                restored_previews += f"""
                    <div class="image-container">
                        <img src="/output/Enhanced_{os.path.splitext(filename)[0]}.png" class="preview-image">
                        <a href="/output/Enhanced_{os.path.splitext(filename)[0]}.png" download>
                            <div class="download"><button class="custom-file-upload">Download</button></div>
                        </a>
                    </div>
                """

                # Add image preview for uploaded image from server input folder
                image_previews += f"""
                    <div class="image-container" id="uploaded-{filename}">
                        <img src="/input/{filename}" class="preview-image">
                        <div class="progress-container" id="progress-{filename}">
                            <div class="progress-bar" id="progress-bar-{filename}"></div>
                        </div>
                        <div class="remove"><button class="remove-button custom-file-upload" onclick="removeImage('{filename}')">Remove</button></div>
                    </div>
                """

                # Store the filename in the session
                session['uploaded_files'].append(output_path)

            if num_uploaded > 1:
                show_download_all = True

            uploaded_text = "Images Uploaded"

    return render_template_string(
        generate_html(image_previews, restored_previews, num_uploaded, show_download_all, uploaded_text)
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

@app.route("/reload", methods=["POST"])
def reload_ui():
    # Clear session data
    session.clear()

    # Return a response to open a new tab and close the current one
    return redirect(url_for('index'))

@app.route("/clear_history", methods=["POST"])
def clear_history():
    # Clear session data
    session.clear()

    # Remove all images from input and output directories
    for file in glob.glob(os.path.join(UPLOAD_FOLDER, "*")):
        os.remove(file)
    for file in glob.glob(os.path.join(OUTPUT_FOLDER, "*")):
        os.remove(file)

    # Return the index page with cleared session data
    return render_template_string(generate_html("", "", 0, False, "Images Added"))

@app.route("/download_all", methods=["POST"])
def download_all():
    zip_filename = "Enhanced-Images.zip"
    zip_path = os.path.join(OUTPUT_FOLDER, zip_filename)

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file in session.get('uploaded_files', []):
            zipf.write(file, os.path.basename(file))

    return send_from_directory(OUTPUT_FOLDER, zip_filename, as_attachment=True)

def generate_html(image_previews, restored_previews, num_uploaded, show_download_all, uploaded_text):
    """HTML template with dynamic image previews."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="icon" href="/favicon.ico" type="image/x-icon">
        <title>GFPGAN Image Enhancement</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400..900&display=swap');
        *{{
            font-family: "Orbitron", sans-serif;
            font-weight: 600;
            margin:0;
            padding:0;
            user-select: none;
        }}
        body {{
            text-align: center;
            height: 100vh;
            color: white;
            background: linear-gradient(45deg, white, #00ffb8, cyan, white);
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        a {{
            text-decoration: none;
        }}

        ::selection {{
            color: hotpink;
        }}

        ::-webkit-scrollbar {{
            scrollbar-width: none;
            display: none;
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
            background: #00ffb8;
            color: white;
            font-size: 16px;
            border: 1px solid white;
            margin: 10px;
            background: linear-gradient(45deg, white, #00ffb8, cyan);
        }}

        input[type="file"] {{
            display: none;
        }}

        .file-drop-area {{
            width: 800px;
            height: 200px;
            margin: 20px auto;
            padding: 50px;
            border: 2px dashed white;
            background: linear-gradient(45deg, white, #00ffb8, cyan, white);
            color: white;
            font-size: 24px;
            cursor: pointer;
            text-align: center;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 300ms ease-in;
        }}

        .file-drop-area:hover,
        .file-drop-area:active,
        .file-drop-area:focus {{
            transform: scale(1.05);
            letter-spacing: 3px;
        }}

        input[type="range"] {{
            -webkit-appearance: none;
            width: 300px;
            margin: 20px;
            outline: 1px solid white;
            height: 8px;
            background: #00ffb8;
            border-radius: 4px;
            transition: opacity 0.2s;
        }}

        input[type="range"]:hover {{
            opacity: 1;
        }}

        input[type="range"]::-webkit-slider-runnable-track {{
            height: 8px;
            background: #00ffb8;
            border-radius: 4px;
        }}

        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #00ffb8;
            box-shadow: 0px 0px 5px rgba(0, 0, 0, 0.3);
            transition: background 0.2s;
            margin-top: -6px;
        }}

        input[type="range"]::-webkit-slider-thumb:hover {{ background: #00cc96; }}
        input[type="range"]:active::-webkit-slider-thumb {{ background: #009973; }}
        input[type="range"]:focus::-webkit-slider-thumb {{
            box-shadow: 0px 0px 10px rgba(0, 255, 184, 0.5);
        }}

        button[type="submit"] {{
            background: linear-gradient(45deg, white, #00ffb8, cyan);
            padding: 10px 20px;
            border: 1px solid white;
            color: white;
            font-size: 20px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-weight: 600;
            display: block;
            margin: 10px auto;
        }}

        button[type="button"] {{
            background: linear-gradient(45deg, white, #00ffb8, cyan);
            padding: 10px 20px;
            border: 1px solid white;
            color: white;
            font-size: 20px;
            cursor: pointer;
            font-family: 'Orbitron', sans-serif;
            font-weight: 600;
            display: block;
            margin: 10px auto;
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
            height: 300px;
            width: 300px;
            overflow: hidden;
            margin: 10px;
            border: 1px solid white;
            padding: 10px 10px;
            object-fit: contain;
            transition: transform 300ms ease-in;
        }}

                .preview-image:hover,
        .preview-image:active,
        .preview-image:focus {{
            transform: scale(1.05)
        }}

        .progress-container {{
            width: 40%;
            margin: 20px auto;
            background: #00ffb8;
            border-radius: 5px;
            overflow: hidden;
            display: none;
        }}

        .progress-bar {{
            height: 20px;
            width: 100%;
            background: linear-gradient(45deg, white, #00ffb8, cyan, white);
            overflow: hidden;
            position: relative;
            padding: 10px 20px;
        }}

        .progress-fill {{
            height: 100%;
            width: 200%;
            position: absolute;
            animation: progress-animation 5s linear infinite;
            font-size: 20px;
            background-attachment: fixed;
        }}

        @keyframes progress-animation {{
            0% {{ transform: translateX(-100%); }}
            100% {{ transform: translateX(0); }}
        }}

        .uploaded-images, .restored-images {{
            display: flex;
            flex-wrap:wrap;
            flex-direction:row;
            align-items: flex-start;
            justify-content: center;
            margin: 25px 0px;
            border:1px solid white;
        }}

        .image-container {{
            display: block;
            padding: 10px;
            margin: 35px 0px;
            width: 45%;
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

        output{{
            color: white;
        }}

        .letter-spacing {{
            transition: letter-spacing 300ms ease-in;
        }}

        .letter-spacing:hover,
        .letter-spacing:active,
        .letter-spacing:focus {{
            letter-spacing: 3px;
        }}

        button{{
            transition: transform 300ms ease-in;
        }}

        button:hover,
        button:focus,
        button:active{{
            transform: scale(1.05)
        }}

        /* Responsive Design */
        @media (max-width: 768px) {{
            .container {{
                margin-top: 20px;
            }}

            h1 {{
                font-size: 30px;
            }}

            .file-drop-area {{
                height: 150px;
                padding: 20px;
                font-size: 18px;
                width: 70%;
            }}

            input[type="range"] {{
                width: 200px;
            }}

            button[type="submit"], button[type="button"] {{
                padding: 8px 16px;
                font-size: 18px;
            }}

            .preview-image {{
                height: 350px;
                width: 350px;
            }}

            .image-container {{
                width: 90%;
                margin: auto;
            }}

            .main-image-container {{
                flex-direction: column;
            }}

            .letter-spacing{{
                letter-spacing: 1px;
            }}
            .letter-spacing:hover,
            .letter-spacing.focus,
            .letter-spacing:active{{
                letter-spacing: 1px;
            }}

            .progress-container{{
                width: 70%;
            }}

            .progress-bar{{
                height: 10px;
            }}

            .text{{
                font-size: 20px;
            }}
        }}

        @media (max-width: 480px) {{
            .container {{
                margin-top: 10px;
            }}

            h1, h2, label {{
                font-size: 20px;
            }}

            .file-drop-area {{
                height: 100px;
                padding: 10px;
                font-size: 14px;
                width: 70%;
            }}

            input[type="range"] {{
                width: 150px;
            }}

            button[type="submit"], button[type="button"] {{
                padding: 6px 12px;
                font-size: 18px;
            }}

            .preview-image {{
                height: 350px;
                width: 350px;
            }}

            .image-container {{
                width: 95%;
                margin: auto;
            }}

            .text{{
                font-size:18px;
            }}

            .letter-spacing{{
                letter-spacing: 1px;
            }}
            .letter-spacing:hover,
            .letter-spacing.focus,
            .letter-spacing:active{{
                letter-spacing: 1px;
            }}

            .progress-container{{
                width: 70%;
            }}

            .progress-bar{{
                height: 10px;
            }}
        }}
    </style>
    </head>
    <body>
        <div class="container">
            <h1 class="letter-spacing">GFPGAN Image Enhancement</h1>
            <h2 id="uploaded-count" class="letter-spacing">{{num_uploaded}} {{uploaded_text}}</h2>

            <form id="upload-form" method="POST" enctype="multipart/form-data">
            <label for="upscale_factor" class="text letter-spacing">Select Upscaling Factor (1X-4X):</label>
            <input type="range" id="upscale_factor" name="upscale_factor" min="1" max="4" value="4" oninput="this.nextElementSibling.value = this.value">
            <output>4</output>
            <br>

            <label for="tile_size" class="text letter-spacing">Select Tile Size (100-400):</label>
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
                <button type="button" onclick="clearHistory()">Clear History</button>
                <button type="button" onclick="reloadUI()">Reload UI</button>
                <button type="button" onclick="downloadAll()" style="margin-top: 15px; display: {'none' if not show_download_all else 'block'};">Download All</button>
            </form>

            <!-- Progress Bar -->
            <div class="progress-container" id="progress-container">
                <div class="progress-bar" id="progress-bar"><div class="progress-fill">Enhancing...</div></div>
            </div>
        <div class="main-image-container">
        <div class="image-container">
           <!-- Preview Images -->
                <div class="title"><p class="text letter-spacing">Enhanced Images</p></div>
            <div class="restored-images">
                {restored_previews}
            </div>
         </div>

        <div class="image-container">
                <div class="title"><p class="text letter-spacing">Uploaded Images</p></div>
            <div class="uploaded-images">
                {image_previews}
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
                fileDropArea.style.borderColor = 'white';
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

                // Show preview of all images
                imagePreviewsDiv.innerHTML = '';  // Clear previous previews
                Array.from(files).forEach(file => {{
                    const previewImg = document.createElement('img');
                    previewImg.src = URL.createObjectURL(file);  // Create a preview URL
                    previewImg.className = 'preview-image';
                    imagePreviewsDiv.appendChild(previewImg);
                }});

                // Update the uploaded count
                document.getElementById('uploaded-count').innerText = files.length + ' Images Added';
            }});

            fileInput.addEventListener('change', () => {{
                const files = Array.from(fileInput.files);
                const uploadedCount = files.length;
                document.getElementById('uploaded-count').innerText = uploadedCount + ' Images Added';

                // Show preview of all images
                imagePreviewsDiv.innerHTML = '';  // Clear previous previews
                files.forEach(file => {{
                    const previewImg = document.createElement('img');
                    previewImg.src = URL.createObjectURL(file);  // Create a preview URL
                    previewImg.className = 'preview-image';
                    imagePreviewsDiv.appendChild(previewImg);
                }});
                toggleUploadButton(uploadedCount);  // Enable button if files are present
            }});

            function toggleUploadButton(count) {{
                uploadButton.disabled = count === 0;  // Disable button if no images are uploaded
            }}

            document.getElementById('upload-form').addEventListener('submit', function(e) {{
                progressContainer.style.display = 'block';  // Show the progress bar

                // Start infinite progress bar animation until image processing completes
                progressBar.style.animationPlayState = 'running';
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

            // Function to update the progress bar for each image
            function updateProgress(filename, progress) {{
                const progressBar = document.getElementById('progress-bar-' + filename);
                if (progressBar) {{
                    progressBar.style.width = progress + '%';
                }}
            }}

            // Function to clear history
            function clearHistory() {{
                fetch('/clear_history', {{
                    method: 'POST',
                }}).then(response => response.text()).then(html => {{
                    document.open();
                    document.write(html);
                    document.close();
                }});
            }}

            // Function to download all images
            function downloadAll() {{
                fetch('/download_all', {{
                    method: 'POST',
                }}).then(response => response.blob()).then(blob => {{
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = 'Enhanced-Images.zip';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                }});
            }}

            // Function to reload UI
            function reloadUI() {{
                fetch('/reload', {{
                    method: 'POST',
                }}).then(response => {{
                    if (response.ok) {{
                        window.open(response.url, '_blank');
                        window.close();
                    }}
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
