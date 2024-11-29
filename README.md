## Documentation: Running the YOLO Detection Script and API with `uv` Package Manager

This guide explains how to set up and run both the YOLO detection standalone script and the FastAPI-based YOLO object detection API using the `uv` package manager. The dependencies required for the project are pre-defined in the `uv` package manager.

---

### Prerequisites
- Python 3.8 or later installed on your system.
- `uv` package manager installed. You can install it using:
  ```bash
  pip install uv
  ```

---

### Dependencies
The following dependencies are required for both the script and API:
```plaintext
fastapi>=0.115.5
opencv-python-headless>=4.10.0.84
opencv-python>=4.10.0.84
python-multipart>=0.0.18
ultralytics>=8.3.39
uvicorn>=0.32.1
numpy>=1.26.4
```

---

### Installation Steps

#### 1. **Add Dependencies with `uv`**
Run the following command to add the dependencies:
```bash
uv add fastapi>=0.115.5 \
       opencv-python-headless>=4.10.0.84 \
       opencv-python>=4.10.0.84 \
       python-multipart>=0.0.18 \
       ultralytics>=8.3.39 \
       uvicorn>=0.32.1 \
       numpy>=1.26.4
```

This command will install the required libraries and ensure your environment is set up properly.

#### 2. **Download the Code**
Save the following two scripts in the same directory:
- **Standalone Detection Script**: Save this as `test-run.py`.
- **API Script**: Save this as `main.py`.

---

### Running the Standalone Script

The standalone script performs detection on a single image and logs the results to the console.

#### Steps:
1. Edit `test-run.py` and replace `path/to/your/image.jpg` with the path to an image file on your system.
2. Run the script:
   ```bash
   python test-run.py
   ```

#### Output:
- Detection results will be logged in the terminal in the following format:
  ```
  2024-11-29 12:00:00 - INFO - Model loaded successfully
  2024-11-29 12:00:01 - INFO - Image loaded successfully from path/to/your/image.jpg
  2024-11-29 12:00:01 - INFO - Running object detection...
  2024-11-29 12:00:02 - INFO - Detection: Class='person', Confidence=0.95, Box=(50, 75, 200, 300)
  ```

---

### Running the API

The API allows users to perform object detection by uploading images through a REST endpoint.

#### Steps:
1. Run the API script:
   ```bash
   python main.py
   ```

2. Open your browser and navigate to the Swagger UI for API documentation and testing:
   [http://127.0.0.1:8000/docs#/](http://127.0.0.1:8000/docs#/)

3. Test the `/detect` endpoint:
   - Upload an image via the file input.
   - The API will return a JSON response with detected objects, e.g.:
     ```json
     {
         "predictions": [
             {
                 "x1": 50,
                 "y1": 75,
                 "x2": 200,
                 "y2": 300,
                 "confidence": 0.95,
                 "class_id": 0,
                 "class_name": "person"
             }
         ]
     }
     ```

---

### Troubleshooting

- **Dependency Issues**:
  Ensure all dependencies are installed correctly using `uv list` to verify. Reinstall missing dependencies with:
  ```bash
  uv add <package_name>
  ```

- **Model Loading Failure**:
  Verify the YOLO model file `best.pt` exists in the current directory. Replace its path in the scripts if necessary.

- **API Not Responding**:
  Ensure the script is running and check for conflicts on port `8000`. You can change the port in the script:
  ```python
  uvicorn.run(app, host="0.0.0.0", port=8001)
  ```



This documentation ensures a seamless setup and execution process for both standalone and API-based object detection workflows.