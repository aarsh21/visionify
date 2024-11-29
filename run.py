from ultralytics import YOLO

model = YOLO('best.pt')  

# Load an image
image_path = 'Untitled.jpg'

results = model(image_path)

print(results)

