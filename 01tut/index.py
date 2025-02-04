import pytesseract
import cv2
import numpy as np
import os


image = cv2.imread(r"C:\Users\abrar\Desktop\new\Abrar_OCR\01tut\index.PNG")
base_image = image.copy()

# Convertint  to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(r"C:\Users\abrar\Desktop\new\Abrar_OCR\01tut\gray_image.jpg",gray)

# Applying  threshold
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
cv2.imwrite(r"C:\Users\abrar\Desktop\new\Abrar_OCR\01tut\thresh_image.jpg",thresh)

# Create a kernel for dilation
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 2))

# Dilate to connect text components
dilate = cv2.dilate(thresh, kernel, iterations=1)
cv2.imwrite(r"C:\Users\abrar\Desktop\new\Abrar_OCR\01tut\dilate_image.jpg",dilate)

# Finding contours and sort them from top to bottom
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# Sorting  contours by y-coordinates
def get_y_coord(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return y

cnts = sorted(cnts, key=get_y_coord)

# Calculating average line height

heights = [cv2.boundingRect(cnt)[3] for cnt in cnts if cv2.boundingRect(cnt)[3] > 5]
avg_height = np.mean(heights) if heights else 0

results = []
prev_y = 0
min_line_gap = avg_height * 0.5
line_count = 1  # Counter for naming the images

# Get the directory path
output_dir = r"C:\Users\abrar\Desktop\new\Abrar_OCR\01tut\lines"

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    
    # Filtering based on size and distance from previous line
    if h > 5 and w > 40 and (y - prev_y > min_line_gap or prev_y == 0):
        # Add small padding
        padding = 2
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(w + 2*padding, image.shape[1] - x)
        h = min(h + 2*padding, image.shape[0] - y)
        
        # Main page rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2) 
        
        # Extract line image from base image
        line_image = base_image[y:y + h, x:x + w]
        
        # Save individual line image
        line_filename = os.path.join(output_dir, f'line{line_count}.png')
        cv2.imwrite(line_filename, line_image)
        
        # OCR
        text = pytesseract.image_to_string(line_image, config='--psm 7').strip()
        if text:
            results.append(text)
        
        prev_y = y + h
        line_count += 1

# Saving the main image with all boxes
cv2.imwrite(r"C:\Users\abrar\Desktop\new\Abrar_OCR\01tut\bbox\index_bobx.PNG", image)

# Printing results on console 
for idx, text in enumerate(results, 1):
    print(f"Line {idx}: {text}")

print(f"\nTotal no. of Lines saved are : {line_count-1}")