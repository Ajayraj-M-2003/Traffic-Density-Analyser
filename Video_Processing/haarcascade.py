import cv2

# Load pre-trained Haar cascade classifier for car detection
car_cascade = cv2.CascadeClassifier('cars.xml')

# Load the image
image_path = 'car.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform car detection
cars = car_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Draw rectangles around the detected cars
for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Display the result
cv2.imshow('Car Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
