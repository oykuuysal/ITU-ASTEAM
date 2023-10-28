import cv2
import numpy as np

# Resmi yükle
image = cv2.imread('img.jpeg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Kırmızı renk aralığı
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)

# Yeşil renk aralığı
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Siyah renk aralığı
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])
mask_black = cv2.inRange(hsv, lower_black, upper_black)

# Renk maskeleme işlemleri
mask = mask_red + mask_green + mask_black

# Morfolojik işlemleri uygula (gürültüyü azaltmak için)
kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Belirlenen renk aralığındaki nesneleri bul
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Her bir renkteki nesneleri birer dikdörtgen içine al
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Sonucu göster
cv2.imshow('Renkli Duba Tespiti', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
