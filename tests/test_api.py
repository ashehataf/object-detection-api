
import requests

files = {
    "image": ("sample.jpg", open("sample.jpg", "rb"), "image/jpeg")
}
data = {"image_id": "test-image"}

response = requests.post("http://localhost:8011/predict", files=files, data=data)
print(response.json())
