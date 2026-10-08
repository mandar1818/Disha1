import urllib.request
import json
import cv2
import numpy as np

# Create synthetic test frame
img = np.zeros((480, 640, 3), dtype=np.uint8)
img[100:200, 200:400] = 200
_, img_encoded = cv2.imencode('.jpg', img)

boundary = '----SmartVisionPhase4Boundary'
body = bytearray()
body.extend(f'--{boundary}\r\n'.encode('utf-8'))
body.extend(b'Content-Disposition: form-data; name="file"; filename="frame.jpg"\r\n')
body.extend(b'Content-Type: image/jpeg\r\n\r\n')
body.extend(img_encoded.tobytes())
body.extend(f'\r\n--{boundary}--\r\n'.encode('utf-8'))

# 1. Start destination navigation for Room 302
start_payload = json.dumps({"destination": "Room 302"}).encode("utf-8")
start_req = urllib.request.Request(
    "http://127.0.0.1:8000/navigation/start",
    data=start_payload,
    headers={"Content-Type": "application/json"},
)
start_res = json.loads(urllib.request.urlopen(start_req).read())
print("1. Navigation Start:", start_res.get("message"))
print("   Session State:", start_res.get("session", {}).get("state"))

# 2. Process frame via /detect
detect_req = urllib.request.Request(
    "http://127.0.0.1:8000/detect",
    data=bytes(body),
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
)
detect_res = json.loads(urllib.request.urlopen(detect_req).read())
print("\n2. Detect Endpoint:")
print("   Success:", detect_res.get("success"))
print("   Safety Level:", detect_res.get("safety_level"))
print("   Navigation Session Present:", "navigation_session" in detect_res)
print("   Navigation Session State:", detect_res.get("navigation_session", {}).get("state"))
print("   Navigation Breakdown MS:", detect_res.get("processing_breakdown_ms", {}).get("navigation"))

# 3. Stop destination navigation
stop_req = urllib.request.Request(
    "http://127.0.0.1:8000/navigation/stop",
    data=b"",
    headers={"Content-Type": "application/json"},
)
stop_res = json.loads(urllib.request.urlopen(stop_req).read())
print("\n3. Navigation Stop:", stop_res.get("message"))
print("   Session State:", stop_res.get("session", {}).get("state"))

