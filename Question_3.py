import cv2
import numpy as np
import matplotlib.pyplot as plt

points      = []
img_display = None

def mouse_callback(event, x, y, flags, param):
    global points, img_display
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")
            cv2.circle(img_display, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", img_display)

        if len(points) == 4:
            print("\nFour points selected:")
            for i, p in enumerate(points):
                print(f"P{i+1}: {p}")
            print("Press any key to exit.")

turf_img = cv2.imread("data/turf.jpg")
flag_img = cv2.imread("data/srilanka_flag.png")

if turf_img is None:
    raise FileNotFoundError("turf.jpg not found in data/")
if flag_img is None:
    raise FileNotFoundError("srilanka_flag.png not found in data/")

print(f"Turf image : {turf_img.shape[1]} x {turf_img.shape[0]} px")
print(f"Flag image : {flag_img.shape[1]} x {flag_img.shape[0]} px")


img_display = turf_img.copy()
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)
cv2.imshow("Image", img_display)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(points) != 4:
    print("4 points not selected. Exiting.")
    exit()

points = np.array(points, dtype=np.float32)
print("\nFinal array of selected points:")
print(points)


h, w, _ = flag_img.shape

# 4 corners of the flag image
pts_src = np.float32([
    [0,     0    ],   # Top-Left
    [w - 1, 0    ],   # Top-Right
    [w - 1, h - 1],   # Bottom-Right
    [0,     h - 1],   # Bottom-Left
])

pts_dst = points

H_mat, status = cv2.findHomography(pts_src, pts_dst)
print("\nHomography matrix H:")
print(np.round(H_mat, 4))

warped_flag = cv2.warpPerspective(
    flag_img, H_mat,
    (turf_img.shape[1], turf_img.shape[0])
)

mask        = np.ones((h, w, 3), dtype=np.uint8) * 255
warped_mask = cv2.warpPerspective(
    mask, H_mat,
    (turf_img.shape[1], turf_img.shape[0])
)

inverse_mask = cv2.bitwise_not(warped_mask)

turf_bg  = cv2.bitwise_and(turf_img, inverse_mask)

opacity  = 0.75
turf_roi = cv2.bitwise_and(turf_img, warped_mask)
blended  = cv2.addWeighted(warped_flag, opacity, turf_roi, 1 - opacity, 0)

final_result = cv2.add(turf_bg, blended)

cv2.imshow("Superimposed Flag", final_result)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("data/Q3_result.jpg", final_result)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].imshow(cv2.cvtColor(turf_img, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Turf", fontsize=13)
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
axes[1].set_title("Sri Lanka Flag Superimposed on Pitch", fontsize=13)
axes[1].axis("off")

plt.tight_layout()
plt.savefig("data/Q3_comparison.png", dpi=150)
plt.show()