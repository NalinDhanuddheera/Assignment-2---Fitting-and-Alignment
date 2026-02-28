import cv2
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Camera Parameters
f_mm           = 8.0
Z_mm           = 720.0
pixel_pitch_mm = 0.0022
mm_per_pixel   = (pixel_pitch_mm * Z_mm) / f_mm

img = cv2.imread("data/earrings.jpg")
if img is None:
    raise FileNotFoundError("earrings.jpg not found in data/")

img_annotated = img.copy()
gray          = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Threshold and Find Contours
_, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

boxes = {}
for i, contour in enumerate(contours):
    boxes[i] = cv2.boundingRect(contour)

# Measure ONE Outer and ONE Inner Earring
measured_outer = False
measured_inner = False
outer_data     = {}
inner_data     = {}

for i, contour in enumerate(contours):
    if cv2.contourArea(contour) < 5000:
        continue

    parent_idx = hierarchy[0][i][3]
    is_outer   = (parent_idx == -1)
    box_color  = (255, 0, 0) if is_outer else (0, 0, 255)

    bx, by, bw, bh = boxes[i]

    # Align inner top with outer top
    if not is_outer and parent_idx in boxes:
        p_bx, p_by, p_bw, p_bh = boxes[parent_idx]
        original_bottom = by + bh
        by = p_by
        bh = original_bottom - by

    if is_outer and not measured_outer:
        outer_data = {
            'bw': bw, 'bh': bh,
            'real_w': bw * mm_per_pixel,
            'real_h': bh * mm_per_pixel
        }
        measured_outer = True

    elif not is_outer and not measured_inner:
        inner_data = {
            'bw': bw, 'bh': bh,
            'real_w': bw * mm_per_pixel,
            'real_h': bh * mm_per_pixel
        }
        measured_inner = True

    # Draw bounding box 
    cv2.rectangle(img_annotated,(bx, by), (bx + bw, by + bh),box_color, 2)

    # Horizontal width line
    y_center = by + bh // 2
    cv2.line(img_annotated,(bx, y_center), (bx + bw, y_center),box_color, 1)
    cv2.circle(img_annotated, (bx,      y_center), 4, box_color, -1)
    cv2.circle(img_annotated, (bx + bw, y_center), 4, box_color, -1)

    # Vertical height line
    x_center = bx + bw // 2
    cv2.line(img_annotated,(x_center, by), (x_center, by + bh),box_color, 1)
    cv2.circle(img_annotated, (x_center, by),      4, box_color, -1)
    cv2.circle(img_annotated, (x_center, by + bh), 4, box_color, -1)

print("EARRING MEASUREMENTS (One Earring)")
print(f"\nOUTER:")
print(f"  Width  : {outer_data['bw']} px = {outer_data['real_w']:.2f} mm")
print(f"  Height : {outer_data['bh']} px = {outer_data['real_h']:.2f} mm")
print(f"\nINNER:")
print(f"  Width  : {inner_data['bw']} px = {inner_data['real_w']:.2f} mm")
print(f"  Height : {inner_data['bh']} px = {inner_data['real_h']:.2f} mm")

img_annotated_rgb = cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB)

height_px, width_px, _ = img_rgb.shape
total_width_mm  = width_px  * mm_per_pixel
total_height_mm = height_px * mm_per_pixel

fig, (ax_img, ax_text) = plt.subplots(
    nrows=1, ncols=2,figsize=(14, 7),gridspec_kw={'width_ratios': [3, 1]})

# LEFT — Image with mm grid 
ax_img.imshow(img_annotated_rgb,extent=[0, total_width_mm, total_height_mm, 0])

ax_img.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax_img.yaxis.set_major_locator(ticker.MultipleLocator(10))
ax_img.xaxis.set_minor_locator(ticker.MultipleLocator(2))
ax_img.yaxis.set_minor_locator(ticker.MultipleLocator(2))

ax_img.grid(which='major', color='black',
            linestyle='-',  linewidth=1.2, alpha=0.8)
ax_img.grid(which='minor', color='black',
            linestyle=':',  linewidth=0.7, alpha=0.5)

ax_img.set_xlabel("Width (mm)",  fontsize=12, fontweight='bold')
ax_img.set_ylabel("Height (mm)", fontsize=12, fontweight='bold')
ax_img.set_title("Earring Dimensions with Millimeter Grid",fontsize=13, fontweight='bold')

#RIGHT — Results as Text 
ax_text.axis('off')

result_text = (
    f"CAMERA SETUP\n"
    f"{'─' * 28}\n"
    f"Focal length : {f_mm} mm\n"
    f"Object dist  : {Z_mm} mm\n"
    f"Pixel size   : {pixel_pitch_mm} mm\n"
    f"mm / pixel   : {mm_per_pixel:.4f}\n\n"
    f"OUTER EARRING\n"
    f"{'─' * 28}\n"
    f"Width  : {outer_data['bw']} px\n"
    f"       = {outer_data['real_w']:.2f} mm\n\n"
    f"Height : {outer_data['bh']} px\n"
    f"       = {outer_data['real_h']:.2f} mm\n\n"
    f"INNER HOLE\n"
    f"{'─' * 28}\n"
    f"Width  : {inner_data['bw']} px\n"
    f"       = {inner_data['real_w']:.2f} mm\n\n"
    f"Height : {inner_data['bh']} px\n"
    f"       = {inner_data['real_h']:.2f} mm"
)

ax_text.text(
    0.05, 0.97,
    result_text,
    transform         = ax_text.transAxes,
    fontsize          = 10,
    verticalalignment = 'top',
    fontfamily        = 'monospace',
    bbox              = dict(
        boxstyle  = 'round',
        facecolor = '#f0f4ff',
        edgecolor = '#1a1a2e',
        linewidth = 1.5,
        alpha     = 0.95
    )
)
ax_text.set_title("Results", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig("data/Q2_result.png", dpi=300)
plt.show()