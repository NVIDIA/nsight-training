import os
import math
import traceback
import numba
import numba.cuda
import numpy as np
from PIL import Image

MAX_RGB = 256.0

#
# The following code is executed on the GPU by the numba.cuda.jit decorator.
#

@numba.cuda.jit(device=True, debug=True, opt=False)
def rgb_to_hsv(r, g, b):
    r, g, b = r / MAX_RGB, g / MAX_RGB, b / MAX_RGB
    c_max = max(r, max(g, b))
    c_min = min(r, min(g, b))
    c_diff = c_max - c_min
    if c_diff == 0:
        h = 0.0
    elif c_max == r:
        h = 60.0 * ((g - b) / c_diff)
    elif c_max == g:
        h = 60.0 * (2.0 + (b - r) / c_diff)
    else:
        h = 60.0 * (4.0 + (r - g) / c_diff)
    if h < 0:
        h += 360.0
    s = 0.0 if c_max == 0 else c_diff / c_max
    v = c_max
    return h, s, v

@numba.cuda.jit(device=True, debug=True, opt=False)
def hsv_to_rgb(h, s, v):
    h_i = math.floor(h / 60.0)
    f = h / 60.0 - h_i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    if h_i == 0 or h_i == 6:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return r * MAX_RGB, g, b

@numba.cuda.jit(debug=True, opt=False)
def sharpen_kernel(img_in, img_out, width, height):
    """Bilateral unsharp masking - uses FP64 by default (Python's float type)."""
    tx, ty = numba.cuda.grid(2)
    x, y = tx, ty
    radius = 8
    if x < radius or x >= width - radius or y < radius or y >= height - radius:
        return

    r, g, b = img_in[y, x, 0], img_in[y, x, 1], img_in[y, x, 2]
    h_orig, s_orig, v_orig = rgb_to_hsv(r, g, b)

    # Bilateral filter parameters
    sigma_s = 4.0   # Spatial sigma
    sigma_r = 0.1   # Range/intensity sigma
    sum_blur = 0.0
    weight_sum = 0.0

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            dist_sq = dx * dx + dy * dy
            if dist_sq <= radius * radius:
                nr, ng, nb = img_in[y + dy, x + dx, 0], img_in[y + dy, x + dx, 1], img_in[y + dy, x + dx, 2]
                _, _, v_neighbor = rgb_to_hsv(nr, ng, nb)
                # Bilateral filter: TWO exp() calls per neighbor
                spatial_weight = math.exp(-dist_sq / (2.0 * sigma_s * sigma_s))
                intensity_diff = v_orig - v_neighbor
                range_weight = math.exp(-intensity_diff * intensity_diff / (2.0 * sigma_r * sigma_r))
                weight = spatial_weight * range_weight
                sum_blur += v_neighbor * weight
                weight_sum += weight

    v_blur = sum_blur / weight_sum
    sharpen_amount = 1.5
    v_sharp = v_orig + (v_orig - v_blur) * sharpen_amount
    v_sharp = max(0.0, min(1.0, v_sharp))

    # Uncomment the following two lines to insert a conditional breakpoint
    # which will stop the program at the selected pixel.
    # Substitute your own `x` and `y` values here based on their
    # values in the variables panel. Since only one thread will reach
    # the breakpoint call, the debugger will stop and focus on the thread
    # handling the selected pixel.
#    if x == 8 and y == 12:
#        breakpoint()

    r_out, g_out, b_out = hsv_to_rgb(h_orig, s_orig, v_sharp)
    img_out[y, x, 0] = r_out
    img_out[y, x, 1] = g_out
    img_out[y, x, 2] = b_out

#
# The following code is executed on the CPU by the Python interpreter.
#

input_filename = "/dli/task/debugging/exercise1/images/tabby_tiger_cat_blurred.jpg"
output_filename = "/dli/task/debugging/exercise1/tabby_tiger_cat_sharpened.png"

# Remove old output image
if os.path.exists(output_filename):
    print(f"Removing old image {output_filename}")
    os.remove(output_filename)

# Load image from disk
img_pil = Image.open(input_filename)
img = np.array(img_pil, dtype=np.float64)
height, width = img.shape[:2]
print(f"Loaded image: {img.shape} {input_filename}")

block = (16, 16)
grid = (math.ceil(width / block[0]), math.ceil(height / block[1]))
print(f"Grid size:  {grid} ({grid[0] * grid[1]} blocks)")
print(f"Block size: {block} ({block[0] * block[1]} threads)")

# Create the input and output images on the GPU
d_img_in = numba.cuda.to_device(img)
d_img_out = numba.cuda.device_array_like(img)

# Launch the kernel, catching any errors that
# Numba CUDA might have raised during JIT compilation
# or kernel launch
try:
    print("Launching the Sharpen Kernel")
    sharpen_kernel[grid, block](d_img_in, d_img_out, width, height)

    # Block until the kernel finishes executing
    result = numba.cuda.synchronize()
    print(f"Sharpen Kernel finished executing with status {result}")
except Exception as e:
    print("Exception raised launching Sharpen Kernel")
    print(e)
    traceback.print_exc()
    print("No sharpened image to save, exiting program")
    exit(1)

# Get the sharpened image result
img_out = d_img_out.copy_to_host()
print(f"Sharpened image result: {img_out.shape}")

# Save the sharpened image in PNG format
img_out_u8 = np.clip(img_out, 0, 255).astype(np.uint8)
if img_out_u8.ndim == 2:
    img_out_pil = Image.fromarray(img_out_u8, mode="L")
else:
    img_out_pil = Image.fromarray(img_out_u8, mode="RGB")

print(f"Saving the sharpened image to {output_filename}")
img_out_pil.save(output_filename, format="PNG")

print("Click on the \"Refresh Images\" button in the Jupyter Notebook to view the sharpened image")
