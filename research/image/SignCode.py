import cv2
import numpy as np
import logging
import datetime
import math
import sys
import os
from sklearn.linear_model import RANSACRegressor

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
past_steering_angle = 0
row_threshold = 0
path = "/home/pi/CapstoneProjectBackUp/research/image/Data"
image_import_path = "/home/pi/CapstoneProjectBackUp/research/image/Data/raw_imports"
crop_height = int(SCREEN_HEIGHT * 0.10)  # This will be 48 pixels
ifblue = False
use_live_camera = True  # Set this to False to load image from file
image_center = 0 

# Initialize camera
camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))  # Set YUYV format

def getTime():
    return datetime.datetime.now().strftime("S_%S_M_%M")

def stabilize_steering_angle(curr_steering_angle, last_steering_angle=None, alpha=0.2):
    if last_steering_angle is None:
        return int(curr_steering_angle)
    else:
        if 135 - last_steering_angle <= 5 and curr_steering_angle >= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),
                           last_steering_angle-1, last_steering_angle+1)
        elif last_steering_angle - 55 <= 5 and curr_steering_angle <= last_steering_angle:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),
                           last_steering_angle-1, last_steering_angle+1)
        else:
            return np.clip(int(alpha * curr_steering_angle + (1.-alpha) * last_steering_angle),
                           last_steering_angle-3, last_steering_angle+3)

# Function to estimate mid_star from one lane using a fixed lane width assumption
def estimate_mid_star_from_one_lane(lane_points, lane_side='right', lane_width=60):
    if len(lane_points) > 0:
        avg_x = np.mean(lane_points[:, 0])
        if lane_side == 'right':
            return avg_x - (lane_width / 2.0)
        else:
            return avg_x + (lane_width / 2.0)
    else:
        return 159

times2Run = {1}

for i in times2Run:
    if use_live_camera:
        # Capture a live image from the camera
        camera.read()  # Discard the first frame
        successfulRead, raw_image = camera.read()
        print("Live Image Taken.")
        if not successfulRead:
            print("Image not taken successfully.")
            break
    else:
        # Load image from file instead of capturing from camera
        image_file = os.path.join(image_import_path, "raw_image_S_02_M_53.jpg")
        if not os.path.exists(image_file):
            print(f"Image file {image_file} does not exist.")
            break
        raw_image = cv2.imread(image_file)
        if raw_image is None:
            print("Failed to load image from file.")
            break
        print("Imported Image for Testing, no live data.")

    cv2.imwrite(os.path.join(path, f"raw_image_{getTime()}.jpg"), raw_image)

    if raw_image.shape[1] != SCREEN_WIDTH or raw_image.shape[0] != SCREEN_HEIGHT:
        print(f"Warning: Image dimensions mismatch. Expected: {SCREEN_WIDTH}x{SCREEN_HEIGHT}, Got: {raw_image.shape[1]}x{raw_image.shape[0]}")

    raw_image = cv2.flip(raw_image, -1)
    cv2.imwrite(os.path.join(path, f"flipped_image_raw_{getTime()}.jpg"), raw_image)

    print('Img to color...')
    img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    print('Cropping top portion of the image...')
    img_bottom_half_bgr = raw_image[crop_height:,:]

    print('Performing HSV color space transformation...')
    img_hsv = cv2.cvtColor(img_bottom_half_bgr, cv2.COLOR_BGR2HSV)
    img_crop_hsv = img_hsv

    print('Creating binary mask...')
    if ifblue:
        lower_hsv = np.array([100, 150, 50])
        upper_hsv = np.array([130, 255, 255])
    else:
        lower_hsv = np.array([0, 0, 120])
        upper_hsv = np.array([180, 50, 255])

    mask = cv2.inRange(img_crop_hsv, lower_hsv, upper_hsv)

    print('Applying Gaussian blur on mask...')
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    print('Applying Canny filter...')
    mask_edges = cv2.Canny(mask_blurred, 50, 150)

    # Optional: Apply morphological operations to enhance lane lines
    print('Applying morphological operations to enhance lane lines...')
    kernel = np.ones((5, 5), np.uint8)
    mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_CLOSE, kernel)
    mask_edges = cv2.morphologyEx(mask_edges, cv2.MORPH_OPEN, kernel)

    # Optional: Define and apply Region of Interest (ROI)
    # Uncomment and adjust the ROI polygon as needed
    """
    print('Applying Region of Interest (ROI)...')
    mask_roi = np.zeros_like(mask_edges)
    polygon = np.array([[
        (0, mask_edges.shape[0]),
        (SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2),
        (SCREEN_WIDTH // 2 + 50, SCREEN_HEIGHT // 2),
        (SCREEN_WIDTH, mask_edges.shape[0])
    ]], np.int32)
    cv2.fillPoly(mask_roi, polygon, 255)
    mask_edges = cv2.bitwise_and(mask_edges, mask_roi)
    """

    crop_width = 20
    mask_edges = mask_edges[:, crop_width:]
    adjusted_screen_width = SCREEN_WIDTH - crop_width
    print(f"New width after cropping: {adjusted_screen_width}")

    cv2.imwrite(os.path.join(path, f"cropped_mask_edges_{getTime()}.jpg"), mask_edges)

    # Update Hough Transform parameters
    minLineLength = 60  # Increased from 40 to further filter out shorter lines like street signs
    maxLineGap = 25     # Slightly increased from 20 to maintain continuity
    min_threshold = 50  # Increased from 50 to require more votes for line detection

    print('Applying Probabilistic Hough Transform...')
    lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, min_threshold, minLineLength, maxLineGap)

    if lines is not None:
        hough_debug_img = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)
        print("Detected lines (in mask_edges coords):")
        # Initialize lists to hold filtered lines
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Calculate the angle of the line relative to the vertical axis
            angle = math.degrees(math.atan2((x2 - x1), (y2 - y1)))
            # Filter lines based on angle (e.g., only near vertical)
            if -25 < angle < 25:
                filtered_lines.append(line)
                cv2.line(hough_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                print(f"Accepted Line: ({x1},{y1}) -> ({x2},{y2}) with angle {angle:.2f} degrees")
            else:
                print(f"Rejected Line: ({x1},{y1}) -> ({x2},{y2}) with angle {angle:.2f} degrees")

        print(f"Total detected lines: {len(lines)}, Filtered lines: {len(filtered_lines)}")
    else:
        hough_debug_img = None
        print("No lines detected.")

    print("Saving Images without calculating angle.")
    cv2.imwrite(os.path.join(path, f"img_rgb_{getTime()}.jpg"), img_rgb)
    cv2.imwrite(os.path.join(path, f"img_bottom_half_bgr_{getTime()}.jpg"), img_bottom_half_bgr)
    cv2.imwrite(os.path.join(path, f"img_crop_hsv_{getTime()}.jpg"), img_crop_hsv)
    cv2.imwrite(os.path.join(path, f"mask_{getTime()}.jpg"), mask)
    cv2.imwrite(os.path.join(path, f"mask_blurred_{getTime()}.jpg"), mask_blurred)
    cv2.imwrite(os.path.join(path, f"mask_edges_{getTime()}.jpg"), mask_edges)
    if hough_debug_img is not None:
        cv2.imwrite(os.path.join(path, f"hough_lines_{getTime()}.jpg"), hough_debug_img)

    # Lower threshold more to get patches
    threshold = 12
    print(f"Using threshold={threshold} for lane detection.")
    col_sum = np.sum(mask_edges > 0, axis=0)
    lane_columns = np.where(col_sum > threshold)[0]
    print(f"lane_columns: {lane_columns}")

    if len(lane_columns) == 0:
        list_patch = []
    else:
        segments = []
        start = lane_columns[0]
        prev = lane_columns[0]
        image_center = adjusted_screen_width // 2
        print(f"Start: {start}, Prev: {prev}")

        for c in lane_columns[1:]:
            print(f"Current column: {c}, Previous column: {prev}")  # Debugging current and previous columns
            if c != prev + 1:
                print(f"Non-continuous segment detected. Appending segment: ({start}, {prev})")  # Debug when a segment is finalized
                segments.append((start, prev))
                start = c
                print(f"New start set to: {start}")  # Debug the new start
            prev = c

        segments.append((start, prev))
        print(f"Final segment appended: ({start}, {prev})")

        num_patches_horizontal = 6
        num_patches_vertical = 4 
        patch_height = (SCREEN_HEIGHT - crop_height) // num_patches_vertical
        print(f'Patch Height: {patch_height}')
        patch_width = adjusted_screen_width // num_patches_horizontal
        list_patch = []
        
        for i in range(num_patches_horizontal):
            x0 = i * patch_width
            x1 = min((i + 1) * patch_width, adjusted_screen_width - 1)
            for k in range(num_patches_vertical):
                y0 = k * patch_height
                y1 = (k + 1) * patch_height - 1
                list_patch.append({'x': (x0, x1), 'y': (y0, y1)})
                print(f"Patch: x=({x0},{x1}), y=({y0},{y1})")

    print("Testing if lines is not None")
    if lines is not None:
        print("lines is not None")
        for idx, patch in enumerate(list_patch):
            x0, x1 = patch['x']
            y0, y1 = patch['y']
            print(f"Patch {idx}: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
            print(f'Rectangle First Coordinate: {(x0 + crop_width, y0)}, Second: {(x1 + crop_width, y1)}')
            cv2.rectangle(img_bottom_half_bgr, (x0 + crop_width, y0), (x1 + crop_width, y1), (50,255, 0), 1)
            if hough_debug_img is not None:
                cv2.rectangle(hough_debug_img, (x0 + crop_width, y0), (x1 + crop_width, y1), (50,255, 0), 1)
    else:
        print("lines is NONE")

    print("Saving Image With Lines (Dynamic Patches).")
    if lines is not None and hough_debug_img is not None:
        cv2.imwrite(os.path.join(path, f"image_lines_bottom_half_raw{getTime()}.jpg"), img_bottom_half_bgr)
        cv2.imwrite(os.path.join(path, f"image_lines_masked_edges{getTime()}.jpg"), hough_debug_img)

    if lines is None:
        print("No Lines Detected. Exiting Loop")
        break
    else:
        centroid_debug_image = hough_debug_img.copy()
        patch_centroids_data = []

        print("Calculating centroids...")
        for patch_info in list_patch:
            px_start, px_end = patch_info['x']
            py_start, py_end = patch_info['y']

            inside_points = []
            for detected_line in filtered_lines:  # Use filtered_lines instead of lines
                lx1, ly1, lx2, ly2 = detected_line[0]
                # Check if both endpoints are within the patch
                if (px_start <= lx1 <= px_end and py_start <= ly1 <= py_end and
                    px_start <= lx2 <= px_end and py_start <= ly2 <= py_end):
                    inside_points.append([lx1, ly1])
                    inside_points.append([lx2, ly2])
                    # Optionally, visualize these points
                    cv2.circle(centroid_debug_image, (lx1 + crop_width, ly1), 2, (255,0,0), -1)
                    cv2.circle(centroid_debug_image, (lx2 + crop_width, ly2), 2, (255,0,0), -1)

            if len(inside_points) > 0:
                inside_points = np.array(inside_points)
                centroid_coords = np.mean(inside_points, axis=0).astype(int)
                print(f'Centroid_Coords: {centroid_coords}')
                patch_centroids_data.append({'patch': patch_info, 'centroid': (centroid_coords[0], centroid_coords[1])})
                cv2.circle(centroid_debug_image, (centroid_coords[0] + crop_width, centroid_coords[1]), 5, (0,165,255), -1)
                print(f"Centroid: ({centroid_coords[0]},{centroid_coords[1]})")

        cv2.imwrite(os.path.join(path, f"centroids_visualized_{getTime()}.jpg"), centroid_debug_image)
        print("Centroids computed and visualized on debug image.")

        X_left = []
        X_right = []
        # Removed unused variables: n_right_side_right_dir, etc.

        print(f"Using image_center={image_center} to divide left/right lanes.")
        cv2.line(hough_debug_img, (image_center + crop_width, 0),
         (image_center + crop_width, hough_debug_img.shape[0]),
         (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(path, f"hough_image_center_line{getTime()}.jpg"), hough_debug_img)


        print("Separating centroids into left and right sets for polynomial interpolation...")

        numTimes = 0 

        for data_item in patch_centroids_data:
            numTimes += 1
            print(f'Number of times Separating Centroids Ran: {numTimes}')
            cx, cy = data_item['centroid']
            print(f"Centroid found at x={cx}, y={cy}")
            if cx < image_center:
                X_left.append([cx, cy])
                print(f"Added centroid ({cx}, {cy}) to X_left. Current X_left set: {X_left}")
            else:
                X_right.append([cx, cy])
                print(f"Added centroid ({cx}, {cy}) to X_right. Current X_right set: {X_right}")

        X_left = np.array(X_left) if len(X_left) > 0 else np.zeros((0,2))
        X_right = np.array(X_right) if len(X_right) > 0 else np.zeros((0,2))
        print(f"X_left points: {X_left.shape[0]}, X_right points: {X_right.shape[0]}")

        print("Starting polynomial interpolation section...")
        poly_debug_img = hough_debug_img.copy()

        x_start_right = None
        x_start_left = None

        # Process left lane points
        if len(X_left) >= 2:
            try:
                ransac_left = RANSACRegressor()
                ransac_left.fit(X_left[:, 0].reshape(-1, 1), X_left[:, 1])
                line_X_left = np.linspace(X_left[:, 0].min(), X_left[:, 0].max(), 100).reshape(-1, 1)
                line_y_ransac_left = ransac_left.predict(line_X_left)
                cv2.polylines(poly_debug_img, [np.int32(list(zip(line_X_left.flatten(), line_y_ransac_left.flatten())))], False, (255, 0, 0), 2)
                print(f"RANSAC Left Lane: X range {line_X_left.flatten()[0]} to {line_X_left.flatten()[-1]}, Y range {line_y_ransac_left[0]} to {line_y_ransac_left[-1]}")
                x_start_left = line_X_left.flatten()[0]
            except Exception as e:
                print(f"RANSAC fitting failed for left lane: {e}")
                x_start_left = None
        else:
            print("Not enough points for RANSAC left lane detection.")
            x_start_left = None

        # Process right lane points
        if len(X_right) >= 2:
            try:
                ransac_right = RANSACRegressor()
                ransac_right.fit(X_right[:, 0].reshape(-1, 1), X_right[:, 1])
                line_X_right = np.linspace(X_right[:, 0].min(), X_right[:, 0].max(), 100).reshape(-1, 1)
                line_y_ransac_right = ransac_right.predict(line_X_right)
                cv2.polylines(poly_debug_img, [np.int32(list(zip(line_X_right.flatten(), line_y_ransac_right.flatten())))], False, (0, 255, 255), 2)
                print(f"RANSAC Right Lane: X range {line_X_right.flatten()[0]} to {line_X_right.flatten()[-1]}, Y range {line_y_ransac_right[0]} to {line_y_ransac_right[-1]}")
                x_start_right = line_X_right.flatten()[-1]
            except Exception as e:
                print(f"RANSAC fitting failed for right lane: {e}")
                x_start_right = None
        else:
            print("Not enough points for RANSAC right lane detection.")
            x_start_right = None

        cv2.imwrite(os.path.join(path, f"polynomial_lines_{getTime()}.jpg"), poly_debug_img)
        print("Polynomial lines computed and visualized.")

        print("Starting steering angle calculation...")
        
        if (x_start_right is not None) and (x_start_left is not None):
            mid_star = 0.5 * (x_start_right + x_start_left)
            print(f"Both lanes detected. mid_star: {mid_star}")
        elif (x_start_right is not None) and (x_start_left is None):
            # Only right lane detected
            mid_star = estimate_mid_star_from_one_lane(X_right, lane_side='right')
            print(f"Only right lane detected. Estimated mid_star: {mid_star}")
        elif (x_start_right is None) and (x_start_left is not None):
            # Only left lane detected
            mid_star = estimate_mid_star_from_one_lane(X_left, lane_side='left')
            print(f"Only left lane detected. Estimated mid_star: {mid_star}")
        else:
            mid_star = 159
            print("No lanes detected. Using default mid_star: 159")

        print('Computing servo angle from mid_star offset...')

        # Linear mapping from horizontal offset to servo angle
        dx = mid_star - 160  # Offset from center (160)
        # dx=0    => servo=90 (straight)
        # dx=+160 => servo=0 (full right)
        # dx=-160 => servo=180 (full left)
        servo_angle = 90 - (dx * (90/160.0))
        servo_angle = np.clip(servo_angle, 0, 180)
        print(f"Calculated servo angle: {servo_angle}")

        stable_steering_angle = stabilize_steering_angle(servo_angle, past_steering_angle)
        print(f"Stabilized steering angle: {stable_steering_angle}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(stable_steering_angle)
        cv2.putText(poly_debug_img, text, (110, 30 ), font, 1, (0, 0, 255), 2)
        print("Steering angle text written on image.")
        text = str(servo_angle)
        cv2.putText(poly_debug_img, text, (130, 50), font, 1, (0, 0, 255), 2)

        # Concatenate top and bottom images
        top_section = raw_image[:crop_height,:]
        top_h, top_w, _ = top_section.shape
        poly_h, poly_w, _ = poly_debug_img.shape

        if poly_w < top_w:
            diff = top_w - poly_w
            poly_debug_img = cv2.copyMakeBorder(poly_debug_img, 0, 0, 0, diff, cv2.BORDER_CONSTANT, value=(0,0,0))
            print("Padded poly_debug_img to match top width.")

        new_frame = np.concatenate((top_section, poly_debug_img), axis=0)
        print("Concatenated top and bottom images.")

        height, width, _ = new_frame.shape

        # --- Start of the modified line drawing code ---
        # Draw the magenta line at the bottom center representing the steering angle
        center_x = width // 2
        center_y = height - 50  # 50 px from the bottom edge
        line_length = 100

        theta = np.deg2rad(90 - servo_angle)
        end_x = int(center_x + line_length * np.sin(theta))
        end_y = int(center_y - line_length * np.cos(theta))

        cv2.line(new_frame, (center_x, center_y), (end_x, end_y), (255,0,255), 5)
        print("Drew steering line at bottom center of new_frame.")
        # --- End of the modified line drawing code ---

        # Overlay centroids onto new_frame (add offsets)
        print("Overlaying centroids onto the final image...")
        for data_item in patch_centroids_data:
            cx, cy = data_item['centroid']
            cv2.circle(new_frame, (cx + crop_width, cy + crop_height), 5, (0,165,255), -1)

        cv2.imwrite(os.path.join(path, f"final_frame_image_{getTime()}.jpg"), new_frame)
        print("Steering angle computed, visualized, and saved.")
        print("End.")
