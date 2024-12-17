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
# path = "/home/pi/repo2/CapstoneProjectBackUp/research/image/Data" # Directory for pi
path = "C:/Users/Wayne/OneDrive/Desktop/FALL_2024/Classes/Capstone Project/CapstoneProjectBackUp/research/image/Data"
 # Directory for local machine. 
# image_import_path = "/home/pi/repo2/CapstoneProjectBackUp/research/image/Data/raw_imports" # Directory for Pi 
image_import_path = "C:/Users/Wayne/OneDrive/Desktop/FALL_2024/Classes/Capstone Project/CapstoneProjectBackUp/research/image/Data/raw_imports"  # Directory for local machine
crop_height = int(SCREEN_HEIGHT * 0.10)  # This will be 48 pixels
ifblue = False
# Set a threshold at 5% of the observed peak for a clear yellow presence
yellow_threshold = 500 * 0.05  # Adjust based on actual peak observations

# Hough Transform Threshold Values to reduce noise and Fragmentation: 
minLineLength = 40  # Reduced from 60 to capture shorter lines
maxLineGap = 30     # Increased from 25 to allow more gap between segments
angle_threshold = 45  # Widened from 25 degrees to allow more slanted lines
min_threshold = 30 # Increased to require stronger edges for line detection

# Patch Centroid Sampling. 
num_samples = 50

# Midstar
previous_mid_star = None

use_live_camera = False  # Set this to False to load image from file
print(f"Use_live_camera set too: {use_live_camera}")
image_center = SCREEN_WIDTH // 2  # Initialize image_center based on SCREEN_WIDTH

# Initialize camera
camera = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))  # Set YUYV format

def getTime():
    return datetime.datetime.now().strftime("S_%S_M_%M")

def stabilize_steering_angle(curr_steering_angle, last_steering_angle=None, alpha=0.5):
    if last_steering_angle is None:
        return int(curr_steering_angle)
    else:
        smoothed_angle = alpha * curr_steering_angle + (1.0 - alpha) * last_steering_angle
        smoothed_angle = np.clip(smoothed_angle, 0, 180)
        return int(smoothed_angle)

def smooth_mid_star(curr_mid_star, last_mid_star=None, alpha=0.2):
    if last_mid_star is None:
        return curr_mid_star
    return alpha * curr_mid_star + (1 - alpha) * last_mid_star

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
            print("Live Image not taken successfully.")
            break
    else:
        # Load image from file instead of capturing from camera
        #image_file = os.path.join(image_import_path, "raw_image_S_02_M_53.jpg")
        image_file = os.path.join(image_import_path, "test.jpg")
        if not os.path.exists(image_file):
            print(f"Image file {image_file} does not exist.")
            break
        raw_image = cv2.imread(image_file)
        if raw_image is None:
            print("Failed to load image from file.")
            break
        print("Imported Image for Testing successful, no live data.")

    print("Saving Raw Image before flipping.")
    cv2.imwrite(os.path.join(path, f"raw_image_{getTime()}.jpg"), raw_image)

    if raw_image.shape[1] != SCREEN_WIDTH or raw_image.shape[0] != SCREEN_HEIGHT:
        print(f"Warning: Image dimensions mismatch. Expected: {SCREEN_WIDTH}x{SCREEN_HEIGHT}, Got: {raw_image.shape[1]}x{raw_image.shape[0]}")

    raw_image = cv2.flip(raw_image, -1)
    print("Saving Flipped Image.")
    cv2.imwrite(os.path.join(path, f"flipped_image_raw_{getTime()}.jpg"), raw_image)

    print('Img to color...')
    img_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)

    print('Cropping top portion of the image...')
    img_bottom_half_bgr = img_rgb[crop_height:,:]

    print('Performing HSV color space transformation...')
    img_hsv = cv2.cvtColor(img_bottom_half_bgr, cv2.COLOR_RGB2HSV)
    img_crop_hsv = img_hsv

    center_rect_width_start = int(img_hsv.shape[1] * 0.35)
    center_rect_width_end = int(img_hsv.shape[1] * 0.71)
    center_rect = img_hsv[int(img_hsv.shape[0] * 0.35):int(img_hsv.shape[0] * 0.71), center_rect_width_start:center_rect_width_end]

    median_brightness_center = np.median(center_rect[:, :, 2])
    if median_brightness_center > 65:  # Threshold for excessive brightness
        print("Excessive brightness/object detected in the center, adjusting mask...")
        img_hsv[:, int(img_hsv.shape[1] * 0.30):int(img_hsv.shape[1] * 0.71), 2] = 0

    print('Creating binary masks for white and yellow lanes after HSV...')
    if ifblue:
        # Existing blue lane detection
        print("Blue HSV thresholds set, calculating mask based on blue HSV bounds.")
        lower_hsv = np.array([100, 150, 50])
        upper_hsv = np.array([130, 255, 255])
        mask = cv2.inRange(img_crop_hsv, lower_hsv, upper_hsv)
    else:
        print("White and Yello HSV thresholds set, calculating mask based on white and yellow HSV bounds.")
        #Dynamically calculate white threshold values for different brightness. 
        median_brightness = np.median(img_hsv[:, :, 2])

        # Adjust the lower value of the V channel dynamically based on the median brightness. 
        lower_white = np.array([0, 0, max(100, median_brightness - 30)])
        upper_white = np.array([180, 30, 255])

        # dynamically set the yellow lines deteciton format based on conditions. 
        # Computer the historgram for the Hue channel in the typical yellow range
        hue_hist = cv2.calcHist([img_hsv], [0], None, [180], [20,30])
        yellow_presence = np.sum(hue_hist)

        if yellow_presence > yellow_threshold:  # Define a suitable threshold based on your observations
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
        else:
            # Adjust the range if yellow is not prominently found
            lower_yellow = np.array([22, 80, 80])
            upper_yellow = np.array([28, 255, 255])
        
        # Create masks for white and yellow lanes
        mask_white = cv2.inRange(img_crop_hsv, lower_white, upper_white)
        mask_yellow = cv2.inRange(img_crop_hsv, lower_yellow, upper_yellow)
        print("Saving White Mask")
        cv2.imwrite(os.path.join(path, f"mask_white_{getTime()}.jpg"), mask_white)
        print("Saving yellow Mask")
        cv2.imwrite(os.path.join(path, f"mask_yellow_{getTime()}.jpg"), mask_yellow)
        
        # Combine masks to detect both white and yellow lanes
        mask = cv2.bitwise_or(mask_white, mask_yellow)
        print("Saving Yellow and White ored Mask. ")
        cv2.imwrite(os.path.join(path, f"mask_yellow_white_or_{getTime()}.jpg"), mask)
    

    # -------------- STATE: MORPHOLOGICAL OPERATION --------------
    print('Applying morphological operations to enhance lane lines...')
    
    # Step 1: Dilation to close small gaps
    dilate_kernel_1 = np.ones((2, 2), np.uint8)  # Slightly larger kernel for initial dilation
    mask_dilated_1 = cv2.dilate(mask, dilate_kernel_1, iterations=1)
    # cv2.imwrite(os.path.join(path, f"mask_dilated_1_{getTime()}.jpg"), mask_dilated_1)
    # print("Saved mask after initial dilation to close small gaps.")
    
    # Step 2: Closing operation to fill gaps in detected lanes
    close_kernel = np.ones((3, 3), np.uint8)  # Increased kernel size for closing
    mask_closed = cv2.morphologyEx(mask_dilated_1, cv2.MORPH_CLOSE, close_kernel)
    # cv2.imwrite(os.path.join(path, f"mask_closed_{getTime()}.jpg"), mask_closed)
    # print("Saved mask after closing operation to fill gaps.")

    # Step 3: Opening operation to remove small noise
    open_kernel = np.ones((2, 2), np.uint8)
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, open_kernel)
    # cv2.imwrite(os.path.join(path, f"mask_opened_{getTime()}.jpg"), mask_opened)
    # print("Saved mask after opening operation to remove small noise.")

    # Step 4: Second dilation to thicken the lines further
    #dilate_kernel_2 = np.ones((3, 3), np.uint8)
    # mask = cv2.dilate(mask_opened, dilate_kernel_2, iterations=1)
    # cv2.imwrite(os.path.join(path, f"mask_dilated_2_{getTime()}.jpg"), mask)
    # print("Saved mask after second dilation to thicken lines.")
    
    print("Saving final mask after morphological operations.")
    cv2.imwrite(os.path.join(path, f"mask_morphological_{getTime()}.jpg"), mask)

    # -------------- STATE: FIRST GAUSSIAN BLUR  --------------
    print('Applying Gaussian blur on mask...')
    mask_blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # -------------- STATE: CANNY FILTER --------------
    print('Applying Canny filter...')
    mask_edges = cv2.Canny(mask_blurred, 50, 150) 
    cv2.imwrite(os.path.join(path, f"mask_edges_canny_filter_{getTime()}.jpg"), mask_edges)
    
    # -------------- STATE: BITWISE MASK AND MASK_EDGES --------------
    print("Bitwising mask and mask_edges.")
    mask_edges = cv2.bitwise_or(mask, mask_edges)
    cv2.imwrite(os.path.join(path,f"mask_edges_bitwise_{getTime()}.jpg"), mask_edges)
    
    # -------------- STATE: SECOND GUASSIAN BLUR --------------
    # Apply Gaussian Blur to smooth edges before Hough Transform
    print('Applying Gaussian blur to mask_edges to improve Hough Transform results...')
    mask_edges = cv2.GaussianBlur(mask_edges, (5, 5), 0)
    cv2.imwrite(os.path.join(path, f"mask_blurred_second_{getTime()}.jpg"), mask_edges)
    
    # -------------- STATE: CROPPING (LEFT WHITE LINE) --------------
    crop_width = 20
    mask_edges = mask_edges[:, crop_width:]
    adjusted_screen_width = SCREEN_WIDTH - crop_width
    print(f"New width after cropping: {adjusted_screen_width}")
    print("Saving Cropped Masked Edges Image.")
    cv2.imwrite(os.path.join(path, f"cropped_mask_edges_after_some_filters{getTime()}.jpg"), mask_edges)

    # -------------- STATE: ROI --------------
    print('Applying Region of Interest (ROI)...')
    mask_roi = np.zeros_like(mask_edges, dtype=np.uint8)
    cv2.imwrite(os.path.join(path, f"mask_roi_before_{getTime()}.jpg"), mask_roi)

    # Define polygons more narrowly focused on expected lane areas, avoiding the center
    height_roi, width_roi = mask_edges.shape

    # Adjusted ROI polygons to be more rectangular and cover full height
    left_polygon = np.array([
        [0, 0],  # Top-left corner
        [int(width_roi * 0.33), 0],  # Top-inner corner (1/3 width from the left)
        [int(width_roi * 0.33), height_roi],  # Bottom-inner corner
        [0, height_roi]  # Bottom-left corner
    ], dtype=np.int32)

    right_polygon = np.array([
        [width_roi, 0],  # Top-right corner
        [int(width_roi * 0.67), 0],  # Top-inner corner (1/3 width from the right)
        [int(width_roi * 0.67), height_roi],  # Bottom-inner corner
        [width_roi, height_roi]  # Bottom-right corner
    ], dtype=np.int32)

    # Visualize ROI polygons on a copy of the original edges image for debugging
    roi_debug_image = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR for coloring
    cv2.polylines(roi_debug_image, [left_polygon], True, (0, 255, 0), 2)  # Draw left polygon in green
    cv2.polylines(roi_debug_image, [right_polygon], True, (0, 0, 255), 2)  # Draw right polygon in red
    cv2.imwrite(os.path.join(path, f"visual_roi_polygons_{getTime()}.jpg"), roi_debug_image)
    print("Visualized ROI polygons for debugging.")

    # Fill the left and right polygons on the ROI mask
    cv2.fillPoly(mask_roi, [left_polygon, right_polygon], 255)
    cv2.imwrite(os.path.join(path, f"mask_roi_filled_{getTime()}.jpg"), mask_roi)  # Save the filled ROI mask
    print("Saved filled ROI mask.")

    # Apply the ROI mask to the edge-detected image
    mask_edges = cv2.bitwise_and(mask_edges, mask_roi)
    print("Applied ROI mask to edges and saved result.")
    cv2.imwrite(os.path.join(path, f"mask_edges_roi_after_{getTime()}.jpg"), mask_edges)

    # -------------- STATE: Hough Transform --------------
    print('Applying Probabilistic Hough Transform with adjusted parameters...')
    lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, min_threshold, minLineLength, maxLineGap)

    if lines is not None:
        hough_debug_img = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)
        print("Detected lines (in mask_edges coords):")

        # This list keeps lines passing the angle filter
        filtered_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            if -angle_threshold < angle < angle_threshold:
                filtered_lines.append(line)
                cv2.line(hough_debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                #print(f"Accepted Line: ({x1},{y1}) -> ({x2},{y2}) with angle {angle:.2f} degrees")
            else:
                # print(f"Rejected Line: ({x1},{y1}) -> ({x2},{y2}) with angle {angle:.2f} degrees")
                continue
                
        print(f"Total detected lines: {len(lines)}, Filtered lines: {len(filtered_lines)}")
        print(f"Line Data at hough transform first step: {lines}")

        # -------------- STATE: MERGE LINES --------------
        print("Starting merge lines state.")
        slope_epsilon = 0.1
        intercept_epsilon = 20  

        # Convert filtered_lines to slope-intercept representation
        line_params = []
        for fl in filtered_lines:
            x1, y1, x2, y2 = fl[0]
            if x2 == x1:
                # Vertical line handling: slope = large number
                slope = 999999  
                intercept = x1
            else:
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
            line_params.append([slope, intercept, x1, y1, x2, y2])

        print(f"line_params: {line_params}")
        merged_lines = []
        used = [False] * len(line_params)

        print("Merging lines based on slope/intercept proximity...")
        for i, lp1 in enumerate(line_params):
            if used[i]:
                continue
            slope1, intercept1, x1_1, y1_1, x2_1, y2_1 = lp1

            group_slopes = [slope1]
            group_intercepts = [intercept1]
            group_points = [(x1_1, y1_1), (x2_1, y2_1)]

            used[i] = True

            # Merge with any lines that are close in slope/intercept
            for j, lp2 in enumerate(line_params[i+1:], start=i+1):
                if used[j]:
                    continue
                slope2, intercept2, x1_2, y1_2, x2_2, y2_2 = lp2
                if abs(slope1 - slope2) < slope_epsilon and abs(intercept1 - intercept2) < intercept_epsilon:
                    used[j] = True
                    group_slopes.append(slope2)
                    group_intercepts.append(intercept2)
                    group_points.append((x1_2, y1_2))
                    group_points.append((x2_2, y2_2))

            avg_slope = np.mean(group_slopes)
            avg_intercept = np.mean(group_intercepts)

            all_x = [pt[0] for pt in group_points]
            all_y = [pt[1] for pt in group_points]
            min_x, max_x = min(all_x), max(all_x)

            # If slope is not near zero, compute y from slope-intercept
            # If near vertical, just take min_y, max_y from endpoints
            if abs(avg_slope) > 1e-5 and (max_x - min_x > 1):
                min_y = avg_slope * min_x + avg_intercept
                max_y = avg_slope * max_x + avg_intercept
            else:
                min_y = min(all_y)
                max_y = max(all_y)

            merged_lines.append([int(min_x), int(min_y), int(max_x), int(max_y)])

        # Visualize merged lines in cyan
        merged_hough_debug_img = hough_debug_img.copy()
        merged_lines_array = []
        for ml in merged_lines:
            x1m, y1m, x2m, y2m = ml
            cv2.line(merged_hough_debug_img, (x1m, y1m), (x2m, y2m), (255, 255, 0), 2)
            merged_lines_array.append([[x1m, y1m, x2m, y2m]])

        cv2.imwrite(os.path.join(path, f"hough_lines_merged_{getTime()}.jpg"), merged_hough_debug_img)
        print("Merged lines debug image saved (cyan lines).")
        print(f"merged_lines_array: {merged_lines_array}")

        # ---------------- SECOND PASS: ONE LINE PER SIDE ----------------
        # separate the merged lines by sign of slope, then pick the
        # single longest line from each cluster. This should yield ~2 lines total.

        print("Refining merged lines to get only one line per side.")
        final_lines_array = []
        neg_slope_lines = []
        pos_slope_lines = []

        # We re-derive slope from merged lines
        for ml in merged_lines_array:
            x1m, y1m, x2m, y2m = ml[0]
            dx = (x2m - x1m)
            if dx == 0:
                slope_merged = 999999
            else:
                slope_merged = (y2m - y1m) / float(dx)
            # length of the line (for picking the longest)
            length_merged = math.hypot(x2m - x1m, y2m - y1m)

            if slope_merged < 0:
                neg_slope_lines.append((ml[0], length_merged))
            else:
                pos_slope_lines.append((ml[0], length_merged))

        # pick the longest negative slope line (if exists)
        if len(neg_slope_lines) > 0:
            longest_neg_line = max(neg_slope_lines, key=lambda x: x[1])[0]
            final_lines_array.append([longest_neg_line])  # store in nested format
            print("Picked longest negative slope line for left lane:", longest_neg_line)
        else:
            print("No negative slope lines found in merged set.")

        # pick the longest positive slope line (if exists)
        if len(pos_slope_lines) > 0:
            longest_pos_line = max(pos_slope_lines, key=lambda x: x[1])[0]
            final_lines_array.append([longest_pos_line])
            print("Picked longest positive slope line for right lane:", longest_pos_line)
        else:
            print("No positive slope lines found in merged set.")

        # After final_lines_array is formed and before computing centroids,
        # adjust all line endpoints by -crop_width so they match the cropped image coords.
        print(f"Shifting final line endpoints by {-crop_width} to match cropped mask coordinate system.")
        for idx in range(len(final_lines_array)):
            x1, y1, x2, y2 = final_lines_array[idx][0]
            # Shift the x-coords
            x1_adj = max(x1 - crop_width, 0)
            x2_adj = max(x2 - crop_width, 0)

            # Overwrite final line with adjusted coordinates
            final_lines_array[idx] = [[x1_adj, y1, x2_adj, y2]]
            print(f"Adjusted Final Line {idx}: [{x1}->{x1_adj}, {y1}], [{x2}->{x2_adj}, {y2}]")
        
        # We'll visualize the final 1 or 2 lines in magenta on a new debug image
        final_hough_debug_img = hough_debug_img.copy()
        for fl in final_lines_array:
            x1f, y1f, x2f, y2f = fl[0]
            cv2.line(final_hough_debug_img, (x1f, y1f), (x2f, y2f), (255, 0, 255), 3)

        cv2.imwrite(os.path.join(path, f"hough_lines_merged_refined_{getTime()}.jpg"), final_hough_debug_img)
        print("Final refined lines debug image saved (magenta lines).")

        # Overwrite filtered_lines with only these final lines, and shifted final lines. 
        filtered_lines = final_lines_array
    else:
        hough_debug_img = None
        print("No lines detected within the Probabilistic Hough Transform...")
    
    final_hough_debug_img = hough_debug_img.copy()
    for fl in final_lines_array:
        x1f, y1f, x2f, y2f = fl[0]
        cv2.line(final_hough_debug_img, (x1f, y1f), (x2f, y2f), (255, 0, 255), 3)

    print("Saving Images without calculating angle.")
    cv2.imwrite(os.path.join(path, f"img_rgb_{getTime()}.jpg"), img_rgb)
    cv2.imwrite(os.path.join(path, f"img_bottom_half_bgr_{getTime()}.jpg"), img_bottom_half_bgr)
    cv2.imwrite(os.path.join(path, f"img_crop_hsv_{getTime()}.jpg"), img_crop_hsv)
    cv2.imwrite(os.path.join(path, f"mask_blurred_{getTime()}.jpg"), mask_blurred)
    cv2.imwrite(os.path.join(path, f"mask_edges_after_all_filters_{getTime()}.jpg"), mask_edges)
    if lines is not None:
        print("Hough Transform successful, saving hough transform image.")
        cv2.imwrite(os.path.join(path, f"hough_lines_output_{getTime()}.jpg"), hough_debug_img)

    # -------------- STATE: PATCHES --------------
    
    # Lower threshold more to get patches
    threshold = 5
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
        print(f"Start of lane_columns: {start}, Prev: {prev}")

        for c in lane_columns[1:]:
            # print(f"Current column: {c}, Previous column: {prev}")  # Debugging current and previous columns
            if c != prev + 1:
                print(f"Non-continuous segment detected. Appending segment: ({start}, {prev})")  # Debug when a segment is finalized
                segments.append((start, prev))
                start = c
                print(f"New start set to: {start}")  # Debug the new start
            prev = c

        segments.append((start, prev))
        print(f"Final segment appended: ({start}, {prev})")

        num_patches_horizontal = 30
        num_patches_vertical = 10 
        patch_height = int((SCREEN_HEIGHT - crop_height) // num_patches_vertical * 1.1)
        print(f'Patch Height: {patch_height}')
        patch_width = int((SCREEN_WIDTH - crop_width) // num_patches_horizontal)
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
        print("Lines is not None in patch state")
        for idx, patch in enumerate(list_patch):
            x0, x1 = patch['x']
            y0, y1 = patch['y']
            # print(f"Patch {idx}: x=({x0}, {x1}), y=({y0}, {y1})")
            # print(f"Patch {idx}: x0={x0}, x1={x1}, y0={y0}, y1={y1}")
            # cprint(f'Rectangle First Coordinate: {(x0 + crop_width, y0)}, Second: {(x1 + crop_width, y1)}')
            cv2.rectangle(img_bottom_half_bgr, (x0 + crop_width, y0), (x1 + crop_width, y1), (50,255, 0), 1)
            if hough_debug_img is not None:
                cv2.rectangle(hough_debug_img, (x0 + crop_width, y0), (x1 + crop_width, y1), (50,255, 0), 1)
    else:
        print("lines is NONE in patch state.")

    print("Saving Image With Lines (Dynamic Patches).")
    print(f"Patch Data: {patch}")
    if lines is not None:
        cv2.imwrite(os.path.join(path, f"image_lines_bottom_half_raw_{getTime()}.jpg"), img_bottom_half_bgr)
        cv2.imwrite(os.path.join(path, f"image_lines_masked_edges_{getTime()}.jpg"), hough_debug_img)

    # -------------- STATE: CENTROIDS --------------
    if lines is None:
        print("No Lines Detected in Path Seciton. Exiting Loop")
        break
    else:
        print("Calculating centroids...")

        centroid_debug_image = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)
        Blueimgae = centroid_debug_image.copy() 
        greenimage = centroid_debug_image.copy()
        patch_centroids_data = []

        for patch_info in list_patch:
            # Read patch coords in cropped coordinates
            px_start, px_end = patch_info['x']
            py_start, py_end = patch_info['y']

            # Shift patch's X bounds back to original image coords:
            px_start_global = px_start  + crop_width
            px_end_global   = px_end   + crop_width
   
            for detected_line in filtered_lines:
                # Each "detected_line" is a nested list: [[x1, y1, x2, y2]]
                lx1, ly1, lx2, ly2 = detected_line[0]
                        
                # Compute the bounding box of the line segment (original coords)
                lx_min, lx_max = min(lx1, lx2), max(lx1, lx2)
                ly_min, ly_max = min(ly1, ly2), max(ly1, ly2)

                # Compute the bounding box of the intersection
                ix_min = max(px_start_global, lx_min)
                ix_max = min(px_end_global, lx_max)
                iy_min = max(py_start, ly_min)
                iy_max = min(py_end, ly_max)
                
                # Correct the order dynamically if ix_min > ix_max
                if ix_min > ix_max:
                    ix_min, ix_max = ix_max, ix_min

                if iy_min > iy_max:
                    iy_min, iy_max = iy_max, iy_min
                
                
                # Ensure intersection coordinates are valid
                valid_intersection = ix_min <= ix_max and iy_min <= iy_max
                
                # inside_points will store all final-line points falling inside this patch
                inside_points = []
                
                # Draw the intersection region on the debug image
                if valid_intersection:
                    cv2.rectangle(
                        Blueimgae, 
                        (ix_min, iy_min), 
                        (ix_max, iy_max), 
                        (123, 12, 123), 1 
                    )
                    cv2.rectangle(
                        greenimage, 
                        (px_start_global, py_start), 
                        (px_end_global, py_end), 
                        (123, 123, 123), 2
                    )  # Green for patch bounding box
                    for t in np.linspace(0, 1, num_samples):
                        sx = int(lx1 + t * (lx2 - lx1))
                        sy = int(ly1 + t * (ly2 - ly1))
                        # Check if sample (sx, sy) is inside the patch
                        if ix_min <= sx <= ix_max and iy_min <= sy <= iy_max:
                            inside_points.append([sx, sy])
                            # Optionally visualize each sample point in blue
                            cv2.circle(centroid_debug_image,
                                        (sx + crop_width, sy),
                                        2, (255, 0, 0), -1)
                '''        
                print(f"Line bounding box: x_min={lx_min}, x_max={lx_max}, "
                    f"y_min={ly_min}, y_max={ly_max}")
                print(f"Patch bounding box: x_min={px_start_global}, x_max={px_end_global}, "
                    f"y_min={py_start}, y_max={py_end}")
                print(f"Intersection: ix_min={ix_min}, ix_max={ix_max}, "
                    f"iy_min={iy_min}, iy_max={iy_max}. Center: {image_center}")
                print(f"Corrected Intersection: ix_min={ix_min}, ix_max={ix_max}, iy_min={iy_min}, iy_max={iy_max}, valid={valid_intersection} Center: {image_center}")
                '''
                            
                # Now group the inside_points *per patch* to find a single centroid 
                # for that patch. If a patch had no intersection, inside_points is empty.
                if len(inside_points) > 0:
                    inside_points = np.array(inside_points)
                    centroid_coords = np.mean(inside_points, axis=0).astype(int)
                    #print(f'Centroid_Coords: {centroid_coords}')
                    patch_centroids_data.append({'patch': patch_info,
                                                'centroid': (centroid_coords[0], centroid_coords[1])})
                    # Visualize the centroid in orange (0,165,255)
                    cv2.circle(centroid_debug_image,
                            (centroid_coords[0] + crop_width, centroid_coords[1]),
                            5, (0,165,255), -1)
                    # print(f"Centroid: ({centroid_coords[0]},{centroid_coords[1]})")

        cv2.imwrite(os.path.join(path, f"centroids_visualized_{getTime()}.jpg"), centroid_debug_image)
        cv2.imwrite(os.path.join(path, f"greenimage_.jpg"), greenimage)
        cv2.imwrite(os.path.join(path, f"blueimage_.jpg"), Blueimgae)
        print("Centroids computed and visualized on debug image.")
        print(f"Centroid Data: {patch_centroids_data}")

        X_left = []
        X_right = []

        print(f"Using image_center={image_center} to divide left/right lanes.")
        cv2.line(hough_debug_img, (image_center + crop_width, 0),
        (image_center + crop_width, hough_debug_img.shape[0]),
        (0, 255, 0), 2)
        
        cv2.imwrite(os.path.join(path, f"hough_image_center_line{getTime()}.jpg"), hough_debug_img)

        # -------------- STATE: POLYNOMIAL INTERPOLATION --------------
        print("Separating centroids into left and right sets for polynomial interpolation...")

        numTimes = 0 

        for data_item in patch_centroids_data:
            numTimes += 1
            print(f'Number of times Separating Centroids Ran: {numTimes}')
            cx, cy = data_item['centroid']
            print(f"Centroid found at x={cx}, y={cy}")
            if cx < image_center:
                X_left.append([cx, cy])
                #print(f"Center Image: {image_center}: Added centroid ({cx}, {cy}) to X_left. Current X_left set: {X_left}")
            else:
                X_right.append([cx, cy])
                #print(f"Center Image: {image_center}: Added centroid ({cx}, {cy}) to X_right. Current X_right set: {X_right}")

        X_left = np.array(X_left) if len(X_left) > 0 else np.zeros((0,2))
        X_right = np.array(X_right) if len(X_right) > 0 else np.zeros((0,2))
        print(f"X_left points in polynomial state: {X_left.shape[0]}, X_right points: {X_right.shape[0]}")

        print("Starting polynomial interpolation process after seperation sides...")
        poly_debug_img = hough_debug_img.copy()

        x_start_right = None
        x_start_left = None


        # Process left lane points
        if len(X_left) >= 2:
            try:
                ransac_left=RANSACRegressor()
                ransac_left.fit(X_left[:,0].reshape(-1,1),X_left[:,1])
                line_X_left=np.linspace(X_left[:,0].min(),X_left[:,0].max(),100).reshape(-1,1)
                line_y_left=ransac_left.predict(line_X_left)
                if poly_debug_img is not None:
                    cv2.polylines(poly_debug_img, [np.int32(list(zip(line_X_left.flatten(), line_y_left.flatten())))],
                                  False,(255,0,0),2)
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
                ransac_right=RANSACRegressor()
                ransac_right.fit(X_right[:,0].reshape(-1,1),X_right[:,1])
                line_X_right=np.linspace(X_right[:,0].min(),X_right[:,0].max(),100).reshape(-1,1)
                line_y_right=ransac_right.predict(line_X_right)
                if poly_debug_img is not None:
                    cv2.polylines(poly_debug_img,[np.int32(list(zip(line_X_right.flatten(), line_y_right.flatten())))],
                                  False,(0,255,255),2)
                x_start_right=line_X_right.flatten()[-1]
            except Exception as e:
                print(f"RANSAC fitting failed for right lane: {e}")
                x_start_right = None
        else:
            print("Not enough points for RANSAC right lane detection.")
            x_start_right = None

        if poly_debug_img is not None:
            cv2.imwrite(os.path.join(path,f"polynomial_lines_{getTime()}.jpg"),poly_debug_img)
        print("Polynomial lines computed and visualized.")

        # -------------- STATE: STEERING ANGLE CALCULATION + MIDSTAR CALCULATION --------------
        print("Starting steering angle calculation...")
        
        if (x_start_right is not None) and (x_start_left is not None):
            mid_star=0.5*(x_start_right+x_start_left)
            print(f"Both lanes detected. mid_star: {mid_star}")
        elif (x_start_right is not None) and (x_start_left is None):
            mid_star=estimate_mid_star_from_one_lane(X_right,'right')
            print(f"Only right lane detected. Estimated mid_star: {mid_star}")
        elif (x_start_right is None) and (x_start_left is not None):
            mid_star=estimate_mid_star_from_one_lane(X_left,'left')
            print(f"Only left lane detected. Estimated mid_star: {mid_star}")
        else:
            mid_star=159
            print("No lanes detected in midstar process. Using default mid_star: 159")

        smoothed_mid_star = smooth_mid_star(mid_star, previous_mid_star, alpha=0.00000001)
        previous_mid_star = smoothed_mid_star
        
        # -------------- STATE: SERVO ANGLE TRANSLATION --------------
        print('Computing servo angle from mid_star offset...')
        dx = smoothed_mid_star  - 160  # Offset from center (160)
        half_width = 160              # half the 320-wide region
        # servo_angle = 100 - (dx * (80 / 160.0))
        servo_angle_raw = 180 - (smoothed_mid_star * (180.0 / 320.0))  # Range ~ [0..180]
        servo_angle = np.clip(servo_angle_raw , 0, 180)
        print(f"Calculated servo angle before stabilization: {servo_angle}")

        print(f"Past Steering angle:{past_steering_angle}")
        stable_servo_angle = stabilize_steering_angle(servo_angle_raw, past_steering_angle, alpha=27)
        past_steering_angle = stable_servo_angle
        print(f"mid_star={smoothed_mid_star:.1f}, servo_angle_raw={servo_angle_raw:.1f}, stable_servo_angle={stable_servo_angle}")

        # -------------- STATE: NEW FRAME --------------

        # Concatenate top and bottom images
        top_section = raw_image[:crop_height,:]
        top_h, top_w, _ = top_section.shape
        poly_h, poly_w, _ = poly_debug_img.shape

        if poly_debug_img is not None:
            poly_h, poly_w, _ = poly_debug_img.shape
            if poly_w < top_w:
                diff = top_w - poly_w
                poly_debug_img = cv2.copyMakeBorder(poly_debug_img, 0,0,0,diff, cv2.BORDER_CONSTANT,value=(0,0,0))
            new_frame = np.concatenate((top_section, poly_debug_img), axis=0)
        else:
            new_frame = np.concatenate((top_section, mask_edges), axis=0)
        print("Concatenated top and bottom images.")

        height, width, _ = new_frame.shape

        # --- Start of the modified line drawing code ---
        # Draw the magenta line at the bottom center representing the steering angle
        center_x = width // 2
        center_y = height - 50  # 50 px from the bottom edge
        line_length = 100

        theta = np.deg2rad(90 - stable_servo_angle)  # servo=90 => line straight up
        end_x = int(center_x + line_length * np.sin(theta))
        end_y = int(center_y - line_length * np.cos(theta))

        cv2.line(new_frame, (center_x, center_y), (end_x, end_y), (255,0,255), 5)
        print("Drew steering line at bottom center of new_frame.")
        
        # Define our smaller text font & thickness
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Text positions near bottom-left corner of that steering line
        text_x = 40
        text_y = center_y
        
        # Place SERVO text slightly above the MIDSTAR line
        servo_text_y = text_y - 20  # 20 px above mid_star text
        
        cv2.putText(
            new_frame, 
            f"Servo={stable_servo_angle:.1f}",
            (text_x, servo_text_y),
            font,
            font_scale,
            (0, 0, 255),
            thickness
        )

        cv2.putText(
            new_frame, 
            f"MidStar={smoothed_mid_star:.1f}",
            (text_x, text_y),
            font,
            font_scale,
            (0, 255, 0),
            thickness
        )
        print("Steering angle text written on image.")
        # --- End of the modified line drawing code ---

        # Overlay centroids onto new_frame (add offsets)
        print("Overlaying centroids onto the final image...")
        for data_item in patch_centroids_data:
            cx, cy = data_item['centroid']
            cv2.circle(new_frame, (cx + crop_width, cy + crop_height), 5, (0,165,255), -1)
            
        for fl in final_lines_array:
            x1, y1, x2, y2 = fl[0]
            # Transform back to the raw image coordinate system:
            x1_raw = x1 + crop_width
            x2_raw = x2 + crop_width
            y1_raw = y1 + crop_height
            y2_raw = y2 + crop_height
            
            # Draw the line in magenta
            cv2.line(new_frame, (x1_raw, y1_raw), (x2_raw, y2_raw), (255, 0, 255), 3)

        cv2.imwrite(os.path.join(path, f"final_frame_image_{getTime()}.jpg"), new_frame)
        print("Steering angle computed, visualized, and saved.")
        print("End.")
