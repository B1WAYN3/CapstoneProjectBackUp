import cv2
import numpy as np
import logging
import datetime
import math
import sys
import os

# Constants
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
past_steering_angle = 0
row_threshold = 0
path = "/home/pi/repo/Capstone-Dallas-Edgar/research/ESEColorDetection/PatchData"
crop_height = int(SCREEN_HEIGHT * 0.10)  # This will be 120 pixels
ifblue = False

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

times2Run = {1}

for i in times2Run:
    camera.read()  # Discard the first frame
    successfulRead, raw_image = camera.read()
    if not successfulRead:
        print("Image not taken successful.")
        break

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

    crop_width = 20
    mask_edges = mask_edges[:, crop_width:]
    adjusted_screen_width = SCREEN_WIDTH - crop_width
    print(f"New width after cropping: {adjusted_screen_width}")

    cv2.imwrite(os.path.join(path, f"cropped_mask_edges_{getTime()}.jpg"), mask_edges)

    minLineLength = 12
    maxLineGap = 3
    min_threshold = 5

    print('Applying Probabilistic Hough Transform...')
    lines = cv2.HoughLinesP(mask_edges, 1, np.pi/180, min_threshold, minLineLength, maxLineGap)

    if lines is not None:
        hough_debug_img = cv2.cvtColor(mask_edges, cv2.COLOR_GRAY2BGR)
        print("Detected lines (in mask_edges coords):")
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # print(f"Line: ({x1},{y1}) -> ({x2},{y2})")
            cv2.line(hough_debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    print("Saving Images without calculating angle.")
    cv2.imwrite(os.path.join(path, f"img_rgb_{getTime()}.jpg"), img_rgb)
    cv2.imwrite(os.path.join(path, f"img_bottom_half_bgr_{getTime()}.jpg"), img_bottom_half_bgr)
    cv2.imwrite(os.path.join(path, f"img_crop_hsv_{getTime()}.jpg"), img_crop_hsv)
    cv2.imwrite(os.path.join(path, f"mask_{getTime()}.jpg"), mask)
    cv2.imwrite(os.path.join(path, f"mask_blurred_{getTime()}.jpg"), mask_blurred)
    cv2.imwrite(os.path.join(path, f"mask_edges_{getTime()}.jpg"), mask_edges)
    if lines is not None:
        cv2.imwrite(os.path.join(path, f"hough_lines_{getTime()}.jpg"), hough_debug_img)

    # Lower threshold more to get patches
    threshold = 10
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

        num_patches_vertical = 4
        patch_height = (SCREEN_HEIGHT - crop_height) // num_patches_vertical
        print(f'Pathc Height: {patch_height}')
        patch_width = 20
        list_patch = []

        print("Patches (in mask_edges coords):")
        for (seg_start, seg_end) in segments:
            col_center = (seg_start + seg_end) // 2
            x0 = max(col_center - patch_width//2, 0)
            x1 = min(col_center + patch_width//2, adjusted_screen_width - 1)
            print(f'Col_center: {col_center}, x0: {x0}, x1: {x1}')
            for k in range(num_patches_vertical):
                y0 = k * patch_height
                y1 = (k+1) * patch_height - 1
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
            cv2.rectangle(hough_debug_img, (x0 + crop_width, y0), (x1 + crop_width, y1), (50,255, 0), 1)
    else:
        print("lines is NONE")

    print("Saving Image With Lines (Dynamic Patches).")
    cv2.imwrite(os.path.join(path, f"image_lines_bottom_half_raw{getTime()}.jpg"), img_bottom_half_bgr)
    if lines is not None:
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
            for detected_line in lines:
                lx1, ly1, lx2, ly2 = detected_line[0]
                #print(f"Checking line ({lx1},{ly1})-({lx2},{ly2}) against patch x=({px_start},{px_end}), y=({py_start},{py_end})")

                if (lx1 >= px_start and lx1 <= px_end and ly1 >= py_start and ly1 <= py_end and
                    lx2 >= px_start and lx2 <= px_end and ly2 >= py_start and ly2 <= py_end):
                    inside_points.append([lx1, ly1])
                    inside_points.append([lx2, ly2])
                    #print(f"Line endpoints inside patch: Start ({lx1}, {ly1}), End ({lx2}, {ly2})")  # Debug line endpoints

            if len(inside_points) > 0:
                inside_points = np.array(inside_points)
                centroid_coords = np.mean(inside_points, axis=0).astype(int)
                print(f'Centroid_Coords: {centroid_coords}')
                patch_centroids_data.append({'patch': patch_info, 'centroid': (centroid_coords[0], centroid_coords[1])})
                cv2.circle(centroid_debug_image, (centroid_coords[0] + crop_width, centroid_coords[1]), 3, (0,165,255), -1)
                print(f"Centroid: ({centroid_coords[0]},{centroid_coords[1]})")

        cv2.imwrite(os.path.join(path, f"centroids_visualized_{getTime()}.jpg"), centroid_debug_image)
        print("Centroids computed and visualized on debug image.")

        X_left = []
        X_right = []
        n_right_side_right_dir = 0
        n_right_side_left_dir = 0
        n_left_side_right_dir = 0
        n_left_side_left_dir = 0

        image_center = adjusted_screen_width // 2
        print(f"Using image_center={image_center} to divide left/right lanes.")

        print("Separating centroids into left and right sets for polynomial interpolation...")

        numTimes = 0 

        for data_item in patch_centroids_data:
            numTimes
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

        if len(X_right) > 1:
            print(f"Right side has {X_right.shape[0]} points, applying polyfit...")
            right_lane = np.polyfit(X_right[:,0], X_right[:,1], 1, w=X_right[:,1])
            print(f"Polynomial coefficients (slope, intercept) for right lane: {right_lane}")
            y1_right = right_lane[0] * 219 + right_lane[1]
            y2_right = right_lane[0] * 319 + right_lane[1]
            print(f"Calculated y1: {y1_right}, y2: {y2_right} for x=219 and x=319")
            x_start_right = int((25 - right_lane[1])/(right_lane[0]+0.001))
        else:
            print("Not enough points on the right side for polyfit.")

        if len(X_left) > 1:
            print(f"Left side has {X_left.shape[0]} points, applying polyfit...")
            left_lane = np.polyfit(X_left[:,0], X_left[:,1], 1, w=X_left[:,1])
            print(f"Polynomial coefficients (slope, intercept) for left lane: {left_lane}")
            y1_left = left_lane[0] * 0 + left_lane[1]
            y2_left = left_lane[0] * 100 + left_lane[1]
            print(f"Calculated y1: {y1_left}, y2: {y2_left} for x=0 and x=100")
            x_start_left = int((25 - left_lane[1])/(left_lane[0]+0.001))
        else:
            print("Not enough points on the left side for polyfit.")

        cv2.imwrite(os.path.join(path, f"polynomial_lines_{getTime()}.jpg"), poly_debug_img)
        print("Polynomial lines computed and visualized.")

        print("Starting steering angle calculation...")
        if (x_start_right is not None) and (x_start_left is not None):
            mid_star = 0.5 * (x_start_right + x_start_left)
            print(f"Both lanes detected. mid_star: {mid_star}")
        elif (x_start_right is not None) and (x_start_left is None):
            mid_star = (25-100)/right_lane[0] + 160
            print(f"Only right lane detected. mid_star: {mid_star}")
        elif (x_start_right is None) and (x_start_left is not None):
            mid_star = (25-100)/left_lane[0] + 160
            print(f"Only left lane detected. mid_star: {mid_star}")
        else:
            mid_star = 159
            print("No lanes detected. Using default mid_star: 159")

        print('Computing steering angle...')
        if np.abs(mid_star-160)<2:
            steering_angle = 90
            print(f'Current angle for TRUE (default): {np.abs(mid_star-160)}')
            print("Steering angle close to center, set to 90.")
        else:
            steering_angle = 90 + np.degrees(np.arctan((mid_star-160)/75.))
            steering_angle = np.clip(steering_angle,55,135)
            print(f"Calculated steering angle: {steering_angle}")

        stable_steering_angle = stabilize_steering_angle(steering_angle,past_steering_angle)
        print(f"Stabilized steering angle: {stable_steering_angle}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        text = str(stable_steering_angle)
        cv2.putText(poly_debug_img, text, (110, 30 ), font, 1, (0, 0, 255), 2)
        print("Steering angle text written on image.")

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

        # Now apply offsets (crop_width, crop_height) when drawing lines and centroids on new_frame

        # Draw right lane line if available
        if len(X_right) > 1:
            x1_right_adj = 219 + crop_width
            y1_right_adj = int(y1_right) + crop_height
            x2_right_adj = 319 + crop_width
            y2_right_adj = int(y2_right) + crop_height
            cv2.line(new_frame, (x1_right_adj, y1_right_adj), (x2_right_adj, y2_right_adj), (0,255,255), 5)

        # Draw left lane line if available
        if len(X_left) > 1:
            x1_left_adj = 0 + crop_width
            y1_left_adj = int(y1_left) + crop_height
            x2_left_adj = 100 + crop_width
            y2_left_adj = int(y2_left) + crop_height
            cv2.line(new_frame, (x1_left_adj, y1_left_adj), (x2_left_adj, y2_left_adj), (255,0,0), 5)

        # Draw mid_star line
        if (x_start_right is not None) and (x_start_left is not None):
            mid_star_adj_x = int(mid_star) + crop_width
            y_start = 25 + crop_height
            y_end = 100 + crop_height
            cv2.line(new_frame,(int(np.clip(mid_star_adj_x,-10000,10000)),y_start),(160+crop_width,y_end),(255,0,255),5)
        elif (x_start_right is not None) and (x_start_left is None):
            mid_star_adj_x = int(mid_star) + crop_width
            y_start = 25 + crop_height
            y_end = 100 + crop_height
            cv2.line(new_frame,(int(np.clip(mid_star_adj_x,-10000,10000)),y_start),(160+crop_width,y_end),(255,0,255),5)
        elif (x_start_right is None) and (x_start_left is not None):
            mid_star_adj_x = int(mid_star) + crop_width
            y_start = 25 + crop_height
            y_end = 100 + crop_height
            cv2.line(new_frame,(int(np.clip(mid_star_adj_x,-10000,10000)),y_start),(160+crop_width,y_end),(255,0,255),5)
        else:
            mid_star_adj_x = int(mid_star) + crop_width
            y_start = 25 + crop_height
            y_end = 100 + crop_height
            cv2.line(new_frame,(int(np.clip(mid_star_adj_x,-10000,10000)),y_start),(160+crop_width,y_end),(255,0,255),5)

        print("Drew steering line on new_frame.")

        # Overlay centroids onto new_frame (add offsets)
        print("Overlaying centroids onto the final image...")
        for data_item in patch_centroids_data:
            cx, cy = data_item['centroid']
            cv2.circle(new_frame, (cx + crop_width, cy + crop_height), 5, (0,165,255), -1)
            # print(f"Overlayed centroid at ({cx + crop_width}, {cy + crop_height})")

        cv2.imwrite(os.path.join(path, f"final_frame_image_{getTime()}.jpg"), new_frame)
        print("Steering angle computed and visualized.")
        print("End.")
