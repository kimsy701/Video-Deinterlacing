import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

############################################################   Edge Detection 을 활용한 combing noise 잡기 ############################################################

def detect_combing_noise_with_edge(image_path, save_path, th1, th2):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection with tuned parameters
    edges = cv2.Canny(gray, th1, th2)
    # edges = cv2.Canny(gray, 50, 150)
    # edges = cv2.Canny(gray, 100, 200)

    # Define custom filters for detecting smaller diagonal patterns
    kernel_1 = np.array([[1, -2, 1], [-2, 5, -2], [1, -2, 1]])  # Diagonal from top-left to bottom-right
    kernel_2 = np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]])  # Diagonal from top-right to bottom-left

    # Apply the filters
    filtered_1 = cv2.filter2D(edges, -1, kernel_1)
    filtered_2 = cv2.filter2D(edges, -1, kernel_2)

    # Combine the filtered images
    filtered_combined = cv2.addWeighted(filtered_1, 0.5, filtered_2, 0.5, 0)

    # Apply morphological operations to enhance the detected patterns
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(filtered_combined, cv2.MORPH_CLOSE, kernel)

    # Apply dilation and erosion for further enhancement
    dilated = cv2.dilate(morph, kernel, iterations=1)
    eroded = cv2.erode(dilated, kernel, iterations=1)

    # Display the original image and the processed image
    # plt.imsave(save_path,eroded, cmap=cm.gray)
    plt.imsave(save_path,eroded)
    plt.axis('off')
    
    return eroded
    
def compare_edge_detections(image_path, th2_1_save_path, th2_2_save_path, diff_save_path,diff_thres_save_path, th1, th2_1, th2_2, diff_thres):
    # Detect edges with two different th2 values
    edge_1 = detect_combing_noise_with_edge(image_path, th2_1_save_path, th1, th2_1)
    edge_2 = detect_combing_noise_with_edge(image_path, th2_2_save_path, th1, th2_2)

    # Calculate absolute difference between the two edge-detected images
    difference = cv2.absdiff(edge_1, edge_2)
    
    #threshold 걸어보기 
    # Apply threshold to the difference image
    _, difference_thres = cv2.threshold(difference, diff_thres, 255, cv2.THRESH_BINARY)
    
    print(np.max(difference_thres)) #255
    print(np.min(difference_thres)) #0
    print(np.mean(difference_thres)) #9.17

    # plt.subplot(1, 3, 3)
    # plt.title("Absolute Difference")
    # plt.imshow(difference, cmap='gray')
    # plt.axis('off')
    # plt.imsave(diff_save_path, difference, cmap=cm.gray)
    plt.imsave(diff_save_path, difference)
    
    
    plt.imsave(diff_thres_save_path, difference_thres)
    plt.axis('off')


    


######### process #########

th1 = 0
th2_1 = 75
th2_2 = 45 #45이하여야 대각선 노이즈가 다 잡힘 

diff_thres= 254 #차이가 diff_thres 이상인 것만 그리기 


image_path = "/mnt/sdb/deinter/training_code/training_code_fi/training_code_fi_fi/qt_noise.png"
save_path = f"/mnt/sdb/deinter/training_code/training_code_fi/training_code_fi_fi/qt_noise_mask_th1_{th1}_th2_{th2_1}.png"
th2_1_save_path=f"/mnt/sdb/deinter/training_code/training_code_fi/training_code_fi_fi/qt_noise_mask_th1_{th1}_th2_{th2_1}.png"
th2_2_save_path=f"/mnt/sdb/deinter/training_code/training_code_fi/training_code_fi_fi/qt_noise_mask_th1_{th1}_th2_{th2_2}.png"
diff_save_path=f"/mnt/sdb/deinter/training_code/training_code_fi/training_code_fi_fi/qt_noise_mask_diff.png"
diff_thres_save_path=f"/mnt/sdb/deinter/training_code/training_code_fi/training_code_fi_fi/qt_noise_mask_diff_thres_diffthr{diff_thres}.png"

compare_edge_detections(image_path, th2_1_save_path, th2_2_save_path, diff_save_path,diff_thres_save_path, th1, th2_1, th2_2, diff_thres)




############################################################   Flow (Unimatch) 을 활용한 combing noise 잡기 ############################################################
