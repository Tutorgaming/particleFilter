__author__ = 'Theppasith N'

import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import random

# Global Params
x_global, y_global = -1, -1
seed_point = (-1, -1)
frame = -1

# YELLOW [B]
landmark1_hsv_max = np.array([255, 247, 84])
landmark1_hsv_min = np.array([31, 35, 40])
#  PURPLE [G]
landmark2_hsv_max = np.array([255, 190, 255])
landmark2_hsv_min = np.array([0, 145, 0])
# landmark2_hsv_max = np.array([162, 92, 157])
# landmark2_hsv_min = np.array([50, 2, 81])
# BLUE
# landmark2_hsv_max = np.array([143, 168, 255])
# landmark2_hsv_min = np.array([96, 117, 0])



# Logitech C920
#  distance_mm = object_real_world_mm * focal-length_mm / object_image_sensor_mm
real_object_length = 21  # centimeters
focal_length = 3.67  # millimeters
px_per_mm = 165  # px/mm

# Particle Filter
particle_amount = 1000
# Particle Filter Parameters

particle_x_yellow = []
particle_y_yellow = []
particles_weight_yellow = []

particle_x_purple = []
particle_y_purple = []
particles_weight_purple = []

particle_x = []
particle_y = []
particles_weight = []

# RANGE	100	        125	        150	        175	        200	        225	        250	        275	        300
# MEAN	100.035 	125.2485	146.88	    173.265	    195.4572	223.1519	244.108	    272.6847	294.0169
# SD	0.71    	1.233059	0.953124	1.300061	1.426972	1.36659	    1.434685	1.85417	    2.427993

# class Particle(object):
#     def __init__(self, x, y, w=1):
#         self.x = x
#         self.y = y
#         self.w = w
#
#     @property
#     def xy(self):
#         return self.x, self.y
#
#     def distance_to_landmark(self):


def sd_lookup(ask_mean):
    if ask_mean >= 300:
        sd = 2.427
        return sd
    elif 300 > ask_mean >= 275:
        sd = 1.85417
        return sd
    elif 275 > ask_mean >= 250:
        sd = 1.434685
        return sd
    elif 250 > ask_mean >= 225:
        sd = 1.36659
        return sd
    elif 225 > ask_mean >= 200:
        sd = 1.426972
        return sd
    elif 200 > ask_mean >= 175:
        sd = 1.3
        return sd
    elif 175 > ask_mean >= 150:
        sd = 0.953124
        return sd
    elif 150 > ask_mean >= 125:
        sd = 1.233059
        return sd
    elif 125 > ask_mean >= 0:
        sd = 0.71
        return sd


def mouse_callback(event, x, y, flags, param):
    global x_global, y_global, seed_point

    if event == cv2.EVENT_LBUTTONDOWN:
        x_global = x
        y_global = y
        seed_point = (x_global, y_global)
        h, s, v = frame[seed_point[0], seed_point[1]]
        print "point is = " + str(seed_point) + " pixel = " + str(frame[seed_point[0], seed_point[1]])


def nothing(x):
    pass


def detect_in_range(color_min, color_max):
    mask = cv2.inRange(frame, color_min, color_max)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    return mask


def detect_landmark(color_min, color_max):
    # Algorithm To Detect the Contour
    # implemented here

    # Find the Landmark wrt to Color
    mask = detect_in_range(color_min, color_max)

    # Find the Contour to Estimate the bounding box (for pixel estimation)
    # cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    _, cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Cannot Find Contour => Bye GG
    if len(cnts) == 0:
        return 0, 0, 0, 0

    # Find Max
    cnt = max(cnts, key=cv2.contourArea)

    # Normal Bounding Box [OpenCV 2.4.X]
    # xx, yy, ww, hh = cv2.boundingRect(cnt)

    # Rotating Box [OpenCV 3+]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box
    # return xx, yy, ww, hh


def create_trackbar():  # trackbars
    # create trackbars for color change
    cv2.createTrackbar('1', 'output', 0, 255, nothing)
    cv2.createTrackbar('11', 'output', 0, 255, nothing)
    cv2.createTrackbar('2', 'output', 0, 255, nothing)
    cv2.createTrackbar('12', 'output', 0, 255, nothing)
    cv2.createTrackbar('3', 'output', 0, 255, nothing)
    cv2.createTrackbar('13', 'output', 0, 255, nothing)

    cv2.setTrackbarPos('1', 'output', landmark1_hsv_max[0])
    cv2.setTrackbarPos('11', 'output', landmark1_hsv_min[0])
    cv2.setTrackbarPos('2', 'output', landmark1_hsv_max[1])
    cv2.setTrackbarPos('12', 'output', landmark1_hsv_min[1])
    cv2.setTrackbarPos('3', 'output', landmark1_hsv_max[2])
    cv2.setTrackbarPos('13', 'output', landmark1_hsv_min[2])


def trackbar_threshold():
    global landmark1_hsv_max, landmark1_hsv_min
    landmark1_hsv_max = np.array([cv2.getTrackbarPos('1', 'output'),
                                  cv2.getTrackbarPos('2', 'output'), cv2.getTrackbarPos('3', 'output')])
    landmark1_hsv_min = np.array([cv2.getTrackbarPos('11', 'output'),
                                  cv2.getTrackbarPos('12', 'output'), cv2.getTrackbarPos('13', 'output')])


def midpoint(pt_a, pt_b):
    return (pt_a[0] + pt_b[0]) * 0.5, (pt_a[1] + pt_b[1]) * 0.5


def euclidean_dist((p_x, p_y), (q_x, q_y)):
    return math.sqrt(math.pow((p_x - q_x), 2) + math.pow(p_y - q_y, 2))


def calculate_box_area(box):
    # box = perspective.order_points(box)
    tl, tr, br, bl = box
    tltrX, tltrY = midpoint(tl, tr)
    blbrX, blbrY = midpoint(bl, br)
    tlblX, tlblY = midpoint(tl, bl)
    trbrX, trbrY = midpoint(tr, br)
    dA = euclidean_dist((tltrX, tltrY), (blbrX, blbrY))
    dB = euclidean_dist((tlblX, tlblY), (trbrX, trbrY))
    area = dA * dB
    return tltrX, tltrY, math.sqrt(area)


def calculate_box_middle_point(box):
    # box = perspective.order_points(box)
    tl, tr, br, bl = box
    middle_x = (tl[0] + tr[0] + br[0] + bl[0]) / 4
    middle_y = (tl[1] + tr[1] + br[1] + bl[1]) / 4

    return int(middle_x), int(middle_y)


def find_length_landmark(pixel_count):
    #  distance_mm = object_real_world_mm * focal-length_mm / object_image_sensor_mm
    distance = real_object_length * focal_length * px_per_mm / pixel_count
    return distance


def random_spawn_particle():
    return 0

def update_param(detected_distance_yellow,detected_distance_purple):
    global particle_x, particle_y, particles_weight
     # iterate through All Particle in index of particle(X[i],Y[i])
    total_probability = 0

    # Probability For Each Particle in this Round
    particle_probability = []
    acc_prob = []

    # Temp for the Next Gen Particle
    next_gen_particle_x = []
    next_gen_particle_y = []

    # Find the Probability For Each Particle Based on the Normal Distribution from sensor model
    for index in range(0, particle_amount):
        # find the distance from that particle

        particle_distance_yellow = euclidean_dist((particle_x[index], particle_y[index]), (40, 0))
        particle_distance_purple = euclidean_dist((particle_x[index], particle_y[index]), (-40, 0))

        sd_yellow = 10*sd_lookup(particle_distance_yellow)
        sd_purple = 10*sd_lookup(particle_distance_purple)
        # sd = 10

        distance_differential_yellow = particle_distance_yellow - detected_distance_yellow
        distance_differential_purple = particle_distance_purple - detected_distance_purple

        probability = 50*(1 / (sd_yellow * math.sqrt(2 * math.pi))) * np.exp(
            (-1) * (math.pow(distance_differential_yellow, 2) / (2 * math.pow(sd_yellow, 2))))

        probability *= 50*(1 / (sd_purple * math.sqrt(2 * math.pi))) * np.exp(
            (-1) * (math.pow(distance_differential_purple, 2) / (2 * math.pow(sd_purple, 2))))

        # probability = math.e ** -(distance_differential ** 2 / (2 * math.pow(sd, 2)))
        # Accumulate the probability

        total_probability += probability
        acc_prob.append(total_probability)

        # Append the probability by the index of the particles
        particle_probability.append(probability)

    # After we have the probability for each particle
    # Consider to update the original particle
    print total_probability

    for index in range(0, particle_amount):
        # Consider 10 Percent of Particle to be resampling with weight
        if index < particle_amount * 0.90:

            # Random these particles
            select_criteria = random.uniform(0, total_probability)  # np.random.rand()*total_probability

            # Iterate Through the accumulate probability
            for index_new in range(0, particle_amount):
                if acc_prob[index_new] >= select_criteria:
                    next_gen_particle_x.append(particle_x[index_new])
                    next_gen_particle_y.append(particle_y[index_new])
                    break
        # For the other particle (90%) Go on Randomly
        else:
            next_gen_particle_x.append(np.random.rand() * 500 - 250)
            next_gen_particle_y.append(np.random.rand() * 500)

    particle_x = next_gen_particle_x
    particle_y = next_gen_particle_y
    particles_weight = particle_probability


def main():
    global frame, landmark1_hsv_max, landmark1_hsv_min, particle_x, particle_y
    # Main Loop For Program
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('output')
    localization_state = False

    # Mathplotlib Setting
    plt.ion()
    plt.axis([-200, 200, 0, 500])

    # Define Mouse Callback
    cv2.setMouseCallback('output', mouse_callback)

    # create_trackbar()

    particle_x = [np.random.rand() * 500 - 250 for i in range(particle_amount)]
    particle_y = [np.random.rand() * 500 for i in range(particle_amount)]

    while True:
        # Capture frame-by-frame
        plt.axis([-200, 200, 0, 500])
        ret, frame = cap.read()
        original_image = frame
        frame = cv2.GaussianBlur(frame, (5, 5), 0)  # frame = cv2.medianBlur(frame, 15)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # trackbar_threshold()

        # DETECT LANDMARK 1 [HSV]
        mask_landmark1 = detect_in_range(landmark1_hsv_min, landmark1_hsv_max)
        box_landmark1 = detect_landmark(landmark1_hsv_min, landmark1_hsv_max)

        # DETECT LANDMARK 2 [HSV]
        mask_landmark2 = detect_in_range(landmark2_hsv_min, landmark2_hsv_max)
        box_landmark2 = detect_landmark(landmark2_hsv_min, landmark2_hsv_max)

        # DISPLAY LANDMARK 1 RESULT
        landmark1 = cv2.bitwise_and(original_image, original_image, mask=mask_landmark1)
        if box_landmark1 != (0, 0, 0, 0):
            # MEASURE LANDMARK 1
            xl1, yl1, size_landmark1 = calculate_box_area(box_landmark1)
            cv2.drawContours(original_image, [box_landmark1], -1, (0, 255, 0), 2)
            cv2.circle(original_image, calculate_box_middle_point(box_landmark1), 2, (0, 255, 0), 2)
            landmark1_length = find_length_landmark(size_landmark1)
            cv2.putText(original_image, str(landmark1_length), (int(xl1 - 15), int(yl1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255),
                        2)  # cv2.putText(original_image, str(size_landmark1), (int(xl1 - 15), int(yl1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            # Values For Particle Filtering
            distance_landmark1 = landmark1_length
        # DISPLAY LANDMARK 2 RESULT
        landmark2 = cv2.bitwise_and(original_image, original_image, mask=mask_landmark2)
        if box_landmark2 != (0, 0, 0, 0):
            # MEASURE LANDMARK 2
            xl2, yl2, size_landmark2 = calculate_box_area(box_landmark2)
            cv2.drawContours(original_image, [box_landmark2], -1, (255, 0, 0), 2)
            cv2.circle(original_image, calculate_box_middle_point(box_landmark2), 2, (0, 255, 0), 2)
            landmark2_length = find_length_landmark(size_landmark2)
            cv2.putText(original_image, str(landmark2_length), (int(xl2 - 15), int(yl2 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, (255, 255, 255),
                        2)  # cv2.putText(original_image, str(size_landmark2), (int(xl2 - 15), int(yl2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            distance_landmark2 = landmark2_length

        # Display Masking
        landmark_mask = cv2.bitwise_or(landmark1, landmark2)
        cv2.imshow('Landmark Mask', landmark_mask)

        # drawing mouse selecting point
        cv2.circle(original_image, seed_point, 3, (0, 255, 0), -1)
        # Display the resulting frame
        cv2.imshow('output', original_image)

        if localization_state:

            localization_state = False

            # Update Particle For Yellow Landmark[1]
            update_param(distance_landmark1,distance_landmark2)
            s = [particles_weight[n]*20**2 for n in range(len(particles_weight))]
            plt.scatter(particle_x, particle_y, s=s, c='r', alpha=0.5)

            fig = plt.figure(1)
            ax = fig.add_subplot(1, 1, 1)
            circle = plt.Circle((40, 0), radius=distance_landmark1, color='g', fill=False)
            circle2 = plt.Circle((-40, 0), radius=distance_landmark2, color='g', fill=False)
            ax.add_patch(circle)
            ax.add_patch(circle2)
            plt.show()

            plt.pause(0.001)
            plt.clf()


        # Key Listener
        k = cv2.waitKey(1) & 0xFF
        if k == 27 or k == ord('q'):
            break
        if k == ord('l'):
            if not localization_state:
                localization_state = True
                print "Localization Enable"
            else:
                localization_state = False
                plt.close()

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
