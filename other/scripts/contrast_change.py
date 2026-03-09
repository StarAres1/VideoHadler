import cv2
import numpy as np

def contrast_change1(image, alpha):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    distorted = (1 - alpha) * image.astype(np.float32) + alpha * gray_3ch.astype(np.float32)
    distorted = np.clip(distorted, 0, 255).astype(np.uint8)

    return distorted


def contrast_change(image, alpha):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Применяем контраст только к каналу яркости (L)
    l = cv2.convertScaleAbs(l, alpha=alpha, beta=0)

    lab = cv2.merge([l, a, b])
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return result

def gamma_transfer(image, gamma):
    normalized = image.astype(np.float32) / 255.0
    corrected = np.power(normalized, gamma)
    result = (corrected * 255).astype(np.uint8)

    return result


def cubic_distortion(image, a1, a2, a3, a4):
    x = image.astype(np.float32) / 255.0
    y = a1 * (x ** 3) + a2 * (x ** 2) + a3 * x + a4
    y = np.clip(y, 0, 1)
    return (y * 255).astype(np.uint8)


def logistic_distortion(image, p1, p2, p3, p4):
    """
    Применяет логистическое искажение к каналу L в LAB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l, a, b = cv2.split(lab)

    # Нормализуем L в [0, 1]
    l_norm = l / 255.0
    l_new = (p1 - p2) / (1 + np.exp(-(l_norm - p3) / p4)) + p2
    l_new = np.clip(l_new, 0, 1)
    l_new = (l_new * 255).astype(np.float32)

    lab_dist = cv2.merge([l_new, a, b])
    bgr_dist = cv2.cvtColor(lab_dist.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return bgr_dist


def mean_shift(image, delta):

    shifted = image.astype(np.int16) + delta
    shifted = np.clip(shifted, 0, 255).astype(np.uint8)
    return shifted


img = cv2.imread('../2.tiff')

logistic_params = [
    (1.0, 0.0, 0.6, 0.15),
    (0.9, 0.1, 0.5, 0.05)
]

cubic_params = [
    (2.0, -3.0, 2.0, 0.0),
    (-2.0, 3.0, 0.0, 0.0),
    (0.0, 0.0, 3.0, -0.5),
    (-3.0, 4.0, -1.0, 0.8)
]

# Использование всех 12 уровней
deltas = [-120, -100, -80, -60, -40, -20, 20, 40, 60, 80, 100, 120]
for d in deltas:
    degraded = mean_shift(img, d)
    cv2.imwrite(f'../result_contrast_change/meanshift_{d}.jpg', degraded)


"""
for i in [0.5, 0.75, 1.0, 1.25, 1.5]:
    distorted = contrast_change(img, i)
    cv2.imwrite(f'../result_contrast_change/contrast_change_alpha2_{i}.tiff', distorted)

for i in [0.5, 0.75]:
    distorted = contrast_change1(img, i)
    cv2.imwrite(f'../result_contrast_change/contrast_change_alphaNNN_{i}.tiff', distorted)

gammas = [1/5, 1/3, 1/2, 1/1.5, 1.5, 2, 3, 5]
for g in gammas:
    degraded = gamma_transfer(img, g)
    cv2.imwrite(f'../result_contrast_change/gamma_{g}.tiff', degraded)
    
for i, (a1, a2, a3, a4) in enumerate(cubic_params):
    degraded = cubic_distortion(img, a1, a2, a3, a4)
    cv2.imwrite(f'../result_contrast_change/cubic_level_{i+1}.jpg', degraded)
    
for i, (p1, p2, p3, p4) in enumerate(logistic_params):
    degraded = logistic_distortion(img, p1, p2, p3, p4)
    cv2.imwrite(f'../result_contrast_change/logistic_level_{i+1}.jpg', degraded)
"""

