# import the necessary packages
import cv2
import cv2.saliency as S

THRESHOLD = 0.5
MAX_VAL = 255


def compute_saliency_threshold(image, threshold=THRESHOLD, max_val=MAX_VAL):
	saliency = cv2.saliency.StaticSaliencyFineGrained_create()
	_, saliency_map = saliency.computeSaliency(image)
	saliency_map = (saliency_map * 255).astype("uint8")
	_, threshold = cv2.threshold(saliency_map, threshold, max_val, cv2.THRESH_BINARY)
	return threshold
