import cv2
import numpy as np

def auto_findContours(image, resize = 1, window_name = "Coins"):
    (_, cnts, _) = cv2.findContours(image.copy(), 0, cv2.CHAIN_APPROX_SIMPLE)

    # print("Find {} coins".format(len(cnts)))

    coins = image.copy()
    cv2.drawContours(coins, cnts, -1, (0, 255, 0), 5)
    cv2.imshow(window_name, cv2.resize(coins, (coins.shape[1] // resize, coins.shape[0] // resize)))
    for (i, c) in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        if(w < 30 or h < 30 or w > 200 or h > 200):
            continue
        print("Coin #{}".format(i + 1))
        coin = image[y:y + h, x:x + w]
        cv2.imshow("Coin", coin)

        mask = np.zeros(image.shape[:2], dtype="uint8")

        ((centerX, centerY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(mask, (int(centerX), int(centerY)), int(radius), 255, -1)
        mask = mask[y:y + h, x:x + w]
        cv2.imshow("Masked Coin", cv2.bitwise_and(coin, coin, mask=mask))
        cv2.waitKey(0)