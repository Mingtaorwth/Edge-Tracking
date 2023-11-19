import cv2 
import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm


class EdgeTracking():
    def __init__(self, img_dir1:str, img_dir2:str, img_dir3:str, feature_para, lk_params):
        self.img_dir1 = img_dir1
        self.img_dir2 = img_dir2
        self.img_dir3 = img_dir3
        self.feature_para = feature_para
        self.lk_params = lk_params
        self.trajectories = []

    def read_images(self): # Read images
        img1 = cv2.imread(self.img_dir1, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.img_dir2, cv2.IMREAD_GRAYSCALE)
        img_uncoded = cv2.imread(self.img_dir3, cv2.IMREAD_GRAYSCALE)
        return img1, img2, img_uncoded

    def color(self): # Random color for plots
        color1 = np.random.randint(0, 255)
        color2 = np.random.randint(0, 255)
        color3 = np.random.randint(0, 255)
        return color1, color2, color3

    def get_edge_dist(self, image): # Build edge set
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 100) # Canny edge get edges
        num_labels, labels = cv2.connectedComponents(edges) # Get edges so their motion could be estimate independently
        print("Number of Continueing edges:", num_labels-1)


        mask = np.zeros_like(image)
        points_set = {}

        for i in range(1, num_labels):
            mask += np.uint8(labels == i) * 255

            points_set[str(i)] = (np.argwhere(labels==i)) # The points on each edge

        return mask, points_set
    
    def lukas_kanade(self, img1, img2):
        flag_dict1 = {}
        mask1, pts_dict1 = self.get_edge_dist(img1)

        mask_good = np.zeros_like(img1, np.uint8)
        mask_good[:] = 255

        p0 = cv2.goodFeaturesToTrack(mask1, mask=mask_good, **self.feature_para).reshape(-1, 2) # Get Tomasi feature points
        p1, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **self.lk_params) # Lucas-Kanda tracking
        p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img2, img1, p1, None, **self.lk_params) # Inverse tracking to remove outliers
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        good = d < 1
        p0 = p0[good]
        p1 = p1[good] # Get good tracking key points
        for k, v in pts_dict1.items():
            list_kp = []
            
            for i, p in enumerate(p0): # Key points assignment
                if np.any(np.all(abs(v - np.array([p[1], p[0]])) < 2, axis=1)): # Make sure if the Euclidean distance is close to edge
                    list_kp.append(i)
                    np.delete(p0, i)
            flag_dict1[k] = list_kp
        
        flag_dict1 = {key: value for key, value in flag_dict1.items() if len(value) >= 3} # If one edge get at least 3 key points, put this edge in set
        return p0, p1, flag_dict1, pts_dict1
    
    def get_error(self, aff_mat, kp0, kp1): # Calculate Reprojection error of key points
        kp0_homo = np.column_stack((kp0, np.ones(len(kp0))))
        affined_kp0= np.dot(kp0_homo, aff_mat.reshape(2, 3).T)

        error = abs(affined_kp0 - kp1)
        return error.flatten()
    
    
    def evaluation_error(self, warped_edges, GT_edges): # calculate image difference
        return np.mean(cv2.absdiff(warped_edges, GT_edges)**2)
    
    def tracking(self, p0, p1, flag_dict, pts_dict, interpulation, predict): # Tracking edges
        aff_mat_dict = {}
        img1, img2, uncoded_img = self.read_images()
        temp = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR).astype(np.uint8) 
        affined_img = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8) # Empty image
        interpulation_img = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8) # Empty image
        predict_img = np.zeros((img1.shape[0], img1.shape[1]), dtype=np.uint8) # Empty image
        for k, v in flag_dict.items():
            pos0 = p0[flag_dict[k]] # Get key points on each edge
            pos0[:, [0, 1]] = pos0[:, [1, 0]] # openCV coordinate transition
            
            pos1 = p1[flag_dict[k]]
            pos1[:, [0, 1]] = pos1[:, [1, 0]]
            aff_mat_dict[k], _ = cv2.estimateAffine2D(pos0, pos1) # Estimate affine parameters
            pos0_homo = np.column_stack((pos0, np.ones(len(pos0)))) # Get the homogeneous affine transformation matrix (Good for dot product)
            if aff_mat_dict[k] is None:
                continue
            error = np.mean(np.linalg.norm(pos1-np.dot(pos0_homo, aff_mat_dict[k].T), axis=1))
            if error > 1: # If reprojection error larger than threshold, optimize with Levenberg–Marquardt method
                # print('original', error)
                corrected_aff_mat = least_squares(self.get_error, aff_mat_dict[k].flatten(), args=(pos0, pos1), method='lm')
                aff_mat_dict[k] = corrected_aff_mat.x.reshape(2, 3)
                # print('optimized', np.mean(np.linalg.norm(pos1-np.dot(pos0_homo, aff_mat_dict[k].T), axis=1)))
            
            # Use optimized affine matrix to warp edge
            affined_edge = cv2.transform(pts_dict[k].reshape(-1, 1, 2), aff_mat_dict[k].reshape(2, 3)).reshape(-1, 2)
            for p in pts_dict[k]:
                cv2.circle(temp, (int(p[1]), int(p[0])), 1, (200, 0, 0), -1)
            for p in affined_edge:
                cv2.circle(temp, (int(p[1]), int(p[0])), 1, (0, 200, 0), -1)
                if p[0] >= img1.shape[0] or p[1] >= img1.shape[1] or p[0] < 0 or p[1] < 0:
                    continue
                affined_img[p[0], p[1]] = 255

            if interpulation: # Interpolate edge frame
                mv = affined_edge - pts_dict[k]
                interpulated_edge = pts_dict[k] + mv/2 # Interpolate with half of the motion velocity

                for p in interpulated_edge:
                    if p[0] >= img1.shape[0] or p[1] >= img1.shape[1] or p[0] < 0 or p[1] < 0:
                        continue
                    cv2.circle(temp, (int(p[1]), int(p[0])), 1, (150, 100, 200), -1)
                    interpulation_img[int(p[0]), int(p[1])] = 255 
            
            if predict: # Predict edge frame
                mv = affined_edge - pts_dict[k]
                predict_edge = pts_dict[k] + 2 * mv # Predict with 2 times of the motion velocity

                for p in predict_edge:
                    if p[0] >= img1.shape[0] or p[1] >= img1.shape[1] or p[0] < 0 or p[1] < 0:
                        continue
                    cv2.circle(temp, (int(p[1]), int(p[0])), 1, (10, 100, 255), -1)
                    predict_img[int(p[0]), int(p[1])] = 255
                    cv2.imshow('my_result', predict_img)

        trajectories = [[pt0, pt1] for pt0, pt1 in zip(p0, p1)] # Trajectory of key points in 2 frames
        for p in trajectories: # Plot motion vector of key points
            cv2.arrowedLine(temp, tuple(p[0].astype(np.int32).tolist()), tuple(p[1].astype(np.int32).tolist()),
                            (100, 255, 100), 1, 8, 0)
        cv2.imshow('Vis_results', temp)
        cv2.imshow('affined', affined_img)

        return affined_img, temp

    def plot_kps(self, kps, axes):
        axes.plot(kps[:, 0], kps[:, 1], 'x')
    
    def __call__(self):
        img1, img2, img_uncoded = self.read_images()
        p0, p1, flag_dict, pts_dict1= self.lukas_kanade(img1, img2)

        warped_edges, temp = self.tracking(p0, p1, flag_dict, pts_dict1, interpulation=True, predict=False)
        blurred_GT = cv2.GaussianBlur(img_uncoded, (3, 3), 0)
        GT_edges = cv2.Canny(blurred_GT, 50, 100)
        cv2.imshow('GT_edges', GT_edges)
        ev_error = self.evaluation_error(warped_edges, GT_edges)
        cv2.waitKey(0)
        return ev_error

        
if __name__ == "__main__":
    feature_params = dict(maxCorners=8000,
                      qualityLevel=0.06,
                      minDistance=4,
                      blockSize=3)
    
    lk_params = dict(winSize=(7, 7),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    error_lis = []
    for t in tqdm(range(1, 32, 1)):
        pt1 = f"Color_images/BasketballDrill/42-8bit/img-{t-1}.png"
        pt2 = f"Color_images/BasketballDrill/42-8bit/img-{t}.png"
        pt3 = f"Color_images/BasketballDrill/42-8bit/img-{t}.png"
        edge_racking = EdgeTracking(pt1, pt2, pt3, feature_params, lk_params)
        edge_racking()

        
